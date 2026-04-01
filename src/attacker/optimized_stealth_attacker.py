"""
Optimization-based stealthy false-data injection (2-channel).

Each step after activation, chooses an increment du in measurement space
to maximize alignment with a fixed direction (injection angle) subject to:

    (r0 + cum_du + du)^T S^{-1} (r0 + cum_du + du) <= tau_safe

where r0 is the innovation from the noisy truth measurement (no attack yet)
and (S, r0) are computed from the EKF **after predict()** and **before
update()**. This is a standard constrained stealth FDI formulation in
innovation space, instantiated here for bearing+range seekers.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2 as scipy_chi2
from scipy.optimize import minimize


class OptimizedStealthAttacker:
    def __init__(self, config: dict):
        self.config = config
        atk_cfg = config["attacker"]
        opt = atk_cfg.get("optimized", {})
        mon_cfg = config["monitor"]

        self.dt = config["simulation"]["dt"]
        self.enabled = bool(atk_cfg["active"])
        self.injection_angle = float(
            np.radians(atk_cfg["injection_angle_deg"])
        )
        self.range_injection_scale = float(
            atk_cfg.get("range_injection_scale", 5000.0)
        )
        self.physical_accel_max = float(atk_cfg["physical_accel_max"])

        tx0 = config["target"]["x0"]
        ty0 = config["target"]["y0"]
        mx0 = config["missile"]["x0"]
        my0 = config["missile"]["y0"]
        self._r0 = np.sqrt((tx0 - mx0) ** 2 + (ty0 - my0) ** 2)

        self.chi2_margin = float(opt.get("chi2_margin", 0.92))
        self.du_max_bearing = float(
            opt.get("du_max_bearing", 5.0e-5)
        )
        self.du_max_range = float(opt.get("du_max_range", 15.0))

        n_z = int(mon_cfg["n_z"])
        alpha = float(mon_cfg["alpha"])
        self._chi2_threshold = float(
            scipy_chi2.ppf(1.0 - alpha, df=n_z)
        )
        self._tau_safe = self.chi2_margin * self._chi2_threshold

        inj_rate = float(atk_cfg.get("injection_rate", 0.01))
        implied_accel = abs(inj_rate) * self._r0
        if implied_accel > self.physical_accel_max:
            raise ValueError(
                "Kinematic consistency violated for optimized attacker "
                "(injection_rate vs physical_accel_max)."
            )

        self.t_start = None
        self._active = False
        self._cum_du = np.zeros(2)
        self.history = []
        self.t_current = 0.0

        c_b = np.cos(self.injection_angle)
        c_r = self.range_injection_scale * np.sin(self.injection_angle)
        c = np.array([c_b, c_r], dtype=float)
        self._c = c / (np.linalg.norm(c) + 1e-12)

    def reset(self) -> None:
        self.t_start = None
        self._active = False
        self._cum_du = np.zeros(2)
        self.history = []
        self.t_current = 0.0

    def activate(self, t_lock: float) -> None:
        if not self.enabled:
            return
        self.t_start = t_lock
        self._active = True

    def is_active(self) -> bool:
        return self._active and self.enabled

    @staticmethod
    def _mahalanobis_sq(r: np.ndarray, S: np.ndarray) -> float:
        S_inv = np.linalg.inv(S)
        return float(r @ S_inv @ r)

    def _solve_du(self, r_base: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Maximize c^T du s.t. Mahalanobis(r_base + du) <= tau_safe and box bounds."""
        if self._mahalanobis_sq(r_base, S) >= self._tau_safe - 1e-9:
            return np.zeros(2)

        c = self._c
        bounds = [
            (-self.du_max_bearing, self.du_max_bearing),
            (-self.du_max_range, self.du_max_range),
        ]

        def chi2_total(du: np.ndarray) -> float:
            r = r_base + du
            return self._mahalanobis_sq(r, S)

        cons = (
            {
                "type": "ineq",
                "fun": lambda du: self._tau_safe - chi2_total(du),
            },
        )

        res = minimize(
            lambda du: -float(c @ du),
            x0=np.zeros(2),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 80, "ftol": 1e-9},
        )
        du = np.asarray(res.x, dtype=float)
        if not res.success or chi2_total(du) > self._tau_safe + 1e-6:
            return np.zeros(2)
        return du

    def compute_injection(
        self,
        t: float,
        z_true,
        ekf=None,
    ) -> np.ndarray:
        self.t_current = t
        is_scalar = np.isscalar(z_true)
        z_vec = np.atleast_1d(np.asarray(z_true, dtype=float))

        if not self._active or not self.enabled:
            self._log(t, False, np.zeros_like(z_vec), z_vec, z_vec, 0.0)
            return float(z_vec[0]) if is_scalar else z_vec.copy()

        if ekf is None:
            raise ValueError(
                "OptimizedStealthAttacker requires ekf= after predict()."
            )
        if len(z_vec) < 2:
            raise ValueError("Optimized attacker requires 2-channel z.")

        r0, S, _ = ekf.innovation_statistics(z_vec)
        r_under_attack = r0 + self._cum_du
        du = self._solve_du(r_under_attack, S)
        self._cum_du = self._cum_du + du

        delta_vec = self._cum_du.copy()
        z_injected = z_vec + delta_vec

        chi2_after = self._mahalanobis_sq(
            r0 + self._cum_du, S
        )
        self._log(
            t, True, delta_vec, z_vec, z_injected, chi2_after
        )
        if is_scalar:
            return float(z_injected[0])
        return z_injected

    def _log(
        self,
        t: float,
        active: bool,
        delta: np.ndarray,
        z_true: np.ndarray,
        z_injected: np.ndarray,
        chi2_pred: float,
    ) -> None:
        record = {
            "t": t,
            "active": int(active),
            "delta_bearing": float(delta[0]),
            "delta_range": float(delta[1]) if len(delta) >= 2 else 0.0,
            "theta_inj": np.degrees(self.injection_angle),
            "z_true_bearing": float(z_true[0]),
            "z_injected_bearing": float(z_injected[0]),
            "chi2_under_attack_pred": chi2_pred,
        }
        if len(z_true) >= 2:
            record["z_true_range"] = float(z_true[1])
            record["z_injected_range"] = float(z_injected[1])
        self.history.append(record)

    def export_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
