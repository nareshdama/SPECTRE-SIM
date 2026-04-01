import numpy as np
import pandas as pd


class InjectionAttacker:
    """
    Adversarial false target injection attacker (2-channel).

    Attack model (ramp injection):
        delta_z(t) = I_dot * (t - t_start)   [scalar ramp magnitude]

    2-channel injection into bearing + range measurements:
        delta_bearing = delta_z * cos(theta_inj)          [rad]
        delta_range   = delta_z * range_scale * sin(theta_inj)  [m]

    This gives true 2D directional control: cos(theta_inj) controls
    cross-track deflection, sin(theta_inj) controls along-track
    deflection. Together they steer the miss vector direction.

    Kinematic consistency constraint:
        The implied target acceleration from the ramp must not exceed
        physical_accel_max at the reference engagement range.
    """

    def __init__(self, config: dict):
        self.config = config
        atk_cfg = config["attacker"]
        self.dt = config["simulation"]["dt"]

        self.injection_rate = float(atk_cfg["injection_rate"])
        self.injection_angle = float(
            np.radians(atk_cfg["injection_angle_deg"])
        )
        self.physical_accel_max = float(atk_cfg["physical_accel_max"])
        self.range_injection_scale = float(
            atk_cfg.get("range_injection_scale", 5000.0)
        )
        self.enabled = bool(atk_cfg["active"])

        # Approximate initial range for kinematic consistency check
        tx0 = config["target"]["x0"]
        ty0 = config["target"]["y0"]
        mx0 = config["missile"]["x0"]
        my0 = config["missile"]["y0"]
        self._r0 = np.sqrt((tx0 - mx0)**2 + (ty0 - my0)**2)

        self._validate_kinematic_consistency()

        self.t_start = None
        self._active = False
        self.history = []
        self.t_current = 0.0

    def _validate_kinematic_consistency(self) -> None:
        """
        Ensure injection ramp rate implies physically realizable
        target acceleration. Angular acceleration (rad/s^2) maps to
        translational acceleration (m/s^2) at range r:

        implied_accel = |I_dot| * r0
        """
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        implied_accel = abs(self.injection_rate) * self._r0
        if implied_accel > self.physical_accel_max:
            raise ValueError(
                f"Kinematic consistency violated: "
                f"implied acceleration {implied_accel:.2f} m/s^2 "
                f"(at range {self._r0:.0f} m) exceeds "
                f"physical_accel_max "
                f"{self.physical_accel_max:.2f} m/s^2. "
                f"Reduce injection_rate or increase "
                f"physical_accel_max in config."
            )

    def reset(self) -> None:
        self.t_start = None
        self._active = False
        self.history = []
        self.t_current = 0.0

    def activate(self, t_lock: float) -> None:
        if not self.enabled:
            return
        self.t_start = t_lock
        self._active = True

    def get_current_offset(self) -> float:
        if not self._active or self.t_start is None:
            return 0.0
        return self.injection_rate * (self.t_current - self.t_start)

    def compute_injection(self, t: float, z_true, ekf=None) -> np.ndarray:
        """
        Compute injected measurement for current timestep.

        Supports both scalar (legacy 1-channel) and vector
        (2-channel bearing+range) measurements.

        Args:
            t      : current simulation time [s]
            z_true : true measurement -- scalar or np.ndarray
            ekf    : optional; unused for ramp mode (optimized attackers pass EKF)

        Returns:
            z_injected: measurement seen by EKF (same shape as z_true)
        """
        _ = ekf
        self.t_current = t
        is_scalar = np.isscalar(z_true)
        z_vec = np.atleast_1d(np.asarray(z_true, dtype=float))

        if not self._active or not self.enabled:
            self._log(t, False, np.zeros_like(z_vec), z_vec, z_vec)
            return float(z_vec[0]) if is_scalar else z_vec.copy()

        delta_z = self.get_current_offset()

        if len(z_vec) >= 2:
            delta_bearing = delta_z * np.cos(self.injection_angle)
            delta_range = (delta_z * self.range_injection_scale
                           * np.sin(self.injection_angle))
            delta_vec = np.array([delta_bearing, delta_range])
        else:
            delta_vec = np.array([
                delta_z * np.cos(self.injection_angle)
            ])

        z_injected = z_vec + delta_vec

        self._log(t, True, delta_vec, z_vec, z_injected)
        if is_scalar:
            return float(z_injected[0])
        return z_injected

    def _log(self, t: float, active: bool, delta: np.ndarray,
             z_true: np.ndarray, z_injected: np.ndarray) -> None:
        record = {
            "t":            t,
            "active":       int(active),
            "delta_bearing": float(delta[0]),
            "theta_inj":    np.degrees(self.injection_angle),
            "z_true_bearing": float(z_true[0]),
            "z_injected_bearing": float(z_injected[0]),
        }
        if len(delta) >= 2:
            record["delta_range"] = float(delta[1])
            record["z_true_range"] = float(z_true[1])
            record["z_injected_range"] = float(z_injected[1])
        self.history.append(record)

    def is_active(self) -> bool:
        return self._active and self.enabled

    def export_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
