import numpy as np
import pandas as pd


class EKFSeeker:
    """
    Extended Kalman Filter for missile seeker state estimation.

    State vector: x_hat = [r_x, r_y, v_rx, v_ry]
      r_x, r_y   : relative position (target - missile) in meters
      v_rx, v_ry : relative velocity in m/s

    Measurement (2-channel active radar seeker):
      z = [arctan2(r_y, r_x),          # bearing [rad]
           sqrt(r_x^2 + r_y^2)]        # range   [m]
    """

    def __init__(self, config: dict):
        self.config = config
        self.dt = config["simulation"]["dt"]
        ekf_cfg = config["ekf"]

        Q_diag = ekf_cfg["Q_diag"]
        self.Q = np.diag(Q_diag).astype(float)

        # Support both legacy scalar and new 2-channel config
        if "R_diag" in ekf_cfg:
            self.R = np.diag(ekf_cfg["R_diag"]).astype(float)
        else:
            self.R = np.array([[ekf_cfg["R_scalar"]]]).astype(float)

        self.n_z = self.R.shape[0]

        self.lock_threshold = ekf_cfg["lock_threshold"]
        self.t_acquisition = ekf_cfg["t_acquisition"]

        self.x_hat = np.zeros(4)
        self.P = np.eye(4) * 1.0

        self.chi2_stat = 0.0
        self.innovation = np.zeros(self.n_z)
        self.S = np.eye(self.n_z)
        self.K = np.zeros((4, self.n_z))
        self.gain_history = []
        self.t_current = 0.0
        self._locked = False

    def reset(self, x0_relative: np.ndarray) -> None:
        self.x_hat = x0_relative.astype(float).copy()
        self.P = np.eye(4) * 1.0
        self.chi2_stat = 0.0
        self.innovation = np.zeros(self.n_z)
        self.S = np.eye(self.n_z)
        self.K = np.zeros((4, self.n_z))
        self.gain_history = []
        self.t_current = 0.0
        self._locked = False

    def _state_transition(self) -> np.ndarray:
        dt = self.dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)
        return F

    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h(x) w.r.t. state x.

        For 2-channel (bearing + range):
          H = [[-r_y/r^2,  r_x/r^2, 0, 0],   # d(bearing)/d(state)
               [ r_x/r,    r_y/r,   0, 0]]    # d(range)/d(state)

        For legacy 1-channel (bearing only):
          H = [[-r_y/r^2,  r_x/r^2, 0, 0]]
        """
        r_x, r_y = x[0], x[1]
        r2 = r_x**2 + r_y**2 + 1e-9
        r = np.sqrt(r2)

        if self.n_z == 2:
            H = np.array([
                [-r_y / r2, r_x / r2, 0.0, 0.0],
                [r_x / r,   r_y / r,  0.0, 0.0]
            ])
        else:
            H = np.array([[-r_y / r2, r_x / r2, 0.0, 0.0]])
        return H

    def _h(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear measurement function."""
        bearing = np.arctan2(x[1], x[0])
        if self.n_z == 2:
            rng = np.sqrt(x[0]**2 + x[1]**2 + 1e-9)
            return np.array([bearing, rng])
        return np.array([bearing])

    def _normalize_angle(self, angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self) -> None:
        F = self._state_transition()
        self.x_hat = F @ self.x_hat
        self.P = F @ self.P @ F.T + self.Q
        self.t_current += self.dt

    def innovation_statistics(self, z_meas) -> tuple:
        """
        Innovation residual and covariance **after predict(), before update()**.

        Used by optimization-based attackers that choose additive injection
        in measurement space subject to a Mahalanobis constraint.

        Returns:
            innov: innovation vector r = z - h(x_hat_{k|k-1})
            S:     innovation covariance H P H^T + R
            chi2:  r^T S^{-1} r
        """
        if np.isscalar(z_meas):
            z = np.array([z_meas])
        else:
            z = np.asarray(z_meas, dtype=float)

        H = self._measurement_jacobian(self.x_hat)
        z_pred = self._h(self.x_hat)
        innov = z - z_pred
        innov[0] = self._normalize_angle(innov[0])
        S = H @ self.P @ H.T + self.R
        S_inv = np.linalg.inv(S)
        chi2 = float(innov @ S_inv @ innov)
        return innov.copy(), S.copy(), chi2

    def update(self, z_meas) -> dict:
        """
        EKF update step: incorporate measurement (scalar or vector).

        Args:
            z_meas: measurement -- scalar float (legacy 1-channel)
                     or np.ndarray of length n_z (2-channel)

        Returns:
            dict with innovation, chi2_stat, S, K, x_hat_updated
        """
        if np.isscalar(z_meas):
            z = np.array([z_meas])
        else:
            z = np.asarray(z_meas, dtype=float)

        H = self._measurement_jacobian(self.x_hat)
        z_pred = self._h(self.x_hat)

        innov = z - z_pred
        innov[0] = self._normalize_angle(innov[0])
        self.innovation = innov

        self.S = H @ self.P @ H.T + self.R
        self.K = self.P @ H.T @ np.linalg.inv(self.S)

        self.x_hat = self.x_hat + self.K @ innov

        # Joseph form covariance update
        I = np.eye(4)
        IKH = I - self.K @ H
        self.P = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T

        # Chi-squared: innov^T S^{-1} innov (scalar result)
        S_inv = np.linalg.inv(self.S)
        self.chi2_stat = float(innov @ S_inv @ innov)

        if self.t_current >= self.t_acquisition:
            if np.trace(self.P) < self.lock_threshold:
                self._locked = True

        gain_norm = np.linalg.norm(self.K, ord='fro')
        est_range = float(np.sqrt(
            self.x_hat[0]**2 + self.x_hat[1]**2 + 1e-9
        ))
        self.gain_history.append({
            "t": self.t_current,
            "gain_norm": gain_norm,
            "range": est_range,
            "P_trace": np.trace(self.P),
            "chi2_stat": self.chi2_stat,
            "innovation": float(innov[0]),
            "locked": self._locked
        })

        return {
            "innovation": innov,
            "chi2_stat": self.chi2_stat,
            "S": float(self.S[0, 0]) if self.n_z == 1 else self.S,
            "K_norm": gain_norm,
            "x_hat": self.x_hat.copy(),
            "P_trace": np.trace(self.P),
            "locked": self._locked
        }

    def get_los_rate_estimate(self) -> float:
        r_x, r_y = self.x_hat[0], self.x_hat[1]
        v_rx, v_ry = self.x_hat[2], self.x_hat[3]
        r2 = r_x**2 + r_y**2 + 1e-9
        return (r_x * v_ry - r_y * v_rx) / r2

    def is_locked(self) -> bool:
        return self._locked

    def get_gain_norm(self) -> float:
        if self.gain_history:
            return self.gain_history[-1]["gain_norm"]
        return 0.0

    def export_gain_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.gain_history)
