import numpy as np
import pandas as pd


class PNGuidance:
    """
    Proportional Navigation guidance law.

    Guidance command:
        a_c = N * Vc * lambda_dot_hat

    where:
        N             : navigation constant (dimensionless, typically 3-5)
        Vc            : closing velocity (m/s, positive = closing)
        lambda_dot_hat: EKF-estimated LOS rate (rad/s)

    Saturation:
        |a_c| is clipped to a_max (m/s^2)
    """

    def __init__(self, config: dict):
        self.config = config
        guidance_cfg = config["guidance"]

        self.N = float(guidance_cfg["N"])
        self.a_max = float(guidance_cfg["a_max"])

        # Runtime state
        self.t_current = 0.0
        self.dt = config["simulation"]["dt"]
        self.history = []
        self._clip_count = 0

    def reset(self) -> None:
        self.t_current = 0.0
        self.history = []
        self._clip_count = 0

    def compute_command(self, los_rate_hat: float, Vc: float) -> float:
        """
        Compute PN guidance acceleration command.

        Args:
            los_rate_hat : EKF-estimated LOS rate [rad/s]
            Vc           : closing velocity [m/s]

        Returns:
            Clipped lateral acceleration command [m/s^2]
        """
        a_raw = self.N * Vc * los_rate_hat
        a_clipped = self.clip_command(a_raw)

        self.history.append({
            "t":            self.t_current,
            "los_rate_hat": los_rate_hat,
            "Vc":           Vc,
            "a_cmd_raw":    a_raw,
            "a_cmd":        a_clipped,
            "clipped":      int(abs(a_raw) > self.a_max)
        })

        self.t_current += self.dt
        return a_clipped

    def clip_command(self, a_cmd: float) -> float:
        """
        Apply saturation limit to acceleration command.
        Preserves sign of original command.
        """
        if abs(a_cmd) > self.a_max:
            self._clip_count += 1
            return float(np.sign(a_cmd) * self.a_max)
        return float(a_cmd)

    def get_clip_count(self) -> int:
        """Returns total number of saturation events."""
        return self._clip_count

    def export_history(self) -> pd.DataFrame:
        """
        Export full guidance history as DataFrame.
        Columns: t, los_rate_hat, Vc, a_cmd_raw, a_cmd, clipped
        """
        return pd.DataFrame(self.history)

