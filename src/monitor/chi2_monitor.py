import numpy as np
import pandas as pd
from scipy.stats import chi2 as scipy_chi2
from collections import deque


class Chi2InnovationMonitor:
    """
    Chi-squared innovation gate monitor for EKF bad-data detection.

    Decision rule:
        ALARM if: chi2_stat > chi2_threshold
        where:  chi2_threshold = chi2.ppf(1 - alpha, df=n_z)
                chi2_stat      = r_k^T * S_k^{-1} * r_k

    For scalar measurement (n_z=1):
        chi2_stat = innovation^2 / S_scalar
        threshold = chi2.ppf(0.95, df=1) = 3.8415

    Rolling window:
        Tracks detection rate over last `window_size` steps
        to capture burst-mode attack detection.
    """

    def __init__(self, config: dict):
        self.config = config
        mon_cfg = config["monitor"]

        self.alpha = float(mon_cfg["alpha"])
        self.n_z = int(mon_cfg["n_z"])
        self.window_size = int(mon_cfg.get("window_size", 10))

        # Compute threshold analytically from scipy
        self.threshold = float(
            scipy_chi2.ppf(1.0 - self.alpha, df=self.n_z)
        )

        # Runtime state
        self.history = []
        self.t_current = 0.0
        self.dt = config["simulation"]["dt"]
        self._total_checks = 0
        self._total_alarms = 0
        self._window = deque(maxlen=self.window_size)

    def reset(self) -> None:
        """Reset monitor to initial state."""
        self.history = []
        self.t_current = 0.0
        self._total_checks = 0
        self._total_alarms = 0
        self._window = deque(maxlen=self.window_size)

    def check(self, chi2_stat: float) -> bool:
        """
        Evaluate chi-squared statistic against threshold.

        Args:
            chi2_stat: EKF innovation chi-squared statistic (scalar)

        Returns:
            True  = ALARM (anomaly detected, possible attack)
            False = PASS  (measurement within expected bounds)
        """
        alarm = bool(chi2_stat > self.threshold)

        # Update counters
        self._total_checks += 1
        if alarm:
            self._total_alarms += 1
        self._window.append(int(alarm))

        # Log
        self.history.append({
            "t":              self.t_current,
            "chi2_stat":      chi2_stat,
            "threshold":      self.threshold,
            "alarm":          int(alarm),
            "rolling_rate":   self.get_rolling_detection_rate()
        })

        self.t_current += self.dt
        return alarm

    def get_threshold(self) -> float:
        """Return the analytical chi-squared threshold."""
        return self.threshold

    def get_detection_rate(self) -> float:
        """
        Return cumulative detection rate:
            total_alarms / total_checks
        Returns 0.0 if no checks performed yet.
        """
        if self._total_checks == 0:
            return 0.0
        return self._total_alarms / self._total_checks

    def get_rolling_detection_rate(self) -> float:
        """
        Return detection rate over the last `window_size` steps.
        Returns 0.0 if window is empty.
        """
        if len(self._window) == 0:
            return 0.0
        return sum(self._window) / len(self._window)

    def get_total_alarms(self) -> int:
        return self._total_alarms

    def get_total_checks(self) -> int:
        return self._total_checks

    def export_history(self) -> pd.DataFrame:
        """
        Export full monitor history as DataFrame.
        Columns: t, chi2_stat, threshold, alarm, rolling_rate
        """
        return pd.DataFrame(self.history)
