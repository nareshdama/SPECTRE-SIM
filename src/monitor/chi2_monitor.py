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

    Supports n_z = 1 (bearing-only) and n_z = 2 (bearing + range).

    Phase-aware detection rates:
        Tracks separate alarm counts for 'startup', 'tracking',
        and 'endgame' phases so the steady-state (tracking-phase)
        detection rate is not polluted by transient or endgame effects.
    """

    def __init__(self, config: dict):
        self.config = config
        mon_cfg = config["monitor"]

        self.alpha = float(mon_cfg["alpha"])
        self.n_z = int(mon_cfg["n_z"])
        self.window_size = int(mon_cfg.get("window_size", 10))

        self.threshold = float(
            scipy_chi2.ppf(1.0 - self.alpha, df=self.n_z)
        )

        self.history = []
        self.t_current = 0.0
        self.dt = config["simulation"]["dt"]
        self._total_checks = 0
        self._total_alarms = 0
        self._window = deque(maxlen=self.window_size)

        self._phase_checks = {"startup": 0, "tracking": 0, "endgame": 0}
        self._phase_alarms = {"startup": 0, "tracking": 0, "endgame": 0}

    def reset(self) -> None:
        self.history = []
        self.t_current = 0.0
        self._total_checks = 0
        self._total_alarms = 0
        self._window = deque(maxlen=self.window_size)
        self._phase_checks = {"startup": 0, "tracking": 0, "endgame": 0}
        self._phase_alarms = {"startup": 0, "tracking": 0, "endgame": 0}

    def check(self, chi2_stat: float, phase: str = "tracking") -> bool:
        """
        Evaluate chi-squared statistic against threshold.

        Args:
            chi2_stat: EKF innovation chi-squared statistic (scalar)
            phase: engagement phase -- 'startup', 'tracking', or 'endgame'

        Returns:
            True  = ALARM (anomaly detected)
            False = PASS  (within expected bounds)
        """
        alarm = bool(chi2_stat > self.threshold)

        self._total_checks += 1
        if alarm:
            self._total_alarms += 1
        self._window.append(int(alarm))

        if phase in self._phase_checks:
            self._phase_checks[phase] += 1
            if alarm:
                self._phase_alarms[phase] += 1

        self.history.append({
            "t":              self.t_current,
            "chi2_stat":      chi2_stat,
            "threshold":      self.threshold,
            "alarm":          int(alarm),
            "rolling_rate":   self.get_rolling_detection_rate(),
            "phase":          phase
        })

        self.t_current += self.dt
        return alarm

    def get_threshold(self) -> float:
        return self.threshold

    def get_detection_rate(self) -> float:
        """Cumulative detection rate over the entire engagement."""
        if self._total_checks == 0:
            return 0.0
        return self._total_alarms / self._total_checks

    def get_tracking_detection_rate(self) -> float:
        """
        Detection rate during the steady-state tracking phase only.
        Excludes startup transients and endgame range-collapse effects.
        """
        checks = self._phase_checks.get("tracking", 0)
        if checks == 0:
            return 0.0
        return self._phase_alarms["tracking"] / checks

    def get_rolling_detection_rate(self) -> float:
        if len(self._window) == 0:
            return 0.0
        return sum(self._window) / len(self._window)

    def get_total_alarms(self) -> int:
        return self._total_alarms

    def get_total_checks(self) -> int:
        return self._total_checks

    def export_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
