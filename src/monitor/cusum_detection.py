"""
CUSUM-style cumulative test on chi-squared innovation statistics.

Used as an additional baseline detector in simulation summaries.
"""

from __future__ import annotations

import pandas as pd


def cusum_alarm_rate(
    chi2_values,
    threshold: float,
    drift: float = 0.5,
) -> float:
    """
    One-sided CUSUM on the chi-squared sequence.

    S_k = max(0, S_{k-1} + (chi2_k - drift * threshold))
    Alarm when S_k > threshold; counter resets after alarm.
    """
    vals = list(chi2_values)
    if len(vals) == 0:
        return 0.0
    S = 0.0
    h = float(threshold)
    alarms = 0
    for chi2_k in vals:
        S = max(0.0, S + float(chi2_k) - drift * h)
        if S > h:
            alarms += 1
            S = 0.0
    return alarms / len(vals)


def cusum_tracking_alarm_rate(
    monitor_df: pd.DataFrame,
    drift: float = 0.5,
) -> float:
    """CUSUM alarm rate using only rows with phase == 'tracking'."""
    if monitor_df is None or monitor_df.empty:
        return 0.0
    if "phase" not in monitor_df.columns:
        return cusum_alarm_rate(
            monitor_df["chi2_stat"].values,
            float(monitor_df["threshold"].iloc[0]),
            drift=drift,
        )
    sub = monitor_df[monitor_df["phase"] == "tracking"]
    if sub.empty:
        return 0.0
    return cusum_alarm_rate(
        sub["chi2_stat"].values,
        float(sub["threshold"].iloc[0]),
        drift=drift,
    )
