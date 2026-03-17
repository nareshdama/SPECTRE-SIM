import pytest
import numpy as np
import yaml
import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.monitor.chi2_monitor import Chi2InnovationMonitor


@pytest.fixture
def config():
    with open("config/sim_config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def monitor(config):
    return Chi2InnovationMonitor(config)


def test_threshold_value(monitor):
    """
    Threshold for alpha=0.05, n_z=1 must equal 3.8415 +/- 0.001.
    This is a fundamental statistical constant — any deviation
    indicates incorrect ppf call arguments.
    """
    threshold = monitor.get_threshold()
    assert abs(threshold - 3.8415) < 0.001, (
        f"Expected threshold ~3.8415, got {threshold:.6f}. "
        f"Verify: scipy.stats.chi2.ppf(0.95, df=1)"
    )


def test_below_threshold_no_alarm(monitor):
    """
    chi2_stat = 3.0 (below 3.8415) must return False (no alarm).
    """
    monitor.reset()
    result = monitor.check(chi2_stat=3.0)
    assert result is False, (
        f"Expected False (no alarm) for chi2=3.0 < threshold, "
        f"got {result}"
    )


def test_above_threshold_alarm(monitor):
    """
    chi2_stat = 5.0 (above 3.8415) must return True (alarm).
    """
    monitor.reset()
    result = monitor.check(chi2_stat=5.0)
    assert result is True, (
        f"Expected True (alarm) for chi2=5.0 > threshold, "
        f"got {result}"
    )


def test_clean_gaussian_false_alarm_rate(config, monitor):
    """
    1000 clean chi-squared samples (df=1) must produce detection
    rate between 0.03 and 0.07 (alpha=0.05 +/- 2 sigma margin).
    This validates the monitor is correctly calibrated.
    """
    from scipy.stats import chi2 as scipy_chi2
    monitor.reset()
    np.random.seed(42)

    n_samples = 1000
    # Draw samples from true chi2(df=1) distribution
    samples = scipy_chi2.rvs(df=1, size=n_samples, random_state=42)

    for s in samples:
        monitor.check(float(s))

    rate = monitor.get_detection_rate()
    assert 0.03 <= rate <= 0.07, (
        f"False alarm rate {rate:.4f} outside [0.03, 0.07] "
        f"for 1000 chi2(df=1) samples. "
        f"Expected ~0.05 (alpha=0.05)."
    )


def test_export_history_columns(monitor):
    """
    export_history() must return DataFrame with exactly 5 columns
    in the correct order.
    """
    monitor.reset()
    for val in [1.0, 2.5, 4.0, 6.0, 0.5]:
        monitor.check(val)

    df = monitor.export_history()
    expected_cols = [
        "t", "chi2_stat", "threshold", "alarm", "rolling_rate"
    ]
    assert list(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {list(df.columns)}"
    )
    assert len(df) == 5, (
        f"Expected 5 rows, got {len(df)}"
    )


def test_rolling_window_detection_rate(config, monitor):
    """
    Rolling window rate must reflect only the last window_size steps.
    After 10 alarms followed by 10 clean steps,
    rolling rate must be 0.0 (clean window).
    """
    monitor.reset()
    window_size = config["monitor"]["window_size"]  # 10

    # 10 large values → all alarm
    for _ in range(window_size):
        monitor.check(10.0)

    # 10 small values → all clean
    for _ in range(window_size):
        monitor.check(1.0)

    rolling = monitor.get_rolling_detection_rate()
    assert rolling == 0.0, (
        f"Rolling rate should be 0.0 after {window_size} clean steps, "
        f"got {rolling:.4f}"
    )


def test_cumulative_counts_accurate(monitor):
    """
    Total alarms and total checks must be exactly tracked.
    """
    monitor.reset()
    # 3 alarms (>3.8415), 2 clean (<3.8415)
    monitor.check(5.0)   # alarm
    monitor.check(1.0)   # clean
    monitor.check(6.0)   # alarm
    monitor.check(2.0)   # clean
    monitor.check(4.5)   # alarm

    assert monitor.get_total_checks() == 5, (
        f"Expected 5 total checks, got {monitor.get_total_checks()}"
    )
    assert monitor.get_total_alarms() == 3, (
        f"Expected 3 total alarms, got {monitor.get_total_alarms()}"
    )
    assert monitor.get_detection_rate() == pytest.approx(
        3/5, abs=1e-9
    ), (
        f"Expected detection rate 0.6, got "
        f"{monitor.get_detection_rate():.6f}"
    )


def test_reset_clears_all_state(monitor):
    """
    reset() must zero out all counters, history, and window.
    """
    monitor.reset()
    for val in [5.0, 6.0, 7.0]:
        monitor.check(val)

    monitor.reset()  # Reset

    assert monitor.get_total_checks() == 0
    assert monitor.get_total_alarms() == 0
    assert monitor.get_detection_rate() == 0.0
    assert monitor.get_rolling_detection_rate() == 0.0
    assert len(monitor.export_history()) == 0, (
        "History must be empty after reset()"
    )
