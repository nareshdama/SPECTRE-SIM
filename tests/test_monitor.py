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


def test_threshold_value_nz2(monitor):
    """Threshold for alpha=0.05, n_z=2 must equal 5.9915 +/- 0.001."""
    threshold = monitor.get_threshold()
    assert abs(threshold - 5.9915) < 0.001, (
        f"Expected threshold ~5.9915 (n_z=2), got {threshold:.6f}"
    )


def test_monitor_nz_is_2(monitor):
    """Monitor n_z must be 2 for 2-channel EKF."""
    assert monitor.n_z == 2


def test_below_threshold_no_alarm(monitor):
    """chi2_stat = 4.0 (below 5.9915) must return False."""
    monitor.reset()
    result = monitor.check(chi2_stat=4.0)
    assert result is False


def test_above_threshold_alarm(monitor):
    """chi2_stat = 7.0 (above 5.9915) must return True."""
    monitor.reset()
    result = monitor.check(chi2_stat=7.0)
    assert result is True


def test_clean_gaussian_false_alarm_rate(config, monitor):
    """
    1000 clean chi2(df=2) samples must produce detection rate
    between 0.03 and 0.08.
    """
    from scipy.stats import chi2 as scipy_chi2
    monitor.reset()

    samples = scipy_chi2.rvs(df=2, size=1000, random_state=42)
    for s in samples:
        monitor.check(float(s))

    rate = monitor.get_detection_rate()
    assert 0.03 <= rate <= 0.08, (
        f"False alarm rate {rate:.4f} outside [0.03, 0.08]"
    )


def test_phase_tracking_rate(monitor):
    """
    Tracking-phase detection rate must only count tracking checks.
    """
    monitor.reset()
    monitor.check(10.0, phase="startup")
    monitor.check(10.0, phase="startup")
    monitor.check(1.0, phase="tracking")
    monitor.check(1.0, phase="tracking")
    monitor.check(10.0, phase="endgame")

    tracking_rate = monitor.get_tracking_detection_rate()
    assert tracking_rate == 0.0, (
        f"Expected 0.0 tracking rate (no alarms in tracking), "
        f"got {tracking_rate}"
    )


def test_phase_tracking_rate_with_alarms(monitor):
    """When alarm occurs during tracking, it counts."""
    monitor.reset()
    monitor.check(1.0, phase="startup")
    monitor.check(10.0, phase="tracking")
    monitor.check(1.0, phase="tracking")

    tracking_rate = monitor.get_tracking_detection_rate()
    assert tracking_rate == pytest.approx(0.5, abs=1e-9)


def test_export_history_has_phase(monitor):
    """History DataFrame must include phase column."""
    monitor.reset()
    monitor.check(1.0, phase="tracking")
    monitor.check(10.0, phase="endgame")

    df = monitor.export_history()
    assert "phase" in df.columns
    assert df.iloc[0]["phase"] == "tracking"
    assert df.iloc[1]["phase"] == "endgame"


def test_rolling_window(config, monitor):
    """Rolling rate must reflect only last window_size steps."""
    monitor.reset()
    window_size = config["monitor"]["window_size"]

    for _ in range(window_size):
        monitor.check(10.0)
    for _ in range(window_size):
        monitor.check(1.0)

    rolling = monitor.get_rolling_detection_rate()
    assert rolling == 0.0


def test_cumulative_counts_accurate(monitor):
    """Total alarms and checks must be tracked exactly."""
    monitor.reset()
    monitor.check(7.0)   # alarm (> 5.99)
    monitor.check(1.0)   # clean
    monitor.check(8.0)   # alarm
    monitor.check(2.0)   # clean
    monitor.check(6.5)   # alarm

    assert monitor.get_total_checks() == 5
    assert monitor.get_total_alarms() == 3
    assert monitor.get_detection_rate() == pytest.approx(0.6, abs=1e-9)


def test_reset_clears_all_state(monitor):
    """reset() must zero out all counters and history."""
    monitor.reset()
    for val in [7.0, 8.0, 9.0]:
        monitor.check(val)

    monitor.reset()

    assert monitor.get_total_checks() == 0
    assert monitor.get_total_alarms() == 0
    assert monitor.get_detection_rate() == 0.0
    assert monitor.get_tracking_detection_rate() == 0.0
    assert len(monitor.export_history()) == 0
