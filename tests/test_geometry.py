import pytest
import numpy as np
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engagement.geometry import EngagementGeometry


@pytest.fixture
def config():
    with open("config/sim_config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def geo(config):
    return EngagementGeometry(config)


def test_intercept_within_35_seconds(geo):
    """Head-on geometry must reach intercept within 35 seconds."""
    # Run with zero guidance (free flight) to check geometry closes
    # Re-run with a simple proportional command
    geo.reset()
    for _ in range(3500):  # 35 seconds at dt=0.01
        # Simple proportional command to steer toward target
        los, los_rate = geo.compute_los()
        Vc = geo.compute_closing_velocity()
        accel = 4 * Vc * los_rate
        accel = np.clip(accel, -300, 300)
        geo.step(accel)
        if geo._intercepted:
            break
    assert geo._intercepted, (
        f"Expected intercept within 35s, got t={geo.t_current:.2f}s, "
        f"min_range={geo._min_range:.2f}m"
    )


def test_los_rate_nonzero_at_t0(geo):
    """LOS rate at t=0 must be nonzero (non-trivial engagement geometry)."""
    geo.reset()
    _, los_rate = geo.compute_los()
    assert los_rate != 0.0, "LOS rate at t=0 should not be zero"
    # With target at (10000, 500) above missile at origin, and vrel = (-500, 0),
    # los_rate = (dx*dvy - dy*dvx) / r^2 = (0 - 500*(-500)) / r^2 > 0.
    assert los_rate > 0.0, (
        f"Expected positive LOS rate for target above missile, got {los_rate:.6f}"
    )


def test_export_history_columns(geo, config):
    """export_history() must return DataFrame with exactly 13 columns."""
    geo.reset()
    for _ in range(10):
        geo.step(0.0)
    df = geo.export_history()
    expected_cols = [
        "t", "mx", "my", "mvx", "mvy",
        "tx", "ty", "tvx", "tvy",
        "los", "los_rate", "range", "Vc"
    ]
    assert list(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {list(df.columns)}"
    )
    assert len(df.columns) == 13, f"Expected 13 columns, got {len(df.columns)}"


def test_miss_distance_without_guidance(geo):
    """Without guidance (zero accel), miss distance must exceed 400m."""
    geo.reset()
    for _ in range(6000):
        geo.step(0.0)
        if geo.is_intercept():
            break
    miss = geo.get_miss_distance()
    assert miss > 400.0, (
        f"Expected miss > 400m without guidance, got {miss:.2f}m"
    )


def test_rk4_stability(geo):
    """
    RK4 stability: state norm change per step must not exceed 1.5 * dt * max_accel.
    Checks integration is stable and not diverging.
    """
    geo.reset()
    prev_state = np.concatenate([geo.missile, geo.target])
    max_jump = 0.0
    dt = geo.dt
    for _ in range(100):
        geo.step(50.0)  # Moderate constant command
        curr_state = np.concatenate([geo.missile, geo.target])
        jump = np.linalg.norm(curr_state - prev_state)
        max_jump = max(max_jump, jump)
        prev_state = curr_state.copy()
    # At dt=0.01, with v~300 m/s and a~50 m/s^2:
    # expected jump ~= sqrt((300*0.01)^2 + (200*0.01)^2) ~ 3.6
    # bound: 1.5 * max_speed * dt = 1.5 * 500 * 0.01 = 7.5
    bound = 1.5 * 500 * dt
    assert max_jump < bound, (
        f"RK4 instability detected: max state jump {max_jump:.4f} "
        f"exceeds bound {bound:.4f}"
    )
