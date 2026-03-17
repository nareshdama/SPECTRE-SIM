import pytest
import numpy as np
import yaml
import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.guidance.pn_guidance import PNGuidance


@pytest.fixture
def config():
    with open("config/sim_config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def guidance(config):
    return PNGuidance(config)


def test_zero_los_rate_gives_zero_command(guidance):
    """
    Zero LOS rate input must produce exactly zero command,
    regardless of closing velocity.
    """
    guidance.reset()
    cmd = guidance.compute_command(los_rate_hat=0.0, Vc=350.0)
    assert cmd == 0.0, (
        f"Expected 0.0 for zero LOS rate, got {cmd}"
    )


def test_saturation_clips_to_a_max(config, guidance):
    """
    Command that exceeds a_max must be clipped to exactly +/- a_max.
    Sign must be preserved.
    """
    guidance.reset()
    a_max = config["guidance"]["a_max"]

    # Positive saturation
    cmd_pos = guidance.compute_command(los_rate_hat=10.0, Vc=500.0)
    assert cmd_pos == pytest.approx(a_max, abs=1e-9), (
        f"Expected positive clip to {a_max}, got {cmd_pos}"
    )

    # Negative saturation
    cmd_neg = guidance.compute_command(los_rate_hat=-10.0, Vc=500.0)
    assert cmd_neg == pytest.approx(-a_max, abs=1e-9), (
        f"Expected negative clip to -{a_max}, got {cmd_neg}"
    )


def test_linearity_pre_saturation(config, guidance):
    """
    Pre-saturation: doubling LOS rate must double the command.
    Confirms linear PN law implementation.
    """
    guidance.reset()
    N = config["guidance"]["N"]
    Vc = 300.0

    los_rate_1 = 0.001   # Small value, well within saturation
    los_rate_2 = 0.002   # Double

    cmd1 = guidance.compute_command(los_rate_hat=los_rate_1, Vc=Vc)
    cmd2 = guidance.compute_command(los_rate_hat=los_rate_2, Vc=Vc)

    assert cmd2 == pytest.approx(2.0 * cmd1, rel=1e-6), (
        f"Linearity failed: cmd1={cmd1:.6f}, cmd2={cmd2:.6f}, "
        f"ratio={cmd2/cmd1:.6f} (expected 2.0)"
    )


def test_sign_convention(config, guidance):
    """
    Positive Vc (closing) + positive LOS rate = positive command.
    Confirms sign convention matches PN law definition.
    """
    guidance.reset()
    cmd = guidance.compute_command(los_rate_hat=0.01, Vc=300.0)
    assert cmd > 0.0, (
        f"Expected positive command for positive Vc and positive "
        f"LOS rate, got {cmd}"
    )


def test_export_history_structure(config, guidance):
    """
    export_history() must return DataFrame with exactly 6 columns
    and number of rows equal to number of compute_command calls.
    """
    guidance.reset()
    n_calls = 15
    for i in range(n_calls):
        guidance.compute_command(
            los_rate_hat=0.001 * i,
            Vc=300.0
        )

    df = guidance.export_history()
    expected_cols = [
        "t", "los_rate_hat", "Vc",
        "a_cmd_raw", "a_cmd", "clipped"
    ]

    assert list(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {list(df.columns)}"
    )
    assert len(df) == n_calls, (
        f"Expected {n_calls} rows, got {len(df)}"
    )


def test_clip_count_tracking(config, guidance):
    """
    Clip count must increment exactly once per saturated command.
    """
    guidance.reset()
    # 3 saturating commands, 2 non-saturating
    guidance.compute_command(10.0, 500.0)   # saturates
    guidance.compute_command(-10.0, 500.0)  # saturates
    guidance.compute_command(0.001, 300.0)  # does NOT saturate
    guidance.compute_command(10.0, 500.0)   # saturates
    guidance.compute_command(0.0, 300.0)    # does NOT saturate

    assert guidance.get_clip_count() == 3, (
        f"Expected clip count 3, got {guidance.get_clip_count()}"
    )

