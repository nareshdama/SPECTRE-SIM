import pytest
import json
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

SUMMARY_PATH = "results/data/covert_threshold_summary.json"
SWEEP_CSV    = "results/data/covert_threshold_sweep.csv"
FIG3_PATH    = "results/figures/fig3_covert_injection_threshold.png"


@pytest.fixture
def summary():
    """Load experiment summary JSON. Fails if experiment not run."""
    assert os.path.exists(SUMMARY_PATH), (
        f"Summary not found: {SUMMARY_PATH}\n"
        f"Run: python experiments/run_covert_threshold.py"
    )
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def test_analytical_threshold_is_positive_finite(summary):
    """
    Analytically derived critical injection rate I_dot* must be
    a positive, finite, physically meaningful value.
    """
    I_dot_star = summary["I_dot_star_analytical"]
    assert I_dot_star > 0, (
        f"I_dot* analytical = {I_dot_star} is not positive. "
        f"Check EKF steady-state derivation in the experiment."
    )
    assert I_dot_star < 10.0, (
        f"I_dot* analytical = {I_dot_star:.4f} is unrealistically "
        f"large (> 10 rad/s^2). Check bandwidth estimation."
    )
    assert I_dot_star == I_dot_star, "I_dot* is NaN."


def test_covert_zone_below_detection_trigger(summary):
    """
    At injection rate = 0.5 * I_dot*, the mean detection rate
    must be below 0.10 (inside the covert zone).
    """
    rate = summary.get("covert_zone_detection_at_half_star")
    assert rate is not None, (
        "covert_zone_detection_at_half_star is None. "
        "Check that sweep covers 0.5 * I_dot_star range."
    )
    assert rate < 0.10, (
        f"Detection rate at 0.5 * I_dot* = {rate:.4f} >= 0.10. "
        f"Covert zone is not confirmed. "
        f"Reduce R_scalar or increase t_max in config."
    )


def test_detectable_zone_above_detection_trigger(summary):
    """
    At high injection rates (~4x I_dot*), the mean detection rate
    must be above 0.30 (clearly in the detectable zone).
    """
    rate = summary.get("detectable_zone_detection_at_high")
    assert rate is not None, (
        "detectable_zone_detection_at_high is None. "
        "Check that sweep covers 4 * I_dot_star range."
    )
    assert rate > 0.30, (
        f"Detection rate at high rates = {rate:.4f} <= 0.30. "
        f"Attack is not detectable above threshold. "
        f"Check injection angle = 0 degrees."
    )


def test_analytical_empirical_agreement_within_15pct(summary):
    """
    Analytical and empirical I_dot* must agree within 15%.
    Validates the theoretical derivation against simulation.
    """
    agreement = summary["agreement_pct"]
    assert agreement < 15.0, (
        f"Analytical vs empirical agreement = {agreement:.2f}% "
        f"exceeds 15% tolerance. "
        f"Increase N_MONTE_CARLO to 100 or refine S_inf estimate."
    )


def test_sweep_csv_has_correct_structure():
    """
    Sweep CSV must exist, have 100 rows and 4 required columns.
    """
    assert os.path.exists(SWEEP_CSV), (
        f"Sweep CSV not found: {SWEEP_CSV}"
    )
    df = pd.read_csv(SWEEP_CSV)
    assert len(df) == 100, (
        f"Expected 100 rows in sweep CSV, got {len(df)}."
    )
    required_cols = [
        "injection_rate", "mean_detection_rate",
        "std_detection_rate", "n_trials"
    ]
    for col in required_cols:
        assert col in df.columns, (
            f"Missing column '{col}' in sweep CSV."
        )


def test_figure_exists_and_nonempty():
    """
    Figure 3 PNG must exist and be larger than 50KB.
    """
    assert os.path.exists(FIG3_PATH), (
        f"Figure not found: {FIG3_PATH}"
    )
    size = os.path.getsize(FIG3_PATH)
    assert size > 50_000, (
        f"Figure {FIG3_PATH} is too small: {size} bytes. "
        f"Matplotlib may have rendered an empty plot."
    )
