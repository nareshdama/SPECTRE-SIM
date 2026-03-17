import pytest
import json
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

SUMMARY_PATH = "results/data/miss_proportionality_summary.json"
SWEEP_CSV    = "results/data/miss_proportionality_sweep.csv"
FIG1_PATH    = "results/figures/fig1_miss_distance_vs_injection_rate.png"
FIG2_PATH    = "results/figures/fig2_detection_rate_vs_injection_rate.png"


@pytest.fixture
def summary():
    """Load experiment summary JSON. Fails if experiment not run."""
    assert os.path.exists(SUMMARY_PATH), (
        f"Summary file not found: {SUMMARY_PATH}\n"
        f"Run: python experiments/run_miss_distance_proportionality.py"
    )
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def test_r_squared_above_threshold(summary):
    """
    R-squared of linear fit D_m = Ca * I_dot must exceed 0.95.
    Confirms miss distance scales proportionally with injection rate.
    """
    R2 = summary["regression"]["R_squared"]
    assert R2 > 0.95, (
        f"R-squared {R2:.4f} is below 0.95. "
        f"Miss distance is not proportional to injection rate. "
        f"Increase N_MONTE_CARLO to 50 or check EKF lock timing."
    )


def test_ca_is_positive_and_finite(summary):
    """
    Proportionality coefficient Ca must be positive and finite.
    Negative Ca means miss distance DECREASES with injection — wrong.
    """
    Ca = summary["regression"]["Ca"]
    assert Ca > 0, (
        f"Ca = {Ca:.4f} is not positive. "
        f"Check injection angle is 0 degrees (cos=1, full projection)."
    )
    assert not (Ca != Ca), f"Ca is NaN — regression failed."
    assert Ca < 1e6,       f"Ca = {Ca:.2e} is unrealistically large."


def test_baseline_miss_under_5m():
    """
    Zero injection rate run must achieve miss distance < 5m.
    Validates PN + EKF guidance baseline before attack.
    """
    import pandas as pd
    assert os.path.exists(SWEEP_CSV), (
        f"Sweep CSV not found: {SWEEP_CSV}"
    )
    df = pd.read_csv(SWEEP_CSV)
    baseline = df[df["injection_rate"] == 0.0]["mean_miss"].values
    assert len(baseline) > 0, "No zero injection rate row in sweep CSV."
    assert baseline[0] < 5.0, (
        f"Baseline miss distance {baseline[0]:.2f}m exceeds 5m. "
        f"PN+EKF guidance is not working correctly."
    )


def test_high_injection_miss_over_80m():
    """
    Injection rate = 0.10 rad/s^2 must produce mean miss > 80m.
    Validates attack is effective at moderate injection rates.
    """
    import pandas as pd
    df = pd.read_csv(SWEEP_CSV)
    high_inj = df[df["injection_rate"] == 0.10]["mean_miss"].values
    assert len(high_inj) > 0, "No injection_rate=0.10 row in sweep CSV."
    assert high_inj[0] > 80.0, (
        f"High injection miss {high_inj[0]:.2f}m is below 80m. "
        f"Attack at I_dot=0.10 is not sufficiently effective."
    )


def test_anova_p_value_significant(summary):
    """
    ANOVA p-value across injection rate groups must be < 0.05.
    Confirms injection rate is a statistically significant factor.
    """
    p = summary["anova"]["p_value"]
    assert p < 0.05, (
        f"ANOVA p-value {p:.4f} >= 0.05. "
        f"Injection rate does not significantly affect miss distance. "
        f"Increase N_MONTE_CARLO or check engagement geometry."
    )


def test_figures_exist_and_nonempty():
    """
    Both figure PNG files must exist and be larger than 50KB.
    Validates matplotlib successfully rendered the plots.
    """
    for path in [FIG1_PATH, FIG2_PATH]:
        assert os.path.exists(path), (
            f"Figure not found: {path}"
        )
        size = os.path.getsize(path)
        assert size > 50_000, (
            f"Figure {path} is suspiciously small: {size} bytes. "
            f"Matplotlib may have rendered an empty plot."
        )
