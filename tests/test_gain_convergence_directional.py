import pytest
import json
import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

SUMMARY_PATH  = "results/data/gain_convergence_directional_summary.json"
GAIN_CSV      = "results/data/gain_convergence_grouped.csv"
DIR_CSV       = "results/data/directional_control.csv"
FIG4_PATH     = "results/figures/fig4_kalman_gain_convergence.png"
FIG5_PATH     = "results/figures/fig5_directional_miss_vector_control.png"


@pytest.fixture
def summary():
    """Load experiment summary JSON. Fails if experiment not run."""
    assert os.path.exists(SUMMARY_PATH), (
        f"Summary not found: {SUMMARY_PATH}\n"
        f"Run: python experiments/"
        f"run_gain_convergence_directional.py"
    )
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def test_gain_convergence_ratio_below_threshold(summary):
    """
    Ratio of late-period standard deviation to early-period
    standard deviation of Kalman gain norm must be below 0.10.
    Confirms gain has stabilized 10x after acquisition lock.
    """
    ratio = summary["gain_convergence"]["convergence_ratio"]
    assert ratio < 0.10, (
        f"Gain convergence ratio {ratio:.4f} >= 0.10. "
        f"Kalman gain has not stabilized after lock. "
        f"Reduce Q process noise or extend t_max in config."
    )


def test_ca_coefficient_of_variation_below_threshold(summary):
    """
    Coefficient of variation of Ca across all injection angles
    must be below 0.05 (5%).
    Confirms Ca is a stable, predictable constant post-lock.
    """
    CoV = summary["ca_stability"]["Ca_CoV"]
    assert CoV is not None, (
        "Ca_CoV is None. Check directional control study ran "
        "with injection_rate > 0."
    )
    assert CoV < 0.05, (
        f"Ca coefficient of variation {CoV:.4f} >= 0.05. "
        f"Ca is not stable across injection angles. "
        f"Increase N_MONTE_CARLO or reduce injection rate."
    )


def test_pearson_r_above_threshold(summary):
    """
    Pearson correlation between injection angle and miss vector
    angle must exceed 0.95.
    Confirms directional control: attacker steers miss vector.
    """
    best_r = summary["directional_correlation"]["best_r"]
    assert best_r > 0.95, (
        f"Pearson r = {best_r:.4f} is below 0.95. "
        f"Directional control not confirmed. "
        f"Verify injection angle cos() projection in attacker. "
        f"Try increasing injection rate to 0.8 * I_dot_star."
    )


def test_gain_csv_structure():
    """
    Gain convergence grouped CSV must have columns
    t, mean_gain, std_gain and at least 100 timestep rows.
    """
    assert os.path.exists(GAIN_CSV), (
        f"Gain CSV not found: {GAIN_CSV}"
    )
    df = pd.read_csv(GAIN_CSV)
    required_cols = ["t", "mean_gain", "std_gain"]
    for col in required_cols:
        assert col in df.columns, (
            f"Missing column '{col}' in gain convergence CSV."
        )
    assert len(df) >= 100, (
        f"Expected at least 100 timestep rows, got {len(df)}. "
        f"Check simulation t_max and dt."
    )


def test_directional_csv_has_8_angles():
    """
    Directional control CSV must have exactly 8 rows
    (one per injection angle) with correct angle values.
    """
    assert os.path.exists(DIR_CSV), (
        f"Directional CSV not found: {DIR_CSV}"
    )
    df = pd.read_csv(DIR_CSV)
    assert len(df) == 8, (
        f"Expected 8 rows (one per angle), got {len(df)}."
    )
    expected_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    actual_angles   = sorted(df["injection_angle_deg"].tolist())
    for exp, act in zip(expected_angles, actual_angles):
        assert abs(exp - act) < 0.1, (
            f"Expected angle {exp}, got {act}."
        )


def test_ca_is_positive_across_all_angles():
    """
    Mean Ca must be positive for all injection angles.
    Confirms attack increases miss distance (not decreases it).
    """
    df = pd.read_csv(DIR_CSV)
    for _, row in df.iterrows():
        Ca = row["mean_Ca"]
        assert Ca > 0, (
            f"Ca = {Ca:.4f} is not positive for angle "
            f"{row['injection_angle_deg']} degrees."
        )


def test_figures_exist_and_nonempty():
    """
    Figure 4 and Figure 5 PNG files must exist and
    each be larger than 50KB.
    """
    for path in [FIG4_PATH, FIG5_PATH]:
        assert os.path.exists(path), (
            f"Figure not found: {path}"
        )
        size = os.path.getsize(path)
        assert size > 50_000, (
            f"Figure {path} is too small: {size} bytes. "
            f"Matplotlib may have rendered an empty plot."
        )
