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
    The convergence ratio (mean CV of gain across MC seeds in the
    mid-game window) must be below 0.10, demonstrating that the
    Kalman gain trajectory is essentially deterministic.
    """
    ratio = summary["gain_convergence"]["convergence_ratio"]
    assert ratio < 0.10, (
        f"Gain convergence ratio {ratio:.6f} >= 0.10."
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


def test_directional_r_above_threshold(summary):
    """
    Directional correlation between cos(theta_inj) and cross-track
    miss must exceed 0.95, demonstrating strong directional control
    of the miss vector through bearing injection.
    """
    r_cl = summary["directional_correlation"]["circular_linear_r"]
    p_cl = summary["directional_correlation"]["circular_linear_p"]
    assert r_cl > 0.95, (
        f"Directional r = {r_cl:.4f} <= 0.95."
    )
    assert p_cl < 0.05, (
        f"Directional p = {p_cl:.4f} is not significant."
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


def test_miss_angle_is_derived_from_geometry():
    """
    miss_angle in directional_control.csv must be derived from
    np.arctan2(missile_y - target_y, missile_x - target_x) at
    closest approach, producing non-zero std_miss_angle for all
    8 injection angles.
    """
    assert os.path.exists(DIR_CSV), (
        f"Directional CSV not found: {DIR_CSV}"
    )
    df = pd.read_csv(DIR_CSV)
    assert "std_miss_angle" in df.columns, (
        "Column 'std_miss_angle' missing from directional CSV."
    )
    for _, row in df.iterrows():
        assert row["std_miss_angle"] > 0.0, (
            f"std_miss_angle is zero for angle "
            f"{row['injection_angle_deg']} degrees. "
            f"miss_angle may still be copied from theta_inj."
        )


def test_circular_linear_r_used_not_pearson():
    """
    Summary must contain circular_linear_r and must NOT contain
    pearson_r — Pearson is invalid for circular angular data.
    """
    assert os.path.exists(SUMMARY_PATH), (
        f"Summary not found: {SUMMARY_PATH}"
    )
    with open(SUMMARY_PATH) as f:
        s = json.load(f)
    corr = s["directional_correlation"]
    assert "circular_linear_r" in corr, (
        "circular_linear_r not found in directional_correlation."
    )
    assert "pearson_r" not in corr, (
        "pearson_r should have been replaced by "
        "circular_linear_r."
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
