import pytest
import yaml
import json
import os
import sys
import copy
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.simulation_runner import SPECTRESimulation

CONFIG_PATH = "config/sim_config.yaml"
OUTPUT_DIR  = "results/data/test_runner"


@pytest.fixture
def clean_sim():
    """Simulation with no attack (injection_rate=0, active=False)."""
    sim = SPECTRESimulation.from_config_override(
        CONFIG_PATH,
        {
            "attacker.injection_rate": 0.0,
            "attacker.active": False
        }
    )
    return sim


@pytest.fixture
def attack_sim():
    """Simulation with active attack (injection_rate=0.05)."""
    sim = SPECTRESimulation.from_config_override(
        CONFIG_PATH,
        {
            "attacker.injection_rate": 0.05,
            "attacker.injection_angle_deg": 0.0,
            "attacker.active": True
        }
    )
    return sim


def test_clean_run_miss_distance(clean_sim):
    """
    Clean run (no attack) must achieve miss distance < 5m.
    Validates that PN + EKF guidance works correctly.
    """
    results = clean_sim.run(seed=42)
    miss = results["miss_distance"]
    assert miss < 5.0, (
        f"Clean run miss distance {miss:.2f}m exceeds 5m. "
        f"EKF or PN guidance is not working correctly."
    )


def test_clean_run_detection_rate(clean_sim):
    """
    Clean run must have detection rate < 0.07.
    Validates monitor is not falsely alarming on clean measurements.
    """
    results = clean_sim.run(seed=42)
    rate = results["detection_rate"]
    assert rate < 0.07, (
        f"Clean run detection rate {rate:.4f} exceeds 0.07. "
        f"Monitor is falsely alarming on clean measurements."
    )


def test_attack_run_miss_distance(attack_sim):
    """
    Attack run (I_dot=0.05) must achieve miss distance > 50m.
    Validates that adversarial injection degrades guidance.
    """
    results = attack_sim.run(seed=42)
    miss = results["miss_distance"]
    assert miss > 50.0, (
        f"Attack run miss distance {miss:.2f}m is under 50m. "
        f"Attack is not effective at I_dot=0.05."
    )


def test_results_dict_keys(clean_sim):
    """
    Results dict must contain all 10 required scalar keys.
    """
    results = clean_sim.run(seed=0)
    required_keys = [
        "miss_distance", "t_final", "detection_rate",
        "max_chi2", "Ca_estimate", "lock_time",
        "clipping_events", "total_alarms",
        "injection_rate", "seed"
    ]
    for key in required_keys:
        assert key in results, (
            f"Missing required key '{key}' in results dict."
        )


def test_csv_files_exist_and_under_99mb(clean_sim):
    """
    save_results() must create all 5 CSV files,
    each under 99MB.
    """
    results = clean_sim.run(seed=0)
    clean_sim.save_results(OUTPUT_DIR, run_id="test_clean")

    size_limit = 99 * 1024 * 1024
    expected_files = [
        "test_clean_geometry_df.csv",
        "test_clean_ekf_df.csv",
        "test_clean_guidance_df.csv",
        "test_clean_attacker_df.csv",
        "test_clean_monitor_df.csv",
    ]
    for fname in expected_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        assert os.path.exists(fpath), (
            f"Expected file not found: {fpath}"
        )
        fsize = os.path.getsize(fpath)
        assert fsize < size_limit, (
            f"File {fname} exceeds 99MB: {fsize/1024**2:.2f}MB"
        )


def test_summary_json_valid(clean_sim):
    """
    summary JSON must be valid and contain all scalar metrics.
    """
    clean_sim.run(seed=0)
    clean_sim.save_results(OUTPUT_DIR, run_id="test_clean")

    json_path = os.path.join(
        OUTPUT_DIR, "test_clean_summary.json"
    )
    assert os.path.exists(json_path), (
        f"Summary JSON not found at {json_path}"
    )
    with open(json_path, "r") as f:
        summary = json.load(f)

    assert "miss_distance" in summary, (
        "miss_distance missing from summary JSON"
    )
    assert isinstance(summary["miss_distance"], float), (
        "miss_distance must be float in JSON"
    )


def test_config_override_works():
    """
    from_config_override() must correctly apply overrides
    without modifying base config file.
    """
    sim1 = SPECTRESimulation.from_config_override(
        CONFIG_PATH,
        {"attacker.injection_rate": 0.03, "attacker.active": True}
    )
    sim2 = SPECTRESimulation.from_config_override(
        CONFIG_PATH,
        {"attacker.injection_rate": 0.07, "attacker.active": True}
    )
    assert sim1.config["attacker"]["injection_rate"] == 0.03
    assert sim2.config["attacker"]["injection_rate"] == 0.07

    # Base config must be unchanged
    with open(CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    assert base["attacker"]["injection_rate"] == 0.0, (
        "Base config was modified by from_config_override()!"
    )
