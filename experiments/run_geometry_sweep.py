"""
Monte Carlo sweep over engagement geometries (initial conditions).

Scenarios (2D crossing geometry in config plane):
  - baseline: default sim_config missile/target
  - head_on:  target approaching along +x toward missile at origin
  - crossing: large initial cross-track offset
  - tail_chase: missile behind target (both +x velocity, missile faster)

Outputs per scenario: mean/std miss, chi-squared and CUSUM tracking
detection rates for active ramp attack at fixed injection_rate.
"""

import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulation_runner import SPECTRESimulation

CONFIG_PATH = "config/sim_config.yaml"
DATA_DIR = "results/data"
os.makedirs(DATA_DIR, exist_ok=True)

N_MC = 8
INJECTION_RATE = 0.05


def _scenario_overrides(name: str) -> dict:
    """Return flat dotted overrides for missile/target initial state."""
    if name == "baseline":
        return {}
    if name == "head_on":
        return {
            "missile.x0": 0.0,
            "missile.y0": 0.0,
            "missile.vx0": 400.0,
            "missile.vy0": 0.0,
            "target.x0": 8000.0,
            "target.y0": 0.0,
            "target.vx0": -250.0,
            "target.vy0": 0.0,
        }
    if name == "crossing":
        return {
            "missile.x0": 0.0,
            "missile.y0": 0.0,
            "missile.vx0": 300.0,
            "missile.vy0": 0.0,
            "target.x0": 12000.0,
            "target.y0": 2500.0,
            "target.vx0": -180.0,
            "target.vy0": -120.0,
        }
    if name == "tail_chase":
        return {
            "missile.x0": 0.0,
            "missile.y0": 0.0,
            "missile.vx0": 350.0,
            "missile.vy0": 0.0,
            "target.x0": 5000.0,
            "target.y0": 0.0,
            "target.vx0": 200.0,
            "target.vy0": 0.0,
        }
    raise ValueError(f"unknown scenario {name}")


def _merge_flat(base: dict, flat: dict) -> dict:
    out = copy.deepcopy(base)
    for dotted_key, value in flat.items():
        keys = dotted_key.split(".")
        d = out
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return out


def run_scenario(name: str, n_mc: int) -> dict:
    with open(CONFIG_PATH, "r") as f:
        base = yaml.safe_load(f)
    extra = _scenario_overrides(name)
    merged = _merge_flat(base, extra)
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(merged, tmp)
        tmp_path = tmp.name

    misses = []
    det_chi2 = []
    det_cusum = []
    try:
        for i in range(n_mc):
            sim = SPECTRESimulation.from_config_override(
                tmp_path,
                {
                    "attacker.mode": "ramp",
                    "attacker.injection_rate": INJECTION_RATE,
                    "attacker.injection_angle_deg": 0.0,
                    "attacker.active": True,
                },
            )
            r = sim.run(seed=1000 + i)
            misses.append(r["miss_distance"])
            det_chi2.append(r["detection_rate"])
            det_cusum.append(r["detection_rate_cusum_tracking"])
    finally:
        os.unlink(tmp_path)

    return {
        "scenario": name,
        "mean_miss_m": float(np.mean(misses)),
        "std_miss_m": float(np.std(misses)),
        "mean_det_chi2_tracking": float(np.mean(det_chi2)),
        "std_det_chi2_tracking": float(np.std(det_chi2)),
        "mean_det_cusum_tracking": float(np.mean(det_cusum)),
        "std_det_cusum_tracking": float(np.std(det_cusum)),
        "n_mc": n_mc,
    }


def main():
    print("SPECTRE-SIM: geometry sweep (ramp attack)")
    scenarios = ["baseline", "head_on", "crossing", "tail_chase"]
    rows = []
    for s in scenarios:
        print(f"  scenario={s} ...")
        stat = run_scenario(s, N_MC)
        rows.append(stat)
        print(
            f"    mean miss = {stat['mean_miss_m']:.1f} m, "
            f"chi2 det = {stat['mean_det_chi2_tracking']:.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATA_DIR, "geometry_sweep.csv")
    df.to_csv(csv_path, index=False)
    summary_path = os.path.join(DATA_DIR, "geometry_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"experiment": "geometry_sweep", "n_mc": N_MC, "results": rows},
            f,
            indent=2,
        )
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
