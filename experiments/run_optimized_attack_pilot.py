"""
Pilot Monte Carlo: ramp vs optimization-constrained stealth injection.

Compares mean miss distance and chi-squared / CUSUM tracking-phase
detection rates at matched aggressiveness (injection_rate used only
for ramp; optimized uses du caps and chi2_margin from config).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.simulation_runner import SPECTRESimulation

CONFIG_PATH = "config/sim_config.yaml"
DATA_DIR = "results/data"
os.makedirs(DATA_DIR, exist_ok=True)

N_MC = 20
INJECTION_RATE = 0.05
INJECTION_ANGLE_DEG = 0.0


def _overrides_for_mode(mode: str) -> dict:
    base = {
        "attacker.mode": mode,
        "attacker.injection_rate": INJECTION_RATE,
        "attacker.injection_angle_deg": INJECTION_ANGLE_DEG,
        "attacker.active": True,
    }
    if mode == "optimized":
        # Slightly larger per-step caps so the pilot shows a stealth–impact trade.
        base["attacker.optimized.du_max_bearing"] = 0.00025
        base["attacker.optimized.du_max_range"] = 50.0
        base["attacker.optimized.chi2_margin"] = 0.94
    return base


def _run_batch(mode: str, n_mc: int, seed_offset: int = 0) -> dict:
    misses = []
    det_chi2 = []
    det_cusum = []
    for i in range(n_mc):
        sim = SPECTRESimulation.from_config_override(
            CONFIG_PATH,
            _overrides_for_mode(mode),
        )
        r = sim.run(seed=seed_offset + i)
        misses.append(r["miss_distance"])
        det_chi2.append(r["detection_rate"])
        det_cusum.append(r["detection_rate_cusum_tracking"])
    return {
        "mode": mode,
        "mean_miss_m": float(np.mean(misses)),
        "std_miss_m": float(np.std(misses)),
        "mean_det_chi2_tracking": float(np.mean(det_chi2)),
        "std_det_chi2_tracking": float(np.std(det_chi2)),
        "mean_det_cusum_tracking": float(np.mean(det_cusum)),
        "std_det_cusum_tracking": float(np.std(det_cusum)),
        "n_mc": n_mc,
    }


def main():
    print("SPECTRE-SIM: ramp vs optimized stealth attack pilot")
    rows = []
    for mode in ("ramp", "optimized"):
        print(f"  Running mode={mode} ...")
        stat = _run_batch(mode, N_MC)
        rows.append(stat)
        print(f"    mean miss = {stat['mean_miss_m']:.1f} m, "
              f"chi2 det = {stat['mean_det_chi2_tracking']:.4f}, "
              f"cusum det = {stat['mean_det_cusum_tracking']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATA_DIR, "optimized_attack_pilot.csv")
    df.to_csv(csv_path, index=False)
    summary = {
        "experiment": "optimized_attack_pilot",
        "injection_rate_ramp": INJECTION_RATE,
        "injection_angle_deg": INJECTION_ANGLE_DEG,
        "n_mc": N_MC,
        "results": rows,
    }
    json_path = os.path.join(DATA_DIR, "optimized_attack_pilot_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
