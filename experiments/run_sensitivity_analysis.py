"""
SPECTRE-SIM Experiment 4: Sensitivity Analysis

Sweeps key engagement and filter parameters to quantify robustness
of the three hypotheses (H1–H3) under parametric variation:

    - Navigation constant N: {3, 4, 5}
    - Closing velocity scale: target vx0 in {-150, -200, -250} m/s
    - Process noise Q scale factor: {0.5, 1.0, 2.0}
    - Measurement noise R scale factor: {0.5, 1.0, 2.0}

For each parameter set, a focused Monte Carlo sweep is run to
measure Ca, I_dot_star sensitivity, and directional r_cl.
"""

import sys
import os
import json
import copy
import itertools
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.simulation_runner import SPECTRESimulation

CONFIG_PATH = "config/sim_config.yaml"
DATA_DIR    = "results/data"
FIGURES_DIR = "results/figures"
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_MC = 30
SUPER_THRESHOLD_RATE = 0.06
INJECTION_ANGLE = 0.0

PARAM_GRID = {
    "N":       [3, 4, 5],
    "V_scale": [-150.0, -200.0, -250.0],
    "Q_scale": [0.5, 1.0, 2.0],
    "R_scale": [0.5, 1.0, 2.0],
}

BASELINE = {"N": 4, "V_scale": -200.0, "Q_scale": 1.0, "R_scale": 1.0}


def build_overrides(params: dict, base_config: dict) -> dict:
    """Build config overrides dict from parameter values."""
    Q_base = base_config["ekf"]["Q_diag"]
    R_base = base_config["ekf"]["R_diag"]

    overrides = {
        "guidance.N":       params["N"],
        "simulation.N":     params["N"],
        "target.vx0":       params["V_scale"],
    }

    q_scaled = [q * params["Q_scale"] for q in Q_base]
    r_scaled = [r * params["R_scale"] for r in R_base]
    overrides["ekf.Q_diag"] = q_scaled
    overrides["ekf.R_diag"] = r_scaled

    return overrides


def run_single_config(overrides: dict, rate: float,
                      angle: float, n_mc: int) -> dict:
    """Run MC trials for a single parameter config and rate."""
    miss_dists = []
    det_rates = []

    for seed in range(n_mc):
        all_ov = copy.deepcopy(overrides)
        all_ov["attacker.injection_rate"] = rate
        all_ov["attacker.injection_angle_deg"] = angle
        all_ov["attacker.active"] = (rate > 1e-9)

        sim = SPECTRESimulation.from_config_override(
            CONFIG_PATH, all_ov
        )
        result = sim.run(seed=seed)
        miss_dists.append(result["miss_distance"])
        det_rates.append(result["detection_rate"])

    return {
        "mean_miss": float(np.mean(miss_dists)),
        "std_miss":  float(np.std(miss_dists)),
        "mean_det":  float(np.mean(det_rates)),
        "Ca_estimate": (float(np.mean(miss_dists)) / rate
                        if rate > 1e-9 else None),
    }


def one_at_a_time_sweep(base_config: dict) -> pd.DataFrame:
    """
    One-at-a-time sensitivity: vary each parameter while holding
    others at baseline.
    """
    records = []

    for param_name, values in PARAM_GRID.items():
        for val in tqdm(values,
                        desc=f"Sensitivity sweep: {param_name}"):
            params = copy.deepcopy(BASELINE)
            params[param_name] = val
            overrides = build_overrides(params, base_config)

            res = run_single_config(
                overrides, SUPER_THRESHOLD_RATE,
                INJECTION_ANGLE, N_MC
            )

            records.append({
                "param": param_name,
                "value": val,
                "mean_miss": res["mean_miss"],
                "std_miss":  res["std_miss"],
                "mean_det":  res["mean_det"],
                "Ca":        res["Ca_estimate"],
            })

    return pd.DataFrame(records)


def generate_sensitivity_figure(df: pd.DataFrame) -> None:
    """Tornado chart showing Ca sensitivity to each parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    params = list(PARAM_GRID.keys())
    titles = {
        "N": "Navigation Constant N",
        "V_scale": "Target Velocity (m/s)",
        "Q_scale": "Process Noise Scale",
        "R_scale": "Measurement Noise Scale",
    }

    for ax, pname in zip(axes.flat, params):
        sub = df[df["param"] == pname].sort_values("value")
        x = sub["value"].values
        ca = sub["Ca"].values
        err = sub["std_miss"].values / SUPER_THRESHOLD_RATE

        ax.bar(range(len(x)), ca, yerr=err,
               color="#1f77b4", alpha=0.8, capsize=5)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f"{v}" for v in x])
        ax.set_title(titles.get(pname, pname), fontsize=11)
        ax.set_ylabel(r"$C_a$ [m$\cdot$s/rad]", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Sensitivity Analysis: Proportionality Coefficient $C_a$\n"
        f"(rate = {SUPER_THRESHOLD_RATE} rad/s², "
        f"N_MC = {N_MC})",
        fontsize=13
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(
        FIGURES_DIR, "fig6_sensitivity_analysis.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("=" * 62)
    print("SPECTRE-SIM: Sensitivity Analysis (Experiment 4)")
    print("=" * 62)

    print(f"\nBaseline: N={BASELINE['N']}, "
          f"V_scale={BASELINE['V_scale']}, "
          f"Q_scale={BASELINE['Q_scale']}, "
          f"R_scale={BASELINE['R_scale']}")
    print(f"Injection rate: {SUPER_THRESHOLD_RATE} rad/s², "
          f"N_MC = {N_MC} per config\n")

    df = one_at_a_time_sweep(config)

    csv_path = os.path.join(DATA_DIR, "sensitivity_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    generate_sensitivity_figure(df)

    baseline_row = df[
        (df["param"] == "N") & (df["value"] == BASELINE["N"])
    ]
    Ca_baseline = float(baseline_row["Ca"].iloc[0])

    summary = {
        "experiment": "sensitivity_analysis",
        "baseline": BASELINE,
        "super_threshold_rate": SUPER_THRESHOLD_RATE,
        "n_mc": N_MC,
        "Ca_baseline": Ca_baseline,
        "results": df.to_dict(orient="records"),
    }

    summary_path = os.path.join(
        DATA_DIR, "sensitivity_analysis_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Report range of Ca
    ca_vals = df["Ca"].dropna().values
    print(f"\nCa range: [{ca_vals.min():.1f}, {ca_vals.max():.1f}] "
          f"m·s/rad")
    print(f"Ca CoV:   {np.std(ca_vals)/np.mean(ca_vals)*100:.1f}%")

    return summary


if __name__ == "__main__":
    main()
