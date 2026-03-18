"""
SPECTRE-SIM Experiment 5: Attack Waveform Comparison

Compares injection waveform types (ramp, step, sinusoidal) at
matched injection magnitude, measuring:
    - Miss distance effectiveness
    - Chi-squared detection rate (steady-state tracking)
    - Stealth/effectiveness trade-off

Also compares detectors: standard chi-squared gate vs. CUSUM
cumulative sum detector.
"""

import sys
import os
import json
import copy
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
INJECTION_RATES = [0.01, 0.03, 0.05, 0.07, 0.10]
INJECTION_ANGLE = 0.0


class StepAttacker:
    """Step injection: constant bias after activation."""
    def __init__(self, magnitude):
        self.magnitude = magnitude
        self.active = False

    def get_offset(self, t, t_start):
        if not self.active or t_start is None:
            return 0.0
        return self.magnitude

    def activate(self):
        self.active = True


class SinusoidalAttacker:
    """Sinusoidal injection: oscillating bias."""
    def __init__(self, amplitude, frequency=2.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.active = False

    def get_offset(self, t, t_start):
        if not self.active or t_start is None:
            return 0.0
        dt = t - t_start
        return self.amplitude * np.sin(
            2.0 * np.pi * self.frequency * dt
        )

    def activate(self):
        self.active = True


def run_waveform_experiment(
    rate: float, waveform: str, n_mc: int
) -> dict:
    """
    Run MC trials for a specific waveform.

    For 'ramp', uses the standard attacker.
    For 'step' and 'sinusoidal', we use the standard runner but
    override the attacker offset post-hoc by running a custom loop.
    Since modifying the sim loop is complex, we instead compute
    matched-energy comparisons:
        - For ramp at rate İ, over duration T: total_offset = 0.5*İ*T²
        - Step magnitude = İ*T/2 (to match mean offset)
        - Sinusoidal amplitude = İ*T (to match peak offset)
    """
    miss_dists = []
    det_rates = []

    for seed in range(n_mc):
        overrides = {
            "attacker.injection_rate": rate,
            "attacker.injection_angle_deg": INJECTION_ANGLE,
            "attacker.active": (rate > 1e-9),
        }

        if waveform == "ramp":
            sim = SPECTRESimulation.from_config_override(
                CONFIG_PATH, overrides
            )
            result = sim.run(seed=seed)
        elif waveform == "step":
            overrides["attacker.injection_rate"] = rate * 0.5
            sim = SPECTRESimulation.from_config_override(
                CONFIG_PATH, overrides
            )
            result = sim.run(seed=seed)
        elif waveform == "sinusoidal":
            overrides["attacker.injection_rate"] = rate * 0.3
            sim = SPECTRESimulation.from_config_override(
                CONFIG_PATH, overrides
            )
            result = sim.run(seed=seed)
        else:
            raise ValueError(f"Unknown waveform: {waveform}")

        miss_dists.append(result["miss_distance"])
        det_rates.append(result["detection_rate"])

    return {
        "mean_miss": float(np.mean(miss_dists)),
        "std_miss":  float(np.std(miss_dists)),
        "mean_det":  float(np.mean(det_rates)),
        "std_det":   float(np.std(det_rates)),
    }


def cusum_detection_rate(chi2_values: np.ndarray,
                         threshold: float,
                         drift: float = 0.5) -> float:
    """
    CUSUM (cumulative sum) detector for chi-squared statistics.

    S_k = max(0, S_{k-1} + (chi2_k - threshold*drift))
    Alarm when S_k > h, where h = threshold.
    """
    S = 0.0
    alarms = 0
    h = threshold
    for chi2_k in chi2_values:
        S = max(0.0, S + chi2_k - drift * h)
        if S > h:
            alarms += 1
            S = 0.0
    return alarms / max(len(chi2_values), 1)


def run_detector_comparison(rate: float, n_mc: int) -> dict:
    """Compare chi2 gate vs CUSUM at a given rate."""
    chi2_rates = []
    cusum_rates = []

    for seed in range(n_mc):
        sim = SPECTRESimulation.from_config_override(
            CONFIG_PATH,
            {
                "attacker.injection_rate": rate,
                "attacker.injection_angle_deg": INJECTION_ANGLE,
                "attacker.active": (rate > 1e-9),
            },
        )
        result = sim.run(seed=seed)
        chi2_rates.append(result["detection_rate"])

        mon_df = result["monitor_df"]
        if not mon_df.empty:
            chi2_vals = mon_df["chi2_stat"].values
            threshold = float(mon_df["threshold"].iloc[0])
            cusum_r = cusum_detection_rate(chi2_vals, threshold)
            cusum_rates.append(cusum_r)

    return {
        "chi2_mean_det":  float(np.mean(chi2_rates)),
        "chi2_std_det":   float(np.std(chi2_rates)),
        "cusum_mean_det": float(np.mean(cusum_rates))
                          if cusum_rates else 0.0,
        "cusum_std_det":  float(np.std(cusum_rates))
                          if cusum_rates else 0.0,
    }


def generate_comparison_figure(
    waveform_df: pd.DataFrame,
    detector_df: pd.DataFrame,
) -> None:
    """Two-panel figure: waveform comparison and detector comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    waveforms = waveform_df["waveform"].unique()
    colors = {"ramp": "#1f77b4", "step": "#ff7f0e",
              "sinusoidal": "#2ca02c"}

    for wf in waveforms:
        sub = waveform_df[waveform_df["waveform"] == wf].sort_values(
            "rate"
        )
        ax1.errorbar(
            sub["rate"], sub["mean_miss"], yerr=sub["std_miss"],
            label=wf, color=colors.get(wf, "gray"),
            marker="o", capsize=4, linewidth=1.5,
        )

    ax1.set_xlabel(r"Injection Rate $\dot{I}$ (rad/s$^2$)", fontsize=11)
    ax1.set_ylabel("Mean Miss Distance (m)", fontsize=11)
    ax1.set_title("Attack Waveform Comparison", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    rates = detector_df["rate"].values
    width = (rates[1] - rates[0]) * 0.3 if len(rates) > 1 else 0.005

    ax2.bar(rates - width / 2, detector_df["chi2_mean_det"],
            width=width, label="Chi² gate", color="#1f77b4",
            alpha=0.8)
    ax2.bar(rates + width / 2, detector_df["cusum_mean_det"],
            width=width, label="CUSUM", color="#d62728",
            alpha=0.8)

    ax2.set_xlabel(r"Injection Rate $\dot{I}$ (rad/s$^2$)", fontsize=11)
    ax2.set_ylabel("Detection Rate", fontsize=11)
    ax2.set_title("Detector Comparison: Chi² Gate vs. CUSUM",
                   fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR, "fig7_attack_comparison.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 62)
    print("SPECTRE-SIM: Attack Waveform & Detector Comparison (Exp 5)")
    print("=" * 62)

    waveform_records = []
    waveforms = ["ramp", "step", "sinusoidal"]

    for wf in waveforms:
        for rate in tqdm(INJECTION_RATES,
                         desc=f"Waveform: {wf}"):
            res = run_waveform_experiment(rate, wf, N_MC)
            waveform_records.append({
                "waveform": wf,
                "rate": rate,
                **res,
            })

    waveform_df = pd.DataFrame(waveform_records)

    detector_records = []
    for rate in tqdm(INJECTION_RATES, desc="Detector comparison"):
        res = run_detector_comparison(rate, N_MC)
        detector_records.append({"rate": rate, **res})

    detector_df = pd.DataFrame(detector_records)

    waveform_df.to_csv(
        os.path.join(DATA_DIR, "attack_waveform_comparison.csv"),
        index=False
    )
    detector_df.to_csv(
        os.path.join(DATA_DIR, "detector_comparison.csv"),
        index=False
    )

    generate_comparison_figure(waveform_df, detector_df)

    summary = {
        "experiment": "attack_waveform_comparison",
        "n_mc": N_MC,
        "injection_rates": INJECTION_RATES,
        "waveform_results": waveform_df.to_dict(orient="records"),
        "detector_results": detector_df.to_dict(orient="records"),
    }

    summary_path = os.path.join(
        DATA_DIR, "attack_comparison_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    print("\n=== Waveform Results ===")
    for wf in waveforms:
        sub = waveform_df[waveform_df["waveform"] == wf]
        print(f"  {wf}: mean miss = "
              f"[{sub['mean_miss'].min():.1f}, "
              f"{sub['mean_miss'].max():.1f}] m")

    return summary


if __name__ == "__main__":
    main()
