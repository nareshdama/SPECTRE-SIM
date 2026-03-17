import sys
import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.simulation_runner import SPECTRESimulation

# ── Paths ────────────────────────────────────────────────────
CONFIG_PATH  = "config/sim_config.yaml"
DATA_DIR     = "results/data"
FIGURES_DIR  = "results/figures"
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Experiment parameters ────────────────────────────────────
INJECTION_RATES = [0.0, 0.01, 0.02, 0.060, 0.065, 0.070, 0.080, 0.090, 0.10]
N_MONTE_CARLO   = 30
INJECTION_ANGLE = 0.0   # degrees — full projection (cos(0)=1)

# Post-transition rates used for linear regression
# (below ~0.05, the EKF compensates fully; above it, miss scales linearly)
REGRESSION_RATES = [0.060, 0.065, 0.070, 0.080, 0.090, 0.10]


# Store per-trial data for ANOVA (avoids expensive re-run)
_per_trial_data = {}


def run_sweep() -> pd.DataFrame:
    """
    Sweep injection rates. For each rate, run N_MONTE_CARLO trials
    varying only the random seed. Collect miss distance and
    detection rate per trial.
    """
    global _per_trial_data
    _per_trial_data = {}
    records = []

    for I_dot in tqdm(INJECTION_RATES,
                      desc="Injection rate sweep"):
        is_attack = I_dot > 1e-9
        miss_distances   = []
        detection_rates  = []

        for seed in range(N_MONTE_CARLO):
            sim = SPECTRESimulation.from_config_override(
                CONFIG_PATH,
                {
                    "attacker.injection_rate":     I_dot,
                    "attacker.injection_angle_deg": INJECTION_ANGLE,
                    "attacker.active":             is_attack
                }
            )
            results = sim.run(seed=seed)
            miss_distances.append(results["miss_distance"])
            detection_rates.append(results["detection_rate"])

        _per_trial_data[I_dot] = miss_distances

        records.append({
            "injection_rate":      I_dot,
            "mean_miss":           np.mean(miss_distances),
            "std_miss":            np.std(miss_distances),
            "median_miss":         np.median(miss_distances),
            "min_miss":            np.min(miss_distances),
            "max_miss":            np.max(miss_distances),
            "mean_detection_rate": np.mean(detection_rates),
            "std_detection_rate":  np.std(detection_rates),
            "n_trials":            N_MONTE_CARLO
        })

    return pd.DataFrame(records)


def fit_linear_regression(df: pd.DataFrame) -> dict:
    """
    Fit D_m = Ca * I_dot + b using scipy.stats.linregress.
    Uses mean miss distance per injection rate.
    Fits only on post-transition rates where miss scales linearly.
    Returns regression statistics dict.
    """
    df_reg = df[df["injection_rate"].isin(REGRESSION_RATES)]
    x = df_reg["injection_rate"].values
    y = df_reg["mean_miss"].values

    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(x, y)

    return {
        "Ca":          float(slope),
        "intercept":   float(intercept),
        "R_squared":   float(r_value ** 2),
        "p_value":     float(p_value),
        "std_err":     float(std_err)
    }


def run_anova(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA across injection rate groups.
    Tests whether injection rate significantly affects miss distance.
    Uses per-trial data collected during the sweep (no re-run needed).
    """
    groups = [_per_trial_data[I_dot] for I_dot in INJECTION_RATES]

    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "F_statistic": float(f_stat),
        "p_value":     float(p_value),
        "n_groups":    len(INJECTION_RATES),
        "n_per_group": N_MONTE_CARLO
    }


def generate_figure_miss_vs_injection(
        df: pd.DataFrame, reg: dict) -> None:
    """
    Figure 1: Miss distance vs injection rate.
    Scatter with error bars + linear regression line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = df["injection_rate"].values
    y = df["mean_miss"].values
    yerr = df["std_miss"].values

    # Error bars
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o", color="#1f77b4",
        ecolor="#aec7e8", elinewidth=1.5,
        capsize=4, markersize=6,
        label="Monte Carlo mean ± 1σ"
    )

    # Regression line
    x_line = np.linspace(0, max(x), 200)
    y_line = reg["Ca"] * x_line + reg["intercept"]
    ax.plot(
        x_line, y_line,
        "--", color="#d62728", linewidth=1.8,
        label=(
            f"Linear fit: $D_m = {reg['Ca']:.1f} \\cdot \\dot{{I}} "
            f"+ {reg['intercept']:.2f}$\n"
            f"$R^2 = {reg['R_squared']:.4f}$"
        )
    )

    ax.set_xlabel(
        "Injection Rate $\\dot{I}$ (rad/s per second)",
        fontsize=12
    )
    ax.set_ylabel("Miss Distance $D_m$ (m)", fontsize=12)
    ax.set_title(
        "Miss Distance vs. Injection Rate\n"
        "(Miss Distance Proportionality Hypothesis Validation)",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-5)

    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        "fig1_miss_distance_vs_injection_rate.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_detection_rate(df: pd.DataFrame) -> None:
    """
    Figure 2: Detection rate vs injection rate.
    Shows where the attack transitions from covert to detectable.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    x    = df["injection_rate"].values
    y    = df["mean_detection_rate"].values
    yerr = df["std_detection_rate"].values

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="s-", color="#2ca02c",
        ecolor="#98df8a", elinewidth=1.5,
        capsize=4, markersize=6,
        label="Mean detection rate ± 1σ"
    )

    # Alpha threshold line
    ax.axhline(
        y=0.05, color="#d62728",
        linestyle="--", linewidth=1.5,
        label="False alarm threshold α = 0.05"
    )

    ax.set_xlabel(
        "Injection Rate $\\dot{I}$ (rad/s per second)",
        fontsize=12
    )
    ax.set_ylabel("Chi-Squared Detection Rate", fontsize=12)
    ax.set_title(
        "Detection Rate vs. Injection Rate\n"
        "(Transition from Covert to Detectable Attack Zone)",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        "fig2_detection_rate_vs_injection_rate.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("SPECTRE-SIM: Miss Distance Proportionality Experiment")
    print("=" * 60)

    # Step 1: Run injection rate sweep
    print("\n[1/4] Running injection rate sweep...")
    df = run_sweep()

    csv_path = os.path.join(DATA_DIR, "miss_proportionality_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"      Saved sweep data: {csv_path}")

    # Step 2: Linear regression
    print("\n[2/4] Fitting linear regression (D_m = Ca * I_dot)...")
    reg = fit_linear_regression(df)
    print(f"      Ca        = {reg['Ca']:.4f}")
    print(f"      R-squared = {reg['R_squared']:.6f}")
    print(f"      p-value   = {reg['p_value']:.6e}")

    # Step 3: ANOVA
    print("\n[3/4] Running one-way ANOVA across injection groups...")
    anova = run_anova(df)
    print(f"      F-statistic = {anova['F_statistic']:.4f}")
    print(f"      p-value     = {anova['p_value']:.6e}")

    # Step 4: Generate figures
    print("\n[4/4] Generating publication figures...")
    generate_figure_miss_vs_injection(df, reg)
    generate_figure_detection_rate(df)

    # Save statistics summary
    summary = {
        "experiment": "miss_distance_proportionality",
        "regression": reg,
        "anova":      anova,
        "n_injection_rates":  len(INJECTION_RATES),
        "n_monte_carlo_runs": N_MONTE_CARLO,
        "injection_angle_deg": INJECTION_ANGLE
    }
    summary_path = os.path.join(
        DATA_DIR, "miss_proportionality_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n      Saved summary: {summary_path}")

    # Hypothesis verdict
    print("\n" + "=" * 60)
    print("MISS DISTANCE PROPORTIONALITY HYPOTHESIS VERDICT:")
    if reg["R_squared"] > 0.95 and anova["p_value"] < 0.05:
        print("  SUPPORTED")
        print(f"  R^2 = {reg['R_squared']:.4f} > 0.95  [OK]")
        print(f"  ANOVA p = {anova['p_value']:.2e} < 0.05  [OK]")
        print(f"  Ca (proportionality coeff) = {reg['Ca']:.4f}")
    else:
        print("  FAILED - see retry instructions")
        if reg["R_squared"] <= 0.95:
            print(f"  R^2 = {reg['R_squared']:.4f} <= 0.95  [X]")
        if anova["p_value"] >= 0.05:
            print(f"  ANOVA p = {anova['p_value']:.4f} >= 0.05  [X]")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
