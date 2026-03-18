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
INJECTION_RATES = [
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
    0.060, 0.065, 0.070, 0.080, 0.090, 0.10
]
N_MONTE_CARLO   = 100
INJECTION_ANGLE = 0.0   # degrees — full projection (cos(0)=1)
I_DOT_STAR_SPLIT = 0.05  # boundary between sub/super-threshold regimes

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


def fit_linear_regression_super_threshold(df: pd.DataFrame) -> dict:
    """
    Fit D_m = Ca * I_dot + b only on super-threshold points
    (I_dot > I_DOT_STAR_SPLIT).
    Returns segmented-model statistics dictionary.
    """
    df_reg = df[df["injection_rate"] > I_DOT_STAR_SPLIT].copy()
    if len(df_reg) < 2:
        return {
            "Ca": None,
            "intercept": None,
            "R_squared": None,
            "p_value": None,
            "std_err": None,
            "n_points": int(len(df_reg))
        }

    x = df_reg["injection_rate"].values
    y = df_reg["mean_miss"].values

    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(x, y)

    return {
        "Ca":          float(slope),
        "intercept":   float(intercept),
        "R_squared":   float(r_value ** 2),
        "p_value":     float(p_value),
        "std_err":     float(std_err),
        "n_points":    int(len(df_reg))
    }


def run_anova(rate_values: list[float]) -> dict:
    """
    One-way ANOVA across selected injection rate groups.
    Uses per-trial data collected during the sweep (no re-run needed).
    Falls back to N/A if per-trial data is not available.
    """
    groups = []
    for I_dot in rate_values:
        if I_dot in _per_trial_data and _per_trial_data[I_dot]:
            groups.append(_per_trial_data[I_dot])
        else:
            return {
                "F_statistic": None,
                "p_value": None,
                "n_groups": len(rate_values),
                "n_per_group": N_MONTE_CARLO
            }

    if len(groups) < 2:
        return {
            "F_statistic": None,
            "p_value": None,
            "n_groups": len(rate_values),
            "n_per_group": N_MONTE_CARLO
        }

    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "F_statistic": float(f_stat),
        "p_value":     float(p_value),
        "n_groups":    len(rate_values),
        "n_per_group": N_MONTE_CARLO
    }


def compute_regime_statistics(df: pd.DataFrame) -> dict:
    """
    Compute two-regime segmented statistics around I_DOT_STAR_SPLIT.
    """
    sub_df = df[df["injection_rate"] < I_DOT_STAR_SPLIT].copy()
    super_df = df[df["injection_rate"] > I_DOT_STAR_SPLIT].copy()

    reg_super = fit_linear_regression_super_threshold(df)
    anova_super = run_anova(
        sorted(super_df["injection_rate"].tolist())
    )

    if len(sub_df):
        sub_trials = []
        for rate in sub_df["injection_rate"].tolist():
            sub_trials.extend(_per_trial_data.get(rate, []))
        if sub_trials:
            sub_trials = np.asarray(sub_trials, dtype=float)
            sub_mean = float(np.mean(sub_trials))
            sub_std = float(np.std(sub_trials, ddof=0))
        else:
            sub_mean = float(sub_df["mean_miss"].mean())
            sub_std = float(sub_df["std_miss"].mean())
    else:
        sub_mean = None
        sub_std = None

    return {
        "regime": "two-regime segmented",
        "I_dot_star_split": float(I_DOT_STAR_SPLIT),
        "super_threshold": {
            "Ca": reg_super["Ca"],
            "intercept": reg_super["intercept"],
            "R_squared": reg_super["R_squared"],
            "p_value": reg_super["p_value"],
            "ANOVA_p": anova_super["p_value"],
            "ANOVA_F": anova_super["F_statistic"],
            "n_points": int(len(super_df))
        },
        "sub_threshold": {
            "mean_miss_m": sub_mean,
            "std_miss_m": sub_std,
            "n_points": int(len(sub_df))
        },
        # Backward-compatible mirrors used by downstream tooling.
        "regression": {
            "Ca": reg_super["Ca"],
            "intercept": reg_super["intercept"],
            "R_squared": reg_super["R_squared"],
            "p_value": reg_super["p_value"],
            "std_err": reg_super["std_err"]
        },
        "anova": anova_super
    }


def generate_figure_miss_vs_injection(
        df: pd.DataFrame, segmented_stats: dict) -> None:
    """
    Figure 1: Miss distance vs injection rate.
    Two-regime plot with covert threshold, proportionality boundary,
    error bars, and super-threshold OLS regression line.
    """
    I_STAR_COVERT = 0.00165

    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["injection_rate"].values
    y = df["mean_miss"].values
    yerr = df["std_miss"].values

    # --- Zone shading ---
    # Boundary placed between the last sub-proportional point (0.04)
    # and the transition point (0.05) so 0.05 falls in the orange zone.
    shade_edge = (0.04 + I_DOT_STAR_SPLIT) / 2.0   # 0.045
    ax.axvspan(
        -0.005, shade_edge,
        color="gray", alpha=0.13, zorder=0,
        label="EKF-compensated zone"
    )
    ax.axvspan(
        shade_edge, max(x) + 0.005,
        color="#ff7f0e", alpha=0.08, zorder=0,
        label="Super-threshold zone"
    )

    # --- Covert detection threshold İ* from Experiment 2 ---
    ax.axvline(
        I_STAR_COVERT, color="#2ca02c",
        linestyle=":", linewidth=1.3, alpha=0.85, zorder=2
    )
    ax.annotate(
        r"$\dot{I}^{*}\!=\!0.00165$" + "\n(covert limit)",
        xy=(I_STAR_COVERT, max(y) * 0.38),
        xytext=(0.015, max(y) * 0.38),
        fontsize=8, color="#2ca02c", va="center",
        arrowprops=dict(
            arrowstyle="->", color="#2ca02c", lw=1.0
        )
    )

    # --- Proportionality boundary (regression domain) ---
    ax.axvline(
        I_DOT_STAR_SPLIT, color="black",
        linestyle="--", linewidth=1.2, zorder=2
    )
    ax.annotate(
        "Regression\nboundary",
        xy=(I_DOT_STAR_SPLIT, max(y) * 0.50),
        xytext=(I_DOT_STAR_SPLIT - 0.013, max(y) * 0.50),
        fontsize=8, color="black", va="center", ha="right",
        arrowprops=dict(
            arrowstyle="->", color="black", lw=1.0
        )
    )

    # --- Data points with error bars ---
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o", color="#1f77b4",
        ecolor="#aec7e8", elinewidth=1.5,
        capsize=4, markersize=6, zorder=3,
        label=r"Monte Carlo mean $\pm\,1\sigma$"
    )

    # --- Super-threshold regression line (data range only) ---
    reg = segmented_stats["super_threshold"]
    super_mask = x > I_DOT_STAR_SPLIT
    x_super = x[super_mask]
    x_line = np.linspace(min(x_super), max(x_super), 200)
    y_line = reg["Ca"] * x_line + reg["intercept"]
    ax.plot(
        x_line, y_line,
        "--", color="#d62728", linewidth=1.8, zorder=4,
        label=(
            f"OLS fit (super-threshold): "
            f"$D_m = {reg['Ca']:.1f}"
            r" \cdot \dot{I} + \varepsilon$"
            f"\n$R^2 = {reg['R_squared']:.4f}$"
        )
    )

    # --- Axes ---
    ax.set_xlabel(
        r"Injection Rate $\dot{I}$ (rad/s$^2$)", fontsize=12
    )
    ax.set_ylabel(r"Miss Distance $D_m$ (m)", fontsize=12)
    ax.set_title(
        "Miss Distance vs. Injection Rate\n"
        "(Two-Regime Proportionality Validation)",
        fontsize=13
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.008, max(x) + 0.008)
    ax.set_ylim(bottom=-200)

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
        r"Injection Rate $\dot{I}$ (rad/s$^2$)",
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

    # Step 2: Two-regime segmented analysis
    print("\n[2/4] Fitting two-regime segmented model...")
    segmented = compute_regime_statistics(df)
    super_stats = segmented["super_threshold"]
    sub_stats = segmented["sub_threshold"]
    print(f"      I_dot* split = {I_DOT_STAR_SPLIT:.6f} rad/s^2")
    print(f"      Ca_super     = {super_stats['Ca']:.4f}")
    print(f"      R2_super     = {super_stats['R_squared']:.6f}")
    print(f"      ANOVA p_super= {super_stats['ANOVA_p']:.6e}")
    print(f"      mean_miss_sub= {sub_stats['mean_miss_m']:.4f} m")
    print(f"      std_miss_sub = {sub_stats['std_miss_m']:.4f} m")

    # Step 3: ANOVA
    print("\n[3/4] Running one-way ANOVA across super-threshold "
          "injection groups...")
    anova = segmented["anova"]
    print(f"      F-statistic = {anova['F_statistic']:.4f}")
    print(f"      p-value     = {anova['p_value']:.6e}")

    # Step 4: Generate figures
    print("\n[4/4] Generating publication figures...")
    generate_figure_miss_vs_injection(df, segmented)
    generate_figure_detection_rate(df)

    # Save statistics summary
    summary = {
        "experiment": "miss_distance_proportionality",
        "regime": segmented["regime"],
        "I_dot_star_split": segmented["I_dot_star_split"],
        "super_threshold": segmented["super_threshold"],
        "sub_threshold": segmented["sub_threshold"],
        "regression": segmented["regression"],
        "anova": segmented["anova"],
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
    if (super_stats["R_squared"] > 0.95 and
            super_stats["ANOVA_p"] < 0.05):
        print("  SUPPORTED")
        print(f"  R^2_super = {super_stats['R_squared']:.4f} "
              f"> 0.95  [OK]")
        print(f"  ANOVA p_super = {super_stats['ANOVA_p']:.2e} "
              f"< 0.05  [OK]")
        print(f"  Ca_super (proportionality coeff) = "
              f"{super_stats['Ca']:.4f}")
    else:
        print("  FAILED - see retry instructions")
        if super_stats["R_squared"] <= 0.95:
            print(f"  R^2_super = {super_stats['R_squared']:.4f} "
                  f"<= 0.95  [X]")
        if super_stats["ANOVA_p"] >= 0.05:
            print(f"  ANOVA p_super = {super_stats['ANOVA_p']:.4f} "
                  f">= 0.05  [X]")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
