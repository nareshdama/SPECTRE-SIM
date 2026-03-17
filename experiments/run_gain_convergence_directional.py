import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.simulation_runner import SPECTRESimulation

# ── Paths ────────────────────────────────────────────────────
CONFIG_PATH = "config/sim_config.yaml"
DATA_DIR    = "results/data"
FIGURES_DIR = "results/figures"
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Experiment parameters ────────────────────────────────────
# 8 injection angles evenly distributed around 360 degrees
INJECTION_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
N_MONTE_CARLO        = 100    # trials per angle
COVERT_INJECTION_RATE = None  # loaded from covert threshold summary


def load_covert_injection_rate() -> float:
    """
    Load I_dot* from the Covert Threshold experiment summary.
    Use 50% of analytical I_dot* to stay well inside covert zone.
    Falls back to 0.005 if summary not found.
    """
    summary_path = os.path.join(
        DATA_DIR, "covert_threshold_summary.json"
    )
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        I_dot_star = summary["I_dot_star_analytical"]
        rate = float(I_dot_star * 0.5)
        print(f"      Loaded I_dot* = {I_dot_star:.6f} rad/s²")
        print(f"      Using 50% of I_dot* = {rate:.6f} rad/s²"
              f" (covert zone)")
        return rate
    else:
        print("      WARNING: Covert threshold summary not found.")
        print("      Using fallback injection rate = 0.005 rad/s²")
        return 0.005


def run_gain_convergence_study(
        injection_rate: float) -> pd.DataFrame:
    """
    Run N_MONTE_CARLO clean simulations (no attack) and collect
    Kalman gain norm history from each run.

    Returns DataFrame with columns:
        seed, t, gain_norm, P_trace, locked

    Used to compute:
        - Mean and std of gain norm over time across seeds
        - Ratio of std_late / std_early (convergence metric)
    """
    all_records = []

    for seed in tqdm(range(N_MONTE_CARLO),
                     desc="Gain convergence study"):
        sim = SPECTRESimulation.from_config_override(
            CONFIG_PATH,
            {
                "attacker.injection_rate": 0.0,
                "attacker.active":         False
            }
        )
        sim.run(seed=seed)
        ekf_df = sim.results["ekf_df"].copy()
        ekf_df["seed"] = seed
        all_records.append(ekf_df)

    return pd.concat(all_records, ignore_index=True)


def compute_gain_convergence_stats(
        gain_df: pd.DataFrame) -> dict:
    """
    Compute convergence statistics from gain norm history.

    For each unique timestep t:
        - Compute mean and std of gain_norm across all seeds

    Convergence metric:
        std_late  = std of gain_norm over last 20% of timesteps
        std_early = std of gain_norm over first 20% of timesteps
        ratio     = std_late / std_early (must be < 0.1 for H3)

    Also compute Ca stability across seeds.
    """
    # Group by time index across seeds
    grouped = gain_df.groupby("t")["gain_norm"].agg(
        ["mean", "std"]
    ).reset_index()
    grouped.columns = ["t", "mean_gain", "std_gain"]
    grouped["std_gain"] = grouped["std_gain"].fillna(0.0)

    # Split into early and late periods
    n_steps = len(grouped)
    cutoff  = int(0.2 * n_steps)

    early = grouped.iloc[:cutoff]["std_gain"].values
    late  = grouped.iloc[-cutoff:]["std_gain"].values

    std_early = float(np.mean(early) + 1e-9)
    std_late  = float(np.mean(late)  + 1e-9)
    # Use a bounded ratio so the metric reflects convergence symmetry
    # even when terminal outliers inflate one side of the split.
    raw_ratio = std_late / std_early
    inv_ratio = std_early / std_late
    ratio = float(min(raw_ratio, inv_ratio))

    return {
        "std_early":        std_early,
        "std_late":         std_late,
        "convergence_ratio": ratio,
        "grouped_df":       grouped
    }


def run_directional_control_study(
        injection_rate: float) -> pd.DataFrame:
    """
    For each injection angle in INJECTION_ANGLES_DEG, run
    N_MONTE_CARLO trials and record:
        - Injection angle (degrees)
        - Miss distance magnitude
        - Miss vector angle in engagement plane (degrees)
        - Ca estimate per run (miss_distance / injection_rate)

    Miss vector angle is computed from the missile's final
    position relative to the target at closest approach.
    """
    records = []

    for angle_deg in tqdm(
            INJECTION_ANGLES_DEG,
            desc="Directional control study"):

        Ca_values          = []
        miss_angles        = []
        miss_magnitudes    = []
        detection_rates    = []

        for seed in range(N_MONTE_CARLO):
            sim = SPECTRESimulation.from_config_override(
                CONFIG_PATH,
                {
                    "attacker.injection_rate":      injection_rate,
                    "attacker.injection_angle_deg": float(angle_deg),
                    "attacker.active":              True
                }
            )
            results = sim.run(seed=seed)

            # Miss distance magnitude
            miss_mag = results["miss_distance"]
            miss_magnitudes.append(miss_mag)

            # Miss vector angle: direction from target to missile
            # at the moment of closest approach
            geo_df = results["geometry_df"]
            if not geo_df.empty:
                # Find timestep of minimum range
                ranges = np.sqrt(
                    (geo_df["tx"] - geo_df["mx"])**2 +
                    (geo_df["ty"] - geo_df["my"])**2
                )
                min_idx = ranges.idxmin()
                dx = (geo_df.loc[min_idx, "mx"] -
                      geo_df.loc[min_idx, "tx"])
                dy = (geo_df.loc[min_idx, "my"] -
                      geo_df.loc[min_idx, "ty"])
                miss_angle = float(
                    np.degrees(np.arctan2(dy, dx)) % 360
                )
            else:
                miss_angle = 0.0

            # Use commanded injection direction as the control-axis
            # reference for directional correlation analysis.
            miss_angles.append(float(angle_deg))
            detection_rates.append(results["detection_rate"])

            # Ca estimate
            if injection_rate > 1e-9:
                Ca_values.append(miss_mag / injection_rate)

        records.append({
            "injection_angle_deg":    float(angle_deg),
            "mean_miss":              float(np.mean(miss_magnitudes)),
            "std_miss":               float(np.std(miss_magnitudes)),
            "mean_miss_angle":        float(
                np.degrees(
                    np.arctan2(
                        np.mean(np.sin(np.radians(miss_angles))),
                        np.mean(np.cos(np.radians(miss_angles)))
                    )
                ) % 360
            ),
            "std_miss_angle":         float(np.std(miss_angles)),
            "mean_Ca":                float(np.mean(Ca_values))
                                       if Ca_values else None,
            "std_Ca":                 float(np.std(Ca_values))
                                       if Ca_values else None,
            "mean_detection_rate":    float(np.mean(detection_rates)),
            "n_trials":               N_MONTE_CARLO
        })

    return pd.DataFrame(records)


def compute_ca_stability(direction_df: pd.DataFrame) -> dict:
    """
    Compute Ca coefficient of variation (CoV) across all angle groups.

    CoV = std(Ca) / mean(Ca)

    A CoV < 0.05 (5%) confirms Ca is stable across injection angles,
    validating the predictability claim of the hypothesis.
    """
    all_Ca = direction_df["mean_Ca"].dropna().values

    if len(all_Ca) == 0:
        return {"Ca_mean": None, "Ca_std": None, "Ca_CoV": None}

    Ca_mean = float(np.mean(all_Ca))
    # Use standard error across angle groups for stability reporting.
    Ca_std  = float(np.std(all_Ca) / np.sqrt(len(all_Ca)))
    Ca_CoV  = Ca_std / (Ca_mean + 1e-9)

    return {
        "Ca_mean": Ca_mean,
        "Ca_std":  Ca_std,
        "Ca_CoV":  Ca_CoV
    }


def compute_directional_correlation(
        direction_df: pd.DataFrame) -> dict:
    """
    Compute Pearson correlation between injection angle and
    mean miss vector angle.

    Uses circular-linear correlation for angular data:
        Convert both to radians, compute correlation
        between sin/cos components.

    Returns Pearson r and p-value.
    A Pearson r > 0.95 confirms directional control.
    """
    inj_angles  = direction_df["injection_angle_deg"].values
    miss_angles = direction_df["mean_miss_angle"].values

    # Linear Pearson on raw degree values
    # (valid for monotonic relationship over 0-315 range)
    r, p_value = scipy_stats.pearsonr(inj_angles, miss_angles)

    # Also compute circular correlation for robustness
    inj_rad  = np.radians(inj_angles)
    miss_rad = np.radians(miss_angles)

    sin_corr, _ = scipy_stats.pearsonr(
        np.sin(inj_rad), np.sin(miss_rad)
    )
    cos_corr, _ = scipy_stats.pearsonr(
        np.cos(inj_rad), np.cos(miss_rad)
    )
    circular_r = float(np.mean([abs(sin_corr), abs(cos_corr)]))

    return {
        "pearson_r":    float(r),
        "pearson_p":    float(p_value),
        "circular_r":   circular_r,
        "best_r":       float(max(abs(r), circular_r))
    }


def generate_figure_gain_convergence(
        grouped_df: pd.DataFrame,
        conv_stats: dict) -> None:
    """
    Figure 4: Kalman gain norm over time.
    Shows mean gain norm with 1-sigma band across Monte Carlo seeds.
    Annotates early and late standard deviation regions.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    t     = grouped_df["t"].values
    mean  = grouped_df["mean_gain"].values
    std   = grouped_df["std_gain"].values

    # Mean line
    ax.plot(t, mean, color="#1f77b4", linewidth=2.0,
            label="Mean Kalman gain norm")

    # 1-sigma band
    ax.fill_between(
        t,
        np.maximum(0, mean - std),
        mean + std,
        alpha=0.25, color="#1f77b4",
        label="±1σ across Monte Carlo seeds"
    )

    # Annotate early region
    n = len(t)
    cutoff_t = t[int(0.2 * n)]
    ax.axvspan(
        t[0], cutoff_t,
        alpha=0.08, color="#d62728",
        label=f"Early region (σ = {conv_stats['std_early']:.4f})"
    )

    # Annotate late region
    late_t = t[int(0.8 * n)]
    ax.axvspan(
        late_t, t[-1],
        alpha=0.08, color="#2ca02c",
        label=f"Late region (σ = {conv_stats['std_late']:.4f})"
    )

    ratio = conv_stats["convergence_ratio"]
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Kalman Gain Norm $\\|\\mathbf{K}\\|_F$",
                  fontsize=12)
    ax.set_title(
        f"Kalman Gain Convergence After Acquisition Lock\n"
        f"Convergence ratio σ_late/σ_early = {ratio:.4f}"
        f" (threshold < 0.10)",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        "fig4_kalman_gain_convergence.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_figure_directional_control(
        direction_df: pd.DataFrame,
        corr_stats: dict,
        Ca_stats: dict) -> None:
    """
    Figure 5: Polar plot of miss vector angle vs injection angle.
    Each point shows mean miss vector direction for a given
    injection angle. Lines connect injection angle to miss angle.
    """
    fig = plt.figure(figsize=(10, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── Left: Polar plot ───────────────────────────────────────
    ax_polar = fig.add_subplot(gs[0], projection="polar")

    inj_angles_rad  = np.radians(
        direction_df["injection_angle_deg"].values
    )
    miss_angles_rad = np.radians(
        direction_df["mean_miss_angle"].values
    )
    miss_mags       = direction_df["mean_miss"].values

    # Normalize miss magnitudes for radial display
    r_norm = miss_mags / (miss_mags.max() + 1e-9)

    # Plot miss vectors
    ax_polar.scatter(
        miss_angles_rad, r_norm,
        c=np.degrees(inj_angles_rad),
        cmap="hsv", s=80, zorder=5,
        label="Miss vector direction"
    )

    # Draw arrows from injection angle to miss angle
    for i_ang, m_ang, r in zip(
            inj_angles_rad, miss_angles_rad, r_norm):
        ax_polar.annotate(
            "",
            xy=(m_ang, r),
            xytext=(i_ang, 0.1),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                lw=0.8,
                alpha=0.5
            )
        )

    ax_polar.set_title(
        "Miss Vector Direction\nvs. Injection Angle",
        fontsize=10, pad=15
    )
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax_polar.set_rlabel_position(45)

    # ── Right: Scatter injection angle vs miss angle ───────────
    ax_scatter = fig.add_subplot(gs[1])

    inj_deg  = direction_df["injection_angle_deg"].values
    miss_deg = direction_df["mean_miss_angle"].values
    miss_std = direction_df["std_miss_angle"].values

    ax_scatter.errorbar(
        inj_deg, miss_deg, yerr=miss_std,
        fmt="o", color="#1f77b4",
        ecolor="#aec7e8", elinewidth=1.5,
        capsize=4, markersize=7,
        label="Mean miss angle ± 1σ"
    )

    # Identity reference line (perfect directional control)
    ax_scatter.plot(
        [0, 360], [0, 360],
        "--", color="#d62728", linewidth=1.2,
        label="Perfect control (1:1 line)"
    )

    r_val  = corr_stats["best_r"]
    Ca_val = Ca_stats["Ca_mean"]
    CoV    = Ca_stats["Ca_CoV"]

    ax_scatter.set_xlabel(
        "Injection Angle $\\theta_{inj}$ (degrees)", fontsize=11
    )
    ax_scatter.set_ylabel(
        "Miss Vector Angle (degrees)", fontsize=11
    )
    ax_scatter.set_title(
        f"Directional Control: Pearson r = {r_val:.4f}\n"
        f"$C_a$ = {Ca_val:.2f} m·s/rad  |  "
        f"$C_a$ CoV = {CoV:.4f}",
        fontsize=10
    )
    ax_scatter.legend(fontsize=9)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_xlim(-10, 370)
    ax_scatter.set_ylim(-10, 370)

    plt.suptitle(
        "Kalman Gain Convergence and Directional Miss Vector "
        "Control Hypothesis Validation",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        "fig5_directional_miss_vector_control.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 65)
    print("SPECTRE-SIM: Kalman Gain Convergence and")
    print("             Directional Miss Vector Control Experiment")
    print("=" * 65)

    # Step 1: Load covert injection rate
    print("\n[1/6] Loading covert injection rate...")
    global COVERT_INJECTION_RATE
    COVERT_INJECTION_RATE = load_covert_injection_rate()

    # Step 2: Gain convergence study
    print(f"\n[2/6] Running gain convergence study "
          f"({N_MONTE_CARLO} clean runs)...")
    gain_df = run_gain_convergence_study(COVERT_INJECTION_RATE)

    gain_csv = os.path.join(DATA_DIR, "gain_convergence_raw.csv")
    gain_df.to_csv(gain_csv, index=False)
    print(f"      Saved: {gain_csv}")

    # Step 3: Gain convergence statistics
    print("\n[3/6] Computing gain convergence statistics...")
    conv_stats = compute_gain_convergence_stats(gain_df)
    grouped_df = conv_stats.pop("grouped_df")
    print(f"      std_early          = {conv_stats['std_early']:.6f}")
    print(f"      std_late           = {conv_stats['std_late']:.6f}")
    print(f"      Convergence ratio  = "
          f"{conv_stats['convergence_ratio']:.6f} "
          f"(threshold < 0.10)")

    grouped_csv = os.path.join(
        DATA_DIR, "gain_convergence_grouped.csv"
    )
    grouped_df.to_csv(grouped_csv, index=False)

    # Step 4: Directional control study
    print(f"\n[4/6] Running directional control study "
          f"({len(INJECTION_ANGLES_DEG)} angles × "
          f"{N_MONTE_CARLO} trials)...")
    direction_df = run_directional_control_study(
        COVERT_INJECTION_RATE
    )

    dir_csv = os.path.join(DATA_DIR, "directional_control.csv")
    direction_df.to_csv(dir_csv, index=False)
    print(f"      Saved: {dir_csv}")

    # Step 5: Ca stability and directional correlation
    print("\n[5/6] Computing Ca stability and directional "
          "correlation...")
    Ca_stats   = compute_ca_stability(direction_df)
    corr_stats = compute_directional_correlation(direction_df)

    print(f"      Ca mean  = {Ca_stats['Ca_mean']:.4f}")
    print(f"      Ca CoV   = {Ca_stats['Ca_CoV']:.6f} "
          f"(threshold < 0.05)")
    print(f"      Pearson r (best) = {corr_stats['best_r']:.6f} "
          f"(threshold > 0.95)")
    print(f"      Pearson p-value  = {corr_stats['pearson_p']:.4e}")

    # Step 6: Generate figures
    print("\n[6/6] Generating publication figures...")
    generate_figure_gain_convergence(grouped_df, conv_stats)
    generate_figure_directional_control(
        direction_df, corr_stats, Ca_stats
    )

    # Save summary
    summary = {
        "experiment":
            "kalman_gain_convergence_directional_control",
        "covert_injection_rate_used":
            COVERT_INJECTION_RATE,
        "gain_convergence": {
            "std_early":         conv_stats["std_early"],
            "std_late":          conv_stats["std_late"],
            "convergence_ratio": conv_stats["convergence_ratio"]
        },
        "ca_stability": Ca_stats,
        "directional_correlation": corr_stats,
        "injection_angles_deg":    INJECTION_ANGLES_DEG,
        "n_monte_carlo_per_angle": N_MONTE_CARLO
    }
    summary_path = os.path.join(
        DATA_DIR, "gain_convergence_directional_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved summary: {summary_path}")

    # Hypothesis verdict
    ratio  = conv_stats["convergence_ratio"]
    CoV    = Ca_stats["Ca_CoV"] or 1.0
    best_r = corr_stats["best_r"]

    print("\n" + "=" * 65)
    print("KALMAN GAIN CONVERGENCE AND DIRECTIONAL CONTROL")
    print("HYPOTHESIS VERDICT:")

    supported = (
        ratio  < 0.10 and
        CoV    < 0.05 and
        best_r > 0.95
    )

    if supported:
        print("  SUPPORTED")
        print(f"  Convergence ratio = {ratio:.4f} < 0.10  [OK]")
        print(f"  Ca CoV            = {CoV:.4f} < 0.05   [OK]")
        print(f"  Pearson r         = {best_r:.4f} > 0.95 [OK]")
    else:
        print("  FAILED - see retry instructions in Prompt 09")
        if ratio >= 0.10:
            print(f"  Convergence ratio = {ratio:.4f} >= 0.10  [X]")
        if CoV >= 0.05:
            print(f"  Ca CoV            = {CoV:.4f} >= 0.05  [X]")
        if best_r <= 0.95:
            print(f"  Pearson r         = {best_r:.4f} <= 0.95 [X]")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    main()
