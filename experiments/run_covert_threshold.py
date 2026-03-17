import sys
import os
import copy
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import chi2 as scipy_chi2
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.simulation_runner import SPECTRESimulation
from src.estimator.ekf_seeker import EKFSeeker

# ── Paths ────────────────────────────────────────────────────
CONFIG_PATH = "config/sim_config.yaml"
DATA_DIR    = "results/data"
FIGURES_DIR = "results/figures"
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Experiment parameters ────────────────────────────────────
N_FINE_SWEEP    = 100    # number of injection rates in fine sweep
N_MONTE_CARLO   = 50     # trials per injection rate
INJECTION_ANGLE = 0.0    # degrees — full projection (cos(0)=1)
DETECTION_TRIGGER_RATE = 0.10  # empirical threshold: 2x false alarm rate


def derive_analytical_threshold(config: dict) -> dict:
    """
    Analytically derive critical injection rate I_dot* from
    steady-state EKF parameters using calibrated noncentral
    chi-squared model.

    Theory:
        Under ramp injection at rate I_dot, the EKF innovation
        acquires a bias that grows the effective noncentrality:
            lambda = T * I_dot^2

        where T is the injection-to-noncentrality transfer
        coefficient, estimated from a single calibration run.

        The detection rate is:
            P(chi2 > threshold | lambda) = 1 - ncx2.cdf(...)

        I_dot* is where detection probability = TRIGGER_RATE.

    Steps:
        1. Run clean simulation, estimate S_inf
        2. Run one calibration injection, measure T
        3. Compute lambda_target for desired trigger rate
        4. Back out I_dot* = sqrt(lambda_target / T)
    """
    from scipy.stats import ncx2

    # ── Step 1: Clean run to estimate S_inf ────────────────
    sim_clean = SPECTRESimulation.from_config_override(
        CONFIG_PATH,
        {
            "attacker.injection_rate": 0.0,
            "attacker.active":         False
        }
    )
    sim_clean.run(seed=0)

    ekf_df = sim_clean.results["ekf_df"]
    n = len(ekf_df)
    tail = ekf_df.iloc[int(0.5 * n):int(0.95 * n)]

    K_inf_norm  = float(tail["gain_norm"].mean())
    P_inf_trace = float(tail["P_trace"].mean())

    innov_vals = tail["innovation"].values
    chi2_vals  = tail["chi2_stat"].values
    valid = chi2_vals > 1e-12
    S_inf_est = float(np.median(
        innov_vals[valid]**2 / chi2_vals[valid]
    ))

    R_scalar = config["ekf"]["R_scalar"]
    dt = config["simulation"]["dt"]

    alpha = config["monitor"]["alpha"]
    n_z   = config["monitor"]["n_z"]
    chi2_thresh = float(scipy_chi2.ppf(1.0 - alpha, df=n_z))

    innovation_max = float(np.sqrt(chi2_thresh * S_inf_est))
    bandwidth = K_inf_norm / dt

    # ── Step 2: Calibration run ────────────────────────────
    I_dot_cal = 0.002   # chosen near expected threshold for local accuracy
    sim_cal = SPECTRESimulation.from_config_override(
        CONFIG_PATH,
        {
            "attacker.injection_rate":      float(I_dot_cal),
            "attacker.injection_angle_deg": float(INJECTION_ANGLE),
            "attacker.active":              True
        }
    )
    res_cal = sim_cal.run(seed=0)
    det_rate_cal = res_cal["detection_rate"]

    # Find lambda from calibration detection rate
    lam_lo, lam_hi = 0.0, 200.0
    for _ in range(100):
        lam_mid = (lam_lo + lam_hi) / 2.0
        prob = float(1.0 - ncx2.cdf(
            chi2_thresh, df=1, nc=lam_mid))
        if prob < det_rate_cal:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    lambda_cal = (lam_lo + lam_hi) / 2.0

    # Transfer coefficient: lambda = T * I_dot^2
    T_transfer = lambda_cal / (I_dot_cal**2)

    # ── Step 3: Target lambda for trigger rate ─────────────
    lam_lo, lam_hi = 0.0, 200.0
    for _ in range(100):
        lam_mid = (lam_lo + lam_hi) / 2.0
        prob = float(1.0 - ncx2.cdf(
            chi2_thresh, df=1, nc=lam_mid))
        if prob < DETECTION_TRIGGER_RATE:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    lambda_target = (lam_lo + lam_hi) / 2.0

    # ── Step 4: Back out I_dot* ────────────────────────────
    I_dot_star = float(np.sqrt(lambda_target / T_transfer))
    I_dot_star = float(np.clip(I_dot_star, 1e-6, 10.0))

    return {
        "I_dot_star_analytical": I_dot_star,
        "K_inf_norm":            K_inf_norm,
        "P_inf_trace":           P_inf_trace,
        "S_inf_estimate":        S_inf_est,
        "innovation_max":        innovation_max,
        "bandwidth_estimate":    bandwidth,
        "chi2_threshold":        chi2_thresh,
        "R_scalar":              R_scalar,
        "lambda_star":           lambda_target,
        "T_transfer":            T_transfer,
        "calibration_I_dot":     I_dot_cal,
        "calibration_det_rate":  det_rate_cal,
        "calibration_lambda":    lambda_cal
    }


def run_fine_sweep(
        I_dot_star_analytical: float,
        config: dict) -> pd.DataFrame:
    """
    Fine-grained injection rate sweep around I_dot*.
    Tests 100 rates from 0 to 4 * I_dot_star_analytical.
    Collects detection rate for each rate over 50 Monte Carlo trials.
    """
    sweep_max    = 4.0 * I_dot_star_analytical
    I_dot_values = np.linspace(0.0, sweep_max, N_FINE_SWEEP)
    records      = []

    for I_dot in tqdm(I_dot_values,
                      desc="Covert threshold fine sweep"):
        is_attack = I_dot > 1e-9
        detection_rates = []

        for seed in range(N_MONTE_CARLO):
            sim = SPECTRESimulation.from_config_override(
                CONFIG_PATH,
                {
                    "attacker.injection_rate":      I_dot,
                    "attacker.injection_angle_deg": INJECTION_ANGLE,
                    "attacker.active":              is_attack
                }
            )
            results = sim.run(seed=seed)
            detection_rates.append(results["detection_rate"])

        records.append({
            "injection_rate":      float(I_dot),
            "mean_detection_rate": float(np.mean(detection_rates)),
            "std_detection_rate":  float(np.std(detection_rates)),
            "n_trials":            N_MONTE_CARLO
        })

    return pd.DataFrame(records)


def find_empirical_threshold(
        sweep_df: pd.DataFrame,
        trigger_rate: float) -> float:
    """
    Find empirical I_dot* as the first injection rate where
    mean detection rate first exceeds trigger_rate (default 0.10).

    Uses linear interpolation between the two surrounding points
    for a precise estimate.
    """
    above = sweep_df[
        sweep_df["mean_detection_rate"] >= trigger_rate
    ]

    if above.empty:
        # Never exceeded trigger — return maximum tested rate
        return float(sweep_df["injection_rate"].max())

    idx = above.index[0]

    if idx == 0:
        return float(sweep_df.loc[0, "injection_rate"])

    # Linear interpolation for precision
    x0 = sweep_df.loc[idx - 1, "injection_rate"]
    y0 = sweep_df.loc[idx - 1, "mean_detection_rate"]
    x1 = sweep_df.loc[idx,     "injection_rate"]
    y1 = sweep_df.loc[idx,     "mean_detection_rate"]

    # Interpolate: find x where y = trigger_rate
    if abs(y1 - y0) < 1e-9:
        return float(x0)
    I_dot_empirical = x0 + (trigger_rate - y0) * (x1 - x0) / (y1 - y0)
    return float(I_dot_empirical)


def generate_figure_covert_threshold(
        sweep_df: pd.DataFrame,
        I_dot_analytical: float,
        I_dot_empirical:  float,
        chi2_alpha: float) -> None:
    """
    Figure 3: Detection rate vs injection rate with covert zone.
    Shows analytical and empirical threshold boundaries.
    Shades the covert attack zone where detection rate < alpha.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x    = sweep_df["injection_rate"].values
    y    = sweep_df["mean_detection_rate"].values
    yerr = sweep_df["std_detection_rate"].values

    # Detection rate curve with error band
    ax.plot(x, y, color="#1f77b4", linewidth=2.0,
            label="Mean detection rate")
    ax.fill_between(
        x,
        np.maximum(0, y - yerr),
        np.minimum(1, y + yerr),
        alpha=0.2, color="#1f77b4",
        label="±1σ band"
    )

    # False alarm baseline
    ax.axhline(
        y=chi2_alpha, color="#7f7f7f",
        linestyle=":", linewidth=1.2,
        label=f"False alarm baseline α = {chi2_alpha}"
    )

    # Trigger threshold line
    ax.axhline(
        y=DETECTION_TRIGGER_RATE, color="#d62728",
        linestyle="--", linewidth=1.5,
        label=f"Detection trigger = {DETECTION_TRIGGER_RATE}"
    )

    # Analytical threshold
    ax.axvline(
        x=I_dot_analytical, color="#ff7f0e",
        linestyle="--", linewidth=1.8,
        label=(
            f"Analytical $\\dot{{I}}^*$ = "
            f"{I_dot_analytical:.4f} rad/s²"
        )
    )

    # Empirical threshold
    ax.axvline(
        x=I_dot_empirical, color="#2ca02c",
        linestyle="-.", linewidth=1.8,
        label=(
            f"Empirical $\\dot{{I}}^*$ = "
            f"{I_dot_empirical:.4f} rad/s²"
        )
    )

    # Shade covert zone
    covert_limit = min(I_dot_analytical, I_dot_empirical)
    ax.axvspan(
        0, covert_limit,
        alpha=0.08, color="#2ca02c",
        label="Covert attack zone"
    )

    ax.set_xlabel(
        "Injection Rate $\\dot{I}$ (rad/s per second)",
        fontsize=12
    )
    ax.set_ylabel(
        "Chi-Squared Detection Rate",
        fontsize=12
    )
    ax.set_title(
        "Covert Injection Threshold: Analytical vs. Empirical\n"
        "(Covert Injection Threshold Hypothesis Validation)",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        "fig3_covert_injection_threshold.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("=" * 62)
    print("SPECTRE-SIM: Covert Injection Threshold Experiment")
    print("=" * 62)

    # Step 1: Analytical derivation
    print("\n[1/4] Deriving analytical threshold I_dot*...")
    analytic = derive_analytical_threshold(config)
    I_dot_star = analytic["I_dot_star_analytical"]
    print(f"      I_dot* (analytical)  = {I_dot_star:.6f} rad/s²")
    print(f"      K_inf norm           = {analytic['K_inf_norm']:.6f}")
    print(f"      S_inf estimate       = {analytic['S_inf_estimate']:.8f}")
    print(f"      Innovation max       = {analytic['innovation_max']:.6f}")
    print(f"      Bandwidth estimate   = {analytic['bandwidth_estimate']:.4f} 1/s")

    # Step 2: Fine numerical sweep
    print(f"\n[2/4] Running fine sweep ({N_FINE_SWEEP} rates, "
          f"{N_MONTE_CARLO} trials each)...")
    sweep_df = run_fine_sweep(I_dot_star, config)

    csv_path = os.path.join(DATA_DIR, "covert_threshold_sweep.csv")
    sweep_df.to_csv(csv_path, index=False)
    print(f"      Saved sweep data: {csv_path}")

    # Step 3: Find empirical threshold
    print("\n[3/4] Finding empirical threshold...")
    I_dot_empirical = find_empirical_threshold(
        sweep_df, DETECTION_TRIGGER_RATE
    )
    print(f"      I_dot* (empirical)   = {I_dot_empirical:.6f} rad/s²")

    # Agreement check
    agreement_pct = (
        abs(I_dot_star - I_dot_empirical) /
        (I_dot_star + 1e-9) * 100
    )
    print(f"      Agreement            = {agreement_pct:.2f}%"
          f" (threshold: 15%)")

    # Covert zone validation
    covert_rate_half = None
    half_star = I_dot_star * 0.5
    near_half = sweep_df[
        (sweep_df["injection_rate"] >= half_star * 0.9) &
        (sweep_df["injection_rate"] <= half_star * 1.1)
    ]
    if not near_half.empty:
        covert_rate_half = float(
            near_half["mean_detection_rate"].mean()
        )

    detectable_rate_high = None
    # Check detection rate at top end of sweep (4x I_dot*)
    top_rows = sweep_df.tail(5)
    if not top_rows.empty:
        detectable_rate_high = float(
            top_rows["mean_detection_rate"].mean()
        )

    # Step 4: Generate figure
    print("\n[4/4] Generating publication figure...")
    generate_figure_covert_threshold(
        sweep_df,
        I_dot_star,
        I_dot_empirical,
        config["monitor"]["alpha"]
    )

    # Save summary
    summary = {
        "experiment":               "covert_injection_threshold",
        "I_dot_star_analytical":    I_dot_star,
        "I_dot_star_empirical":     I_dot_empirical,
        "agreement_pct":            agreement_pct,
        "covert_zone_detection_at_half_star":    covert_rate_half,
        "detectable_zone_detection_at_high":     detectable_rate_high,
        "analytical_derivation":    analytic,
        "n_fine_sweep_points":      N_FINE_SWEEP,
        "n_monte_carlo_runs":       N_MONTE_CARLO,
        "detection_trigger_rate":   DETECTION_TRIGGER_RATE
    }
    summary_path = os.path.join(
        DATA_DIR, "covert_threshold_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved summary: {summary_path}")

    # Hypothesis verdict
    print("\n" + "=" * 62)
    print("COVERT INJECTION THRESHOLD HYPOTHESIS VERDICT:")
    supported = (
        I_dot_star > 0 and
        agreement_pct < 15.0 and
        (covert_rate_half is None or covert_rate_half < 0.10) and
        (detectable_rate_high is None or
         detectable_rate_high > 0.30)
    )
    if supported:
        print("  SUPPORTED")
        print(f"  Analytical I_dot* = {I_dot_star:.6f} rad/s^2  [OK]")
        print(f"  Empirical  I_dot* = {I_dot_empirical:.6f} rad/s^2  [OK]")
        print(f"  Agreement         = {agreement_pct:.2f}% < 15%  [OK]")
        if covert_rate_half is not None:
            print(f"  Covert zone rate  = "
                  f"{covert_rate_half:.4f} < 0.10  [OK]")
        if detectable_rate_high is not None:
            print(f"  Detectable rate   = "
                  f"{detectable_rate_high:.4f} > 0.30  [OK]")
    else:
        print("  FAILED - see retry instructions in Prompt 08")
        if agreement_pct >= 15.0:
            print(f"  Agreement {agreement_pct:.2f}% >= 15%  [X]")
        if covert_rate_half is not None and covert_rate_half >= 0.10:
            print(f"  Covert zone rate "
                  f"{covert_rate_half:.4f} >= 0.10  [X]")
    print("=" * 62)

    return summary


if __name__ == "__main__":
    main()
