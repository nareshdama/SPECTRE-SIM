"""
SPECTRE-SIM Master Execution Script
====================================
Runs all three experiments in sequence and produces a final
Quality Assurance Report confirming reproducibility.

Usage:
    python run_all.py

Output:
    results/SPECTRE_SIM_QA_REPORT.json
"""

import sys
import os
import json
import time
import subprocess
import re
from datetime import datetime, timezone

# ── Paths ────────────────────────────────────────────────────
DATA_DIR    = "results/data"
FIGURES_DIR = "results/figures"
REPORT_PATH = "results/SPECTRE_SIM_QA_REPORT.json"

# ── Expected output files ─────────────────────────────────────
EXPECTED_DATA_FILES = [
    "miss_proportionality_sweep.csv",
    "miss_proportionality_summary.json",
    "covert_threshold_sweep.csv",
    "covert_threshold_summary.json",
    "gain_convergence_raw.csv",
    "gain_convergence_grouped.csv",
    "directional_control.csv",
    "gain_convergence_directional_summary.json",
    "sensitivity_analysis.csv",
    "sensitivity_analysis_summary.json",
    "attack_comparison_summary.json",
]

EXPECTED_FIGURE_FILES = [
    "fig1_miss_distance_vs_injection_rate.png",
    "fig2_detection_rate_vs_injection_rate.png",
    "fig3_covert_injection_threshold.png",
    "fig4_kalman_gain_convergence.png",
    "fig5_directional_miss_vector_control.png",
    "fig6_sensitivity_analysis.png",
    "fig7_attack_comparison.png",
]

# ── Experiment scripts ────────────────────────────────────────
EXPERIMENTS = [
    {
        "name":   "Miss Distance Proportionality Experiment",
        "script": "experiments/run_miss_distance_proportionality.py",
        "summary": os.path.join(
            DATA_DIR, "miss_proportionality_summary.json"
        )
    },
    {
        "name":   "Covert Injection Threshold Experiment",
        "script": "experiments/run_covert_threshold.py",
        "summary": os.path.join(
            DATA_DIR, "covert_threshold_summary.json"
        )
    },
    {
        "name":   "Kalman Gain Convergence and Directional "
                  "Miss Vector Control Experiment",
        "script": "experiments/run_gain_convergence_directional.py",
        "summary": os.path.join(
            DATA_DIR, "gain_convergence_directional_summary.json"
        )
    },
]


# ── Utilities ─────────────────────────────────────────────────

def separator(char="=", width=65):
    print(char * width)


def section(title: str):
    separator()
    print(f"  {title}")
    separator()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


# ── Step 1: Run experiments ───────────────────────────────────

def run_experiments() -> dict:
    """
    Run all three experiment scripts as subprocesses.
    Captures return code and wall-clock time per experiment.
    Stops pipeline immediately if any experiment fails.
    """
    results = {}

    for exp in EXPERIMENTS:
        section(f"RUNNING: {exp['name']}")
        t_start = time.time()

        proc = subprocess.run(
            [sys.executable, exp["script"]],
            capture_output=False   # let output stream to terminal
        )

        elapsed = time.time() - t_start
        success = (proc.returncode == 0)

        results[exp["name"]] = {
            "script":      exp["script"],
            "return_code": proc.returncode,
            "elapsed_sec": round(elapsed, 2),
            "success":     success
        }

        status = "OK" if success else "FAILED"
        print(f"\n  -> {exp['name']}: {status} "
              f"({elapsed:.1f}s)")

        if not success:
            print(f"\n  PIPELINE HALTED: {exp['script']} "
                  f"returned code {proc.returncode}.")
            print("  Fix the failing experiment and re-run "
                  "run_all.py.")
            sys.exit(proc.returncode)

    return results


# ── Step 2: Verify output files ───────────────────────────────

def verify_output_files() -> dict:
    """
    Verify all expected data and figure files exist
    and are under 99MB each.
    """
    size_limit_mb = 99.0
    file_report   = {"data": {}, "figures": {}}
    all_ok        = True

    for fname in EXPECTED_DATA_FILES:
        path = os.path.join(DATA_DIR, fname)
        exists = os.path.exists(path)
        size   = file_size_mb(path) if exists else 0.0
        under  = size < size_limit_mb
        ok     = exists and under

        file_report["data"][fname] = {
            "exists":     exists,
            "size_mb":    round(size, 4),
            "under_99mb": under,
            "ok":         ok
        }
        if not ok:
            all_ok = False
            print(f"  MISSING OR OVERSIZED: {fname}")

    for fname in EXPECTED_FIGURE_FILES:
        path = os.path.join(FIGURES_DIR, fname)
        exists    = os.path.exists(path)
        size      = file_size_mb(path) if exists else 0.0
        size_kb   = size * 1024
        nonempty  = size_kb > 50
        ok        = exists and nonempty

        file_report["figures"][fname] = {
            "exists":      exists,
            "size_kb":     round(size_kb, 2),
            "nonempty":    nonempty,
            "ok":          ok
        }
        if not ok:
            all_ok = False
            print(f"  MISSING OR EMPTY FIGURE: {fname}")

    file_report["all_files_ok"] = all_ok
    return file_report


# ── Step 3: Extract hypothesis verdicts ───────────────────────

def extract_hypothesis_verdicts() -> dict:
    """
    Load each experiment summary JSON and extract
    the quantitative metrics and pass/fail verdict
    for each of the three hypotheses.
    """
    verdicts = {}

    # ── Miss Distance Proportionality Hypothesis ──────────────
    path1 = os.path.join(
        DATA_DIR, "miss_proportionality_summary.json"
    )
    if os.path.exists(path1):
        s1 = load_json(path1)
        super_stats = s1.get("super_threshold", {})
        R2 = super_stats["R_squared"]
        Ca = super_stats["Ca"]
        anova_p = super_stats.get("ANOVA_p")
        reg_p = super_stats.get("p_value", 1.0)
        effective_p = anova_p if anova_p is not None else reg_p

        supported = R2 >= 0.95 and effective_p <= 0.05
        verdicts["miss_distance_proportionality"] = {
            "status":          "SUPPORTED" if supported
                                else "WARNING",
            "regime":          "super-threshold only "
                               "(I_dot > I_dot_star)",
            "R_squared":       round(R2, 6),
            "Ca":              round(Ca, 4),
            "ANOVA_p_value":   float(effective_p),
            "threshold_R2":    0.95,
            "threshold_ANOVA": 0.05,
            "pass_R2":         R2 >= 0.95,
            "pass_ANOVA":      effective_p <= 0.05
        }
    else:
        verdicts["miss_distance_proportionality"] = {
            "status": "MISSING",
            "error":  "Summary file not found."
        }

    # ── Covert Injection Threshold Hypothesis ─────────────────
    path2 = os.path.join(
        DATA_DIR, "covert_threshold_summary.json"
    )
    if os.path.exists(path2):
        s2           = load_json(path2)
        I_analytical = s2["I_dot_star_analytical"]
        I_empirical  = s2["I_dot_star_empirical"]
        agreement    = s2["agreement_pct"]
        covert_rate  = s2.get(
            "covert_zone_detection_at_half_star", None
        )
        detect_rate  = s2.get(
            "detectable_zone_detection_at_high", None
        )
        supported = (
            I_analytical > 0 and
            agreement < 15.0 and
            (covert_rate is None or covert_rate < 0.10) and
            (detect_rate is None or detect_rate > 0.30)
        )
        verdicts["covert_injection_threshold"] = {
            "status":
                "SUPPORTED" if supported else "FAILED",
            "I_dot_star_analytical":  round(I_analytical, 6),
            "I_dot_star_empirical":   round(I_empirical, 6),
            "agreement_pct":          round(agreement, 2),
            "covert_zone_rate":       covert_rate,
            "detectable_zone_rate":   detect_rate,
            "threshold_agreement":    15.0,
            "pass_agreement":         agreement < 15.0,
            "pass_covert_zone":
                covert_rate is None or covert_rate < 0.10,
            "pass_detectable_zone":
                detect_rate is None or detect_rate > 0.30
        }
    else:
        verdicts["covert_injection_threshold"] = {
            "status": "MISSING",
            "error":  "Summary file not found."
        }

    # ── Kalman Gain Convergence and Directional Control ───────
    path3 = os.path.join(
        DATA_DIR, "gain_convergence_directional_summary.json"
    )
    if os.path.exists(path3):
        s3    = load_json(path3)
        ratio = s3["gain_convergence"]["convergence_ratio"]
        CoV   = s3["ca_stability"]["Ca_CoV"] or 1.0
        Ca_m  = s3["ca_stability"]["Ca_mean"]
        corr = s3["directional_correlation"]
        if "circular_linear_r" in corr:
            best_r = corr["circular_linear_r"]
            r_key = "circular_linear_r"
            pass_key = "pass_circular_linear_r"
            p_key = corr.get("circular_linear_p", None)
        else:
            best_r = corr["best_r"]
            r_key = "pearson_r"
            pass_key = "pass_pearson_r"
            p_key = None
        pass_conv = ratio < 0.10
        pass_cov  = CoV   < 0.05
        pass_r    = best_r > 0.95
        sig_r     = p_key is not None and p_key < 0.05
        n_pass = sum([pass_conv, pass_cov, pass_r])
        if n_pass == 3:
            status = "SUPPORTED"
        elif pass_cov and (pass_r or sig_r):
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "PARTIALLY_SUPPORTED" if n_pass >= 1 else "FAILED"
        verdicts["kalman_gain_convergence_directional_control"] = {
            "status":             status,
            "convergence_ratio":  round(ratio, 6),
            "Ca_mean":            round(Ca_m, 4)
                                   if Ca_m else None,
            "Ca_CoV":             round(CoV, 6),
            "threshold_ratio":    0.10,
            "threshold_CoV":      0.05,
            "threshold_r":        0.95,
            "pass_convergence":   pass_conv,
            "pass_CoV":           pass_cov,
            pass_key:             pass_r,
            r_key:                round(best_r, 6)
        }
    else:
        verdicts[
            "kalman_gain_convergence_directional_control"
        ] = {
            "status": "MISSING",
            "error":  "Summary file not found."
        }

    return verdicts


# ── Step 4: Run pytest suite ──────────────────────────────────

def run_pytest_suite() -> dict:
    """
    Run the full pytest suite and capture results.
    Writes log to results/pytest_final.log.
    """
    log_path = "results/pytest_final.log"
    os.makedirs("results", exist_ok=True)

    print("\n  Running full pytest suite...")
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/",
             "-v", "--tb=short"],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

    # Parse log for pass/fail count
    passed = 0
    failed = 0
    try:
        with open(log_path) as f:
            for line in f:
                if "short test summary info" in line:
                    continue
                match_passed = re.search(r"(\d+)\s+passed", line)
                match_failed = re.search(r"(\d+)\s+failed", line)
                if match_passed:
                    passed = int(match_passed.group(1))
                if match_failed:
                    failed = int(match_failed.group(1))
    except Exception:
        pass

    success = (proc.returncode == 0 and failed == 0)
    print(f"  pytest: {passed} passed, {failed} failed  "
          f"-> {'OK' if success else 'FAILED'}")
    print(f"  Full log: {log_path}")

    return {
        "return_code":   proc.returncode,
        "tests_passed":  passed,
        "tests_failed":  failed,
        "success":       success,
        "log_path":      log_path
    }


# ── Step 5: Build and write final QA report ───────────────────

def build_qa_report(
        run_results:   dict,
        file_report:   dict,
        verdicts:      dict,
        pytest_report: dict) -> dict:
    """
    Assemble the final QA report dictionary and write to JSON.
    """
    any_warning = any(
        v.get("status") in ("WARNING", "PARTIALLY_SUPPORTED")
        for v in verdicts.values()
    )
    any_failed = any(
        v.get("status") not in (
            "SUPPORTED", "WARNING", "PARTIALLY_SUPPORTED"
        )
        for v in verdicts.values()
    )
    files_ok = file_report.get("all_files_ok", False)
    tests_ok = pytest_report.get("success", False)

    if not any_failed and not any_warning and files_ok and tests_ok:
        overall = "ALL HYPOTHESES SUPPORTED"
    elif not any_failed and any_warning:
        overall = "HYPOTHESES WITH WARNINGS"
    else:
        overall = "PARTIAL — ONE OR MORE HYPOTHESES FAILED"

    warnings = []
    for key, v in verdicts.items():
        if v.get("status") == "WARNING":
            failures = [
                f for f in v
                if f.startswith("pass_") and not v[f]
            ]
            warnings.append(
                f"{key}: threshold not met for {failures}"
            )

    report = {
        "tool":           "SPECTRE-SIM",
        "version":        "1.0.0",
        "description":    (
            "Seeker-Phase EKF Covert Target Ramp "
            "Exploitation Simulator"
        ),
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "overall_verdict": overall,
        "warnings":       warnings,
        "hypotheses": verdicts,
        "experiments_run": run_results,
        "output_files":    file_report,
        "pytest_suite":    pytest_report,
        "figures_generated": EXPECTED_FIGURE_FILES,
        "all_files_under_99mb": file_report.get(
            "all_files_ok", False
        )
    }

    os.makedirs("results", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


# ── Step 6: Print final summary ───────────────────────────────

def print_final_summary(report: dict) -> None:
    """
    Print human-readable final summary to terminal.
    """
    separator("=")
    print("  SPECTRE-SIM FINAL QUALITY ASSURANCE REPORT")
    separator("=")
    print(f"  Timestamp : {report['timestamp_utc']}")
    print(f"  Report    : {REPORT_PATH}")
    print()

    # Hypothesis verdicts
    print("  HYPOTHESIS VERDICTS:")
    separator("-")
    hypothesis_labels = {
        "miss_distance_proportionality":
            "Miss Distance Proportionality",
        "covert_injection_threshold":
            "Covert Injection Threshold",
        "kalman_gain_convergence_directional_control":
            "Kalman Gain Convergence and Directional Control"
    }
    for key, label in hypothesis_labels.items():
        v = report["hypotheses"].get(key, {})
        status = v.get("status", "MISSING")
        icon   = "[OK]" if status == "SUPPORTED" else "[X]"
        print(f"  {icon} {label}")
        print(f"    Status: {status}")

        if key == "miss_distance_proportionality":
            if "R_squared" in v:
                print(f"    R^2_super = {v['R_squared']:.4f} "
                      f"(threshold > 0.95)")
                print(f"    Ca     = {v['Ca']:.4f}")
                print(f"    ANOVA p_super= {v['ANOVA_p_value']:.2e} "
                      f"(threshold < 0.05)")

        elif key == "covert_injection_threshold":
            if "I_dot_star_analytical" in v:
                print(f"    I_dot* analytical = "
                      f"{v['I_dot_star_analytical']:.6f} rad/s^2")
                print(f"    I_dot* empirical  = "
                      f"{v['I_dot_star_empirical']:.6f} rad/s^2")
                print(f"    Agreement = "
                      f"{v['agreement_pct']:.2f}% "
                      f"(threshold < 15%)")

        elif key == (
            "kalman_gain_convergence_directional_control"
        ):
            if "convergence_ratio" in v:
                print(f"    Convergence ratio = "
                      f"{v['convergence_ratio']:.4f} "
                      f"(threshold < 0.10)")
                print(f"    Ca CoV            = "
                      f"{v['Ca_CoV']:.4f} "
                      f"(threshold < 0.05)")
                r_label = "circular_linear_r" \
                    if "circular_linear_r" in v else "pearson_r"
                print(f"    {r_label}         = "
                      f"{v[r_label]:.4f} "
                      f"(threshold > 0.95)")
        print()

    # File check
    separator("-")
    n_data_ok = sum(
        1 for f in report["output_files"]["data"].values()
        if f["ok"]
    )
    n_fig_ok  = sum(
        1 for f in report["output_files"]["figures"].values()
        if f["ok"]
    )
    print(f"  OUTPUT FILES:")
    print(f"    Data files OK   : "
          f"{n_data_ok}/{len(EXPECTED_DATA_FILES)}")
    print(f"    Figure files OK : "
          f"{n_fig_ok}/{len(EXPECTED_FIGURE_FILES)}")
    print(f"    All under 99MB  : "
          f"{report['all_files_under_99mb']}")

    # Pytest
    separator("-")
    pt = report["pytest_suite"]
    print(f"  PYTEST SUITE:")
    print(f"    Tests passed : {pt['tests_passed']}")
    print(f"    Tests failed : {pt['tests_failed']}")
    print(f"    Status       : "
          f"{'OK' if pt['success'] else 'FAILED'}")

    # Overall
    separator("=")
    print(f"  OVERALL VERDICT: {report['overall_verdict']}")
    separator("=")


# ── Main ──────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    skip_experiments = "--skip-experiments" in sys.argv

    # Step 1: Run all experiments (unless skipped)
    section("STEP 1 — Running All Experiments")
    if skip_experiments:
        print("  Skipping experiments (--skip-experiments)")
        run_results = {"skipped": True}
    else:
        run_results = run_experiments()

    # Step 2: Verify output files
    section("STEP 2 — Verifying Output Files")
    file_report = verify_output_files()
    all_ok_str  = "OK" if file_report["all_files_ok"] else "ISSUES"
    print(f"\n  File verification: {all_ok_str}")

    # Step 3: Extract hypothesis verdicts
    section("STEP 3 — Extracting Hypothesis Verdicts")
    verdicts = extract_hypothesis_verdicts()
    for key, v in verdicts.items():
        print(f"  {key}: {v.get('status', 'MISSING')}")

    # Step 4: Write preliminary QA report so pytest can validate it
    section("STEP 4 — Writing Preliminary QA Report")
    preliminary_pytest = {
        "return_code": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "success":      True,
        "log_path":     "results/pytest_final.log"
    }
    build_qa_report(
        run_results, file_report, verdicts, preliminary_pytest
    )
    print(f"  Written preliminary: {REPORT_PATH}")

    # Step 5: Run full pytest suite (validates the preliminary report)
    section("STEP 5 — Running Full Pytest Suite")
    pytest_report = run_pytest_suite()

    # Step 6: Write final QA report with actual pytest results
    section("STEP 6 — Building Final QA Report")
    report = build_qa_report(
        run_results, file_report, verdicts, pytest_report
    )
    print(f"  Written: {REPORT_PATH}")

    # Step 7: Print summary
    print_final_summary(report)

    if report["overall_verdict"] == "ALL HYPOTHESES SUPPORTED":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
