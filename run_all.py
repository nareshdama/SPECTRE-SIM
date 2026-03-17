"""
SPECTRE-SIM Master Execution Script
====================================
Runs all three experiments in sequence and produces a final
Quality Assurance Report confirming reproducibility.

Usage:
    python run_all.py              # full pipeline
    python run_all.py --verify     # verify outputs only (no rerun)

Output:
    results/SPECTRE_SIM_QA_REPORT.json
"""

import sys
import os
import json
import time
import argparse
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
]

EXPECTED_FIGURE_FILES = [
    "fig1_miss_distance_vs_injection_rate.png",
    "fig2_detection_rate_vs_injection_rate.png",
    "fig3_covert_injection_threshold.png",
    "fig4_kalman_gain_convergence.png",
    "fig5_directional_miss_vector_control.png",
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
        R2     = s1["regression"]["R_squared"]
        Ca     = s1["regression"]["Ca"]
        anova_p = s1["anova"]["p_value"]
        supported = R2 > 0.95 and anova_p < 0.05
        verdicts["miss_distance_proportionality"] = {
            "status":          "SUPPORTED" if supported
                                else "FAILED",
            "R_squared":       round(R2, 6),
            "Ca":              round(Ca, 4),
            "ANOVA_p_value":   float(anova_p),
            "threshold_R2":    0.95,
            "threshold_ANOVA": 0.05,
            "pass_R2":         R2 > 0.95,
            "pass_ANOVA":      anova_p < 0.05
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
            "detectable_zone_detection_at_2x_star", None
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
        best_r = s3["directional_correlation"]["best_r"]
        supported = (
            ratio  < 0.10 and
            CoV    < 0.05 and
            best_r > 0.95
        )
        verdicts["kalman_gain_convergence_directional_control"] = {
            "status":
                "SUPPORTED" if supported else "FAILED",
            "convergence_ratio":  round(ratio, 6),
            "Ca_mean":            round(Ca_m, 4)
                                   if Ca_m else None,
            "Ca_CoV":             round(CoV, 6),
            "pearson_r":          round(best_r, 6),
            "threshold_ratio":    0.10,
            "threshold_CoV":      0.05,
            "threshold_r":        0.95,
            "pass_convergence":   ratio  < 0.10,
            "pass_CoV":           CoV    < 0.05,
            "pass_pearson_r":     best_r > 0.95
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
             "-v", "--tb=short", "-k", "not test_qa_report"],
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
    all_supported = all(
        v.get("status") == "SUPPORTED"
        for v in verdicts.values()
    )
    files_ok = file_report.get("all_files_ok", False)
    tests_ok = pytest_report.get("success", False)

    if all_supported and files_ok and tests_ok:
        overall = "ALL HYPOTHESES SUPPORTED"
    elif all_supported and not tests_ok:
        overall = "HYPOTHESES SUPPORTED — PYTEST FAILURES PRESENT"
    elif not all_supported:
        overall = "PARTIAL — ONE OR MORE HYPOTHESES FAILED"
    else:
        overall = "FAILED"

    report = {
        "tool":           "SPECTRE-SIM",
        "version":        "1.0.0",
        "description":    (
            "Seeker-Phase EKF Covert Target Ramp "
            "Exploitation Simulator"
        ),
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "overall_verdict": overall,
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
                print(f"    R^2     = {v['R_squared']:.4f} "
                      f"(threshold > 0.95)")
                print(f"    Ca     = {v['Ca']:.4f}")
                print(f"    ANOVA p= {v['ANOVA_p_value']:.2e} "
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
                print(f"    Pearson r         = "
                      f"{v['pearson_r']:.4f} "
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
    parser = argparse.ArgumentParser(
        description="SPECTRE-SIM Master Run Script"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing outputs only — skip re-running "
             "experiments"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: Run or skip experiments
    if args.verify:
        section("VERIFY MODE — Skipping experiment execution")
        run_results = {
            exp["name"]: {"skipped": True}
            for exp in EXPERIMENTS
        }
    else:
        section("STEP 1 — Running All Experiments")
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

    # Step 4: Run pytest suite
    section("STEP 4 — Running Full Pytest Suite")
    pytest_report = run_pytest_suite()

    # Step 5: Build and write QA report
    section("STEP 5 — Building Final QA Report")
    report = build_qa_report(
        run_results, file_report, verdicts, pytest_report
    )
    print(f"  Written: {REPORT_PATH}")

    # Step 6: Print summary
    print_final_summary(report)

    # Exit code: 0 if all supported, 1 otherwise
    if report["overall_verdict"] == "ALL HYPOTHESES SUPPORTED":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
