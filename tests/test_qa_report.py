import pytest
import json
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

REPORT_PATH = "results/SPECTRE_SIM_QA_REPORT.json"


@pytest.fixture
def report():
    """Load QA report. Fails if run_all.py not executed."""
    assert os.path.exists(REPORT_PATH), (
        f"QA report not found: {REPORT_PATH}\n"
        f"Run: python run_all.py"
    )
    with open(REPORT_PATH) as f:
        return json.load(f)


def test_overall_verdict_is_all_supported(report):
    """
    Overall verdict must be ALL HYPOTHESES SUPPORTED.
    """
    verdict = report["overall_verdict"]
    assert verdict == "ALL HYPOTHESES SUPPORTED", (
        f"Overall verdict is '{verdict}', "
        f"expected 'ALL HYPOTHESES SUPPORTED'."
    )


def test_all_three_hypotheses_supported(report):
    """
    All three hypotheses must be present and SUPPORTED.
    """
    hyp = report["hypotheses"]
    expected_keys = [
        "miss_distance_proportionality",
        "covert_injection_threshold",
        "kalman_gain_convergence_directional_control"
    ]
    for key in expected_keys:
        assert key in hyp, (
            f"Hypothesis '{key}' missing from report."
        )
        status = hyp[key].get("status")
        assert status == "SUPPORTED", (
            f"Hypothesis '{key}' status is '{status}', "
            f"expected 'SUPPORTED'."
        )


def test_warnings_present_when_any_hypothesis_warning(report):
    """
    If any hypothesis status is WARNING, report must include a
    non-empty warnings list so paper authors can address it.
    """
    hyp = report["hypotheses"]
    any_warning = any(v.get("status") == "WARNING" for v in hyp.values())
    if any_warning:
        warnings = report.get("warnings")
        assert isinstance(warnings, list) and len(warnings) > 0, (
            "Hypothesis WARNING present but top-level 'warnings' list is "
            "missing or empty."
        )


def test_report_has_required_top_level_keys(report):
    """
    QA report must contain all required top-level keys.
    """
    required = [
        "tool", "version", "timestamp_utc",
        "overall_verdict", "hypotheses",
        "experiments_run", "output_files",
        "pytest_suite", "figures_generated",
        "all_files_under_99mb"
    ]
    for key in required:
        assert key in report, (
            f"Missing top-level key '{key}' in QA report."
        )


def test_all_figures_listed(report):
    """
    QA report must list all expected figure filenames.
    """
    figs = report["figures_generated"]
    expected = [
        "fig1_miss_distance_vs_injection_rate.png",
        "fig2_detection_rate_vs_injection_rate.png",
        "fig3_covert_injection_threshold.png",
        "fig4_kalman_gain_convergence.png",
        "fig5_directional_miss_vector_control.png",
        "fig6_sensitivity_analysis.png",
        "fig7_attack_comparison.png",
    ]
    for fname in expected:
        assert fname in figs, (
            f"Figure '{fname}' not listed in QA report."
        )


def test_all_files_under_99mb(report):
    """
    QA report must confirm all output files are under 99MB.
    """
    assert report["all_files_under_99mb"] is True, (
        "One or more output files exceed 99MB. "
        "Check results/data/ for oversized CSVs."
    )


def test_pytest_suite_passed(report):
    """
    Pytest suite must report zero failed tests.
    """
    failed = report["pytest_suite"]["tests_failed"]
    assert failed == 0, (
        f"Pytest suite has {failed} failing tests. "
        f"Check results/pytest_final.log for details."
    )


def test_qa_evaluator_reads_super_threshold_r2(report):
    """
    The QA report's H1 R² must match the super_threshold R² from
    the experiment summary, not a globally fitted value.
    """
    summary_path = "results/data/miss_proportionality_summary.json"
    assert os.path.exists(summary_path), (
        f"Summary not found: {summary_path}"
    )
    with open(summary_path) as f:
        s = json.load(f)
    super_r2 = s["super_threshold"]["R_squared"]
    qa_r2 = report["hypotheses"][
        "miss_distance_proportionality"
    ]["R_squared"]
    assert abs(qa_r2 - round(super_r2, 6)) < 1e-5, (
        f"QA report R² ({qa_r2}) does not match "
        f"super_threshold R² ({super_r2:.6f}). "
        f"Evaluator may be reading the global regression."
    )


def test_report_is_valid_json():
    """
    QA report file must be parseable as valid JSON.
    No NaN, Infinity, or undefined values allowed.
    """
    assert os.path.exists(REPORT_PATH), (
        f"QA report not found: {REPORT_PATH}"
    )
    with open(REPORT_PATH) as f:
        content = f.read()
    try:
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"QA report is not valid JSON: {e}"
        )
