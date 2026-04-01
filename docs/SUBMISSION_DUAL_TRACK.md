# Dual-track submission strategy (post–TAES prescreen)

This note operationalizes **Track A** (fast, application-oriented resubmission) and **Track B** (TAES-grade extension) without editing the recovery plan file.

## Track A: Application-focused manuscript (about 4–6 weeks)

**Positioning**

- Frame the work as a **reproducible simulation benchmark** for EKF–PN seeker loops under measurement FDI, not as foundational stealthy-FDI theory.
- State explicitly what is standard (innovation-gate stealth, CUSUM baselines) versus what is **instantiated** here (two-channel seeker, lock-time ramp, directional miss control, SPECTRE-SIM QA harness).

**Evidence to cite from the repo**

- Core hypotheses: existing experiment summaries under `results/data/`.
- **Geometry generalization:** run `python experiments/run_geometry_sweep.py` → `results/data/geometry_sweep.csv`.
- **Detector breadth:** simulation summaries now include `detection_rate_cusum_tracking` (see `src/simulation_runner.py`).

**Venue style (examples only—not endorsements)**

- Journals or transactions that welcome **simulation studies**, **GNC application notes**, or **electronic-warfare modeling** with strong reproducibility.
- Adjust title/abstract to match “benchmark / case study / assessment” language.

## Track B: TAES-grade extension (about 8–12+ weeks)

**Minimum bar to revisit TAES-level novelty**

- One **non-myopic** or **analytically characterized** element, e.g. multi-step attack design, detectability bound under time-varying \( \mathbf{S}_k \), or a **principled** resilient estimator/guidance modification with ablations.
- Broader **threat model** (maneuvering target, model mismatch) tied to the optimized attacker or a successor.

**Code hooks already in place**

- `attacker.mode: optimized` and `OptimizedStealthAttacker` in `src/attacker/optimized_stealth_attacker.py`.
- Pilot comparison script: `experiments/run_optimized_attack_pilot.py` → `results/data/optimized_attack_pilot_summary.json`.

**Desk-rejection risk controls**

- Keep a visible **“Novelty vs. prior stealthy FDI”** subsection (see `Latex code.txt` and `docs/REJECTION_ACTION_MATRIX.md`).
- Avoid absolute uniqueness claims; use scoped, falsifiable statements.

## One-page resubmission checklist

1. Update abstract per Track A or B framing.
2. Run `python experiments/run_geometry_sweep.py` and `python experiments/run_optimized_attack_pilot.py`; archive CSV/JSON with the submission supplementary material.
3. Regenerate figures if geometry or attack parameters change.
4. Pass `python -m pytest tests/` before freezing a version tag.
