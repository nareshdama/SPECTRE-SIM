# TAES desk rejection — rejection-to-action matrix

**Manuscript:** TAES-2026-1080 (prescreen rejection: insufficient novelty vs. mature stealthy FDI literature).

This document maps each editor concern to **evidence**, **code or experiment deliverables**, and **manuscript sections** to change.

| Editor concern | Required response | Evidence / deliverable | Manuscript location |
|----------------|-------------------|-------------------------|---------------------|
| “Mature area; no new attack mechanisms” | Introduce a **non-trivial attack class** beyond gradual ramp bias | `OptimizedStealthAttacker`: per-step constrained optimization on innovation Mahalanobis norm; pilot MC in `experiments/run_optimized_attack_pilot.py` | New § method subsection “Optimization-constrained stealthy injection”; § results comparing ramp vs. optimized |
| “No new analytical frameworks” | Add **explicit optimization formulation** + (Track B) detectability–impact trade-off discussion | Problem statement: maximize alignment with miss-vector objective direction subject to \(\mathbf{r}^T \mathbf{S}^{-1}\mathbf{r} \leq \tau\); optional bound discussion in discussion | § mathematical formulation; § discussion |
| “Ramp + chi-squared is well known” | **Scope claims**: position ramp study as baseline; cite prior art explicitly | Related-work bullets + “Novelty vs. prior art” subsection in LaTeX | § introduction / related work |
| “Randomized control trivial in literature” | De-emphasize as sole countermeasure or pair with **stronger baselines** | CUSUM on innovation stream in `simulation_runner` results; adaptive threshold optional in config | § countermeasures; limitations |
| “Application to PN missile straightforward” | **Generalization experiments** | `experiments/run_geometry_sweep.py`: head-on / crossing / tail-chase configs, MC summaries | § experimental setup; § results |
| “Technical report / application note” | **Track A**: reframe as application simulation study + reproducibility; **Track B**: theory + benchmarks | `docs/SUBMISSION_DUAL_TRACK.md`; broadened sweeps (noise, geometry) | Abstract, introduction, contributions list |

## Claim discipline (high level)

- **Do not** claim “first open-literature study” without a verified survey; replace with scoped gaps (e.g., EKF–PN seeker loop under 2D directional injection + benchmark suite).
- **Do** separate: (1) baseline ramp covert FDI, (2) optimized stealth injection, (3) detector baselines (chi-squared, CUSUM).

## Completion checklist

- [x] Matrix documented (this file)
- [x] Optimized attacker + EKF pre-update residual API
- [x] Simulation integration + optional CUSUM rate in results
- [x] Pilot + geometry sweep experiment scripts
- [x] LaTeX: novelty subsection + softened abstract claims
- [x] Dual-track submission guide
