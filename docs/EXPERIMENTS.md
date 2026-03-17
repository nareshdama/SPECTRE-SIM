# SPECTRE-SIM Experiments

Three experiments validate the three research hypotheses.
All experiments are run automatically by `python run_all.py`.

---

## Experiment 1: Miss Distance Proportionality

**Script:** `experiments/run_miss_distance_proportionality.py`

**Hypothesis:** Miss distance scales linearly with injection rate
governed by a stabilized proportionality coefficient Ca:

    D_m = Ca * I_dot + epsilon

**Method:**
- 9 injection rates: [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
- 30 Monte Carlo trials per injection rate (varied random seed)
- Linear regression fit using scipy.stats.linregress
- One-way ANOVA across injection rate groups

**Acceptance Criteria:**
- R-squared > 0.95
- ANOVA p-value < 0.05

**Outputs:**
- results/data/miss_proportionality_sweep.csv
- results/data/miss_proportionality_summary.json
- results/figures/fig1_miss_distance_vs_injection_rate.png
- results/figures/fig2_detection_rate_vs_injection_rate.png

---

## Experiment 2: Covert Injection Threshold

**Script:** `experiments/run_covert_threshold.py`

**Hypothesis:** A critical injection rate I_dot* exists below
which the attack remains statistically undetectable by the
chi-squared innovation gate monitor.

**Method:**
- Analytical derivation of I_dot* from steady-state EKF
  covariance, Kalman gain, and chi-squared gate geometry
- 100-point fine numerical sweep from 0 to 2 * I_dot*
- 50 Monte Carlo trials per injection rate
- Empirical threshold identified by linear interpolation

**Acceptance Criteria:**
- I_dot* is positive and finite
- Analytical vs empirical agreement within 15%
- Detection rate at 0.5 * I_dot* below 0.10
- Detection rate at 2.0 * I_dot* above 0.30

**Outputs:**
- results/data/covert_threshold_sweep.csv
- results/data/covert_threshold_summary.json
- results/figures/fig3_covert_injection_threshold.png

---

## Experiment 3: Kalman Gain Convergence and Directional Control

**Script:** `experiments/run_gain_convergence_directional.py`

**Hypothesis:** After EKF acquisition lock, the Kalman gain
converges to a stable steady-state value making Ca fixed and
predictable, and the miss vector direction is independently
controllable by varying the injection angle.

**Method:**
- 100 clean Monte Carlo runs for gain convergence measurement
- Convergence ratio: std_late / std_early of Kalman gain norm
- 8 injection angles: [0, 45, 90, 135, 180, 225, 270, 315] degrees
- 100 Monte Carlo trials per angle
- Pearson correlation between injection angle and miss vector angle
- Ca coefficient of variation across all angle groups

**Acceptance Criteria:**
- Convergence ratio sigma_late / sigma_early < 0.10
- Ca coefficient of variation < 0.05
- Pearson r (injection angle vs miss angle) > 0.95

**Outputs:**
- results/data/gain_convergence_raw.csv
- results/data/gain_convergence_grouped.csv
- results/data/directional_control.csv
- results/data/gain_convergence_directional_summary.json
- results/figures/fig4_kalman_gain_convergence.png
- results/figures/fig5_directional_miss_vector_control.png

---

## Running Individual Experiments

```bash
# Run all experiments + QA report (recommended)
python run_all.py

# Run one experiment at a time
python experiments/run_miss_distance_proportionality.py
python experiments/run_covert_threshold.py
python experiments/run_gain_convergence_directional.py

# Verify existing outputs without re-running
python run_all.py --verify
