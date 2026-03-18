# SPECTRE-SIM Experiments

Five experiments support the three research hypotheses and supplementary analysis.
The three core experiments (1–3) are run automatically by `python run_all.py`.

---

## Experiment 1: Miss Distance Proportionality

**Script:** `experiments/run_miss_distance_proportionality.py`

**Hypothesis:** Miss distance scales linearly with injection rate
governed by a stabilized proportionality coefficient \(C_a\):

    D_m = C_a * I_dot + epsilon

**Method:**
- 12 injection rates: [0.0, 0.01, 0.02, …, 0.10] rad/s²
- 100 Monte Carlo trials per injection rate
- Linear regression on super-effective regime (I_dot ≥ 0.05 rad/s²)

**Acceptance Criteria:**
- R² > 0.95
- Regression p-value < 0.05

**Outputs:**
- results/data/miss_proportionality_sweep.csv
- results/data/miss_proportionality_summary.json
- results/figures/fig1_miss_distance_vs_injection_rate.png
- results/figures/fig2_detection_rate_vs_injection_rate.png

---

## Experiment 2: Covert Injection Threshold

**Script:** `experiments/run_covert_threshold.py`

**Hypothesis:** A critical injection rate \(\dot{I}^*\) exists below
which the attack remains statistically undetectable by the
chi-squared innovation gate monitor.

**Method:**
- Analytical derivation of \(\dot{I}^*\) from steady-state EKF
  covariance, Kalman gain, and chi-squared gate geometry
- 100-point fine numerical sweep from 0 to 2×\(\dot{I}^*\)
- 50 Monte Carlo trials per injection rate
- Empirical threshold via linear interpolation at 10% trigger level

**Acceptance Criteria:**
- \(\dot{I}^*\) is positive and finite
- Analytical vs empirical agreement within 15%
- Detection rate at 0.5×\(\dot{I}^*\) below 0.10
- Detection rate at 2.0×\(\dot{I}^*\) above 0.30

**Outputs:**
- results/data/covert_threshold_sweep.csv
- results/data/covert_threshold_summary.json
- results/figures/fig3_covert_injection_threshold.png

---

## Experiment 3: Kalman Gain Convergence and Directional Control

**Script:** `experiments/run_gain_convergence_directional.py`

**Hypothesis:** After EKF acquisition lock, the Kalman gain
is essentially deterministic (low coefficient of variation across
Monte Carlo seeds), making \(C_a\) stable, and the miss vector
direction is controllable via the bearing injection component.

**Method:**
- **Gain convergence:** 100 clean Monte Carlo runs; compute
  mean coefficient of variation (CV = σ/μ) of gain norm across
  seeds over the mid-game window (15%–85% of engagement).
  \(\rho = \overline{\text{CV}}_{\text{midgame}}\).
- **Directional control:** 8 injection angles [0°, 45°, …, 315°]
  at \(\dot{I} = 0.10\) rad/s²; 100 trials per angle.
  Pearson correlation between \(\cos\theta_{\text{inj}}\) and
  mean signed cross-track miss displacement.
- \(C_a\) stability: CoV of \(C_a\) across bearing-effective angles
  (\(|\cos\theta_{\text{inj}}| > 0.3\)).

**Acceptance Criteria:**
- \(\rho < 0.10\) (gain convergence)
- \(C_a\) CoV < 0.05
- Pearson r (\(\cos\theta_{\text{inj}}\) vs \(\bar{y}_{\text{miss}}\)) > 0.95

**Outputs:**
- results/data/gain_convergence_raw.csv
- results/data/gain_convergence_grouped.csv
- results/data/directional_control.csv
- results/data/gain_convergence_directional_summary.json
- results/figures/fig4_kalman_gain_convergence.png
- results/figures/fig5_directional_miss_vector_control.png

---

## Experiment 4: Sensitivity Analysis

**Script:** `experiments/run_sensitivity_analysis.py`

**Purpose:** Quantify robustness of \(C_a\) under parametric variation.

**Method:**
- One-at-a-time sweeps: N ∈ {3, 4, 5}, V_c scale, Q scale, R scale
- 30 Monte Carlo trials per configuration at \(\dot{I} = 0.06\) rad/s²

**Outputs:**
- results/data/sensitivity_analysis.csv
- results/data/sensitivity_analysis_summary.json
- results/figures/fig6_sensitivity_analysis.png

---

## Experiment 5: Attack Waveform and Detector Comparison

**Script:** `experiments/run_attack_comparison.py`

**Purpose:** Compare ramp vs step vs sinusoidal injection waveforms and
chi-squared vs CUSUM detectors.

**Method:**
- Waveforms: ramp (baseline), step, sinusoidal at matched rates
- Detectors: chi-squared gate vs CUSUM at several injection rates

**Outputs:**
- results/data/attack_comparison_summary.json
- results/figures/fig7_attack_comparison.png

---

## Running Individual Experiments

```bash
# Run core experiments + QA report (recommended)
python run_all.py

# Skip experiments, verify existing outputs
python run_all.py --skip-experiments

# Individual experiments
python experiments/run_miss_distance_proportionality.py
python experiments/run_covert_threshold.py
python experiments/run_gain_convergence_directional.py
python experiments/run_sensitivity_analysis.py
python experiments/run_attack_comparison.py
```
