# SPECTRE-SIM Architecture

## Overview

SPECTRE-SIM is a 2D missile-target engagement simulator that
models adversarial false target injection attacks on Extended
Kalman Filter (EKF)-based proportional navigation missile
guidance systems.

## Module Dependency Graph

config/sim_config.yaml
│
▼
src/engagement/geometry.py ← RK4 engagement physics
src/estimator/ekf_seeker.py ← EKF bearing-only state estimation
src/guidance/pn_guidance.py ← Proportional navigation law
src/attacker/injection_attacker.py ← Adversarial ramp injection
src/monitor/chi2_monitor.py ← Chi-squared bad-data detector
│
▼
src/simulation_runner.py ← Unified pipeline integrator
│
├── experiments/run_miss_distance_proportionality.py
├── experiments/run_covert_threshold.py
└── experiments/run_gain_convergence_directional.py
│
▼
run_all.py ← Master execution + QA report

text

## Data Flow Per Timestep

geometry.step(accel_cmd)
→ true state [mx, my, tx, ty, Vc]
→ z_true = arctan2(ty-my, tx-mx) + gaussian_noise
attacker.compute_injection(t, z_true)
→ z_measured (injected or pass-through)
ekf.predict()
ekf.update(z_measured)
→ x_hat [r_x, r_y, v_rx, v_ry]
→ chi2_stat, innovation, K, P
guidance.compute_command(los_rate_hat, Vc)
→ a_cmd = N * Vc * los_rate_hat (clipped to a_max)
monitor.check(chi2_stat)
→ alarm (bool)

text

## State Vector

| Symbol | Description | Units |
|--------|-------------|-------|
| r_x    | Relative position x (target - missile) | m |
| r_y    | Relative position y (target - missile) | m |
| v_rx   | Relative velocity x | m/s |
| v_ry   | Relative velocity y | m/s |

## Attack Model

The adversarial injection adds a ramp offset to the angular
bearing measurement:

    delta_z(t) = I_dot * (t - t_start) * cos(theta_inj)

Where:
- I_dot         : injection rate [rad/s per second]
- t_start       : EKF acquisition lock time [s]
- theta_inj     : injection direction angle [rad]

The attack activates only after EKF acquisition lock to
exploit the stabilized steady-state Kalman gain.

## Output Files

| File | Description |
|------|-------------|
| results/data/miss_proportionality_sweep.csv | Miss distance vs injection rate sweep |
| results/data/covert_threshold_sweep.csv | Detection rate fine sweep |
| results/data/directional_control.csv | Miss vector per injection angle |
| results/data/gain_convergence_grouped.csv | Kalman gain over time |
| results/SPECTRE_SIM_QA_REPORT.json | Final machine-readable verdict |
| results/figures/fig1_*.png | Miss distance vs injection rate |
| results/figures/fig2_*.png | Detection rate vs injection rate |
| results/figures/fig3_*.png | Covert injection threshold |
| results/figures/fig4_*.png | Kalman gain convergence |
| results/figures/fig5_*.png | Directional miss vector control |
