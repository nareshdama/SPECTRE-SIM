# SPECTRE-SIM

**Seeker-Phase EKF Covert Target Ramp Exploitation Simulator**

SPECTRE-SIM is a 2D missile-target engagement simulation framework for studying adversarial false target injection attacks on Extended Kalman Filter (EKF)-based proportional navigation guidance systems.

---

## Quick Start

```bash
git clone https://github.com/nareshdama/SPECTRE-SIM.git
cd SPECTRE-SIM
pip install -r requirements.txt
python run_all.py
```

Expected final output:

```text
OVERALL VERDICT: ALL HYPOTHESES SUPPORTED
```

---

## What SPECTRE-SIM Does

SPECTRE-SIM models a covert adversarial attack on a missile seeker.  
The attacker injects a slowly ramping false measurement into the **two-channel (bearing and range)** EKF measurement model **after acquisition lock**.

Because the ramp rate can stay below the chi-squared innovation gate threshold, the attack can remain statistically undetectable while systematically steering the missile away from the true target. Set `attacker.mode: optimized` in `config/sim_config.yaml` for an innovation-bounded, optimization-based injection law (see `docs/EXPERIMENTS.md`).

### Validated Research Hypotheses

| Hypothesis | Claim | Key Metric |
|---|---|---|
| Miss Distance Proportionality | \(D_m = C_a \cdot \dot{I}\) | \(R^2 > 0.95\) |
| Covert Injection Threshold | \(\dot{I}^*\) is analytically derivable | Agreement < 15% |
| Kalman Gain Convergence + Directional Control | Miss vector is steerable | Pearson \(r > 0.95\) |

---

## Repository Structure

```text
SPECTRE-SIM/
├── config/
│   └── sim_config.yaml
├── src/
│   ├── engagement/geometry.py
│   ├── estimator/ekf_seeker.py
│   ├── guidance/pn_guidance.py
│   ├── attacker/injection_attacker.py
│   ├── attacker/optimized_stealth_attacker.py
│   ├── monitor/chi2_monitor.py
│   ├── monitor/cusum_detection.py
│   └── simulation_runner.py
├── experiments/
│   ├── run_miss_distance_proportionality.py
│   ├── run_covert_threshold.py
│   ├── run_gain_convergence_directional.py
│   ├── run_sensitivity_analysis.py
│   └── run_attack_comparison.py
│   ├── run_optimized_attack_pilot.py
│   └── run_geometry_sweep.py
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── CONFIGURATION.md
│   ├── REJECTION_ACTION_MATRIX.md
│   └── SUBMISSION_DUAL_TRACK.md
├── results/
│   ├── data/
│   └── figures/
├── run_all.py
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Running Experiments

```bash
# Full pipeline (recommended)
python run_all.py

# Verify existing outputs without re-running experiments
python run_all.py --skip-experiments

# Individual experiments
python experiments/run_miss_distance_proportionality.py
python experiments/run_covert_threshold.py
python experiments/run_gain_convergence_directional.py
python experiments/run_sensitivity_analysis.py
python experiments/run_attack_comparison.py
python experiments/run_optimized_attack_pilot.py
python experiments/run_geometry_sweep.py
```

---

## Attack Model

The two-channel measurement injection is:

```text
z_inj(t) = z_true(t) + I_dot * (t - t_lock) * [cos(theta_inj); gamma * sin(theta_inj)]
```

where:
- `I_dot` is the injection rate [rad/s²]
- `theta_inj` is the attack direction angle (bearing component ∝ cos, range component ∝ sin)
- `gamma` scales angular ramp to range-equivalent units

The critical covert threshold rate is derived from steady-state EKF quantities:

```text
I_dot* = sqrt(chi2_threshold * S_inf) * bandwidth_ekf
```

---

## Tests and Reproducibility

Run the full test suite:

```bash
python -m pytest tests/ -v --tb=short
```

Current baseline:
- **81 passed, 0 failed**

---

## Generated Figures

| Figure | Description |
|---|---|
| fig1 | Miss distance vs injection rate (linear fit + \(R^2\)) |
| fig2 | Detection rate vs injection rate (covert zone) |
| fig3 | Covert threshold: analytical vs empirical boundary |
| fig4 | Kalman gain convergence (CV-based) after acquisition lock |
| fig5 | Directional miss vector control (cos θ vs cross-track miss) |
| fig6 | Sensitivity of \(C_a\) to engagement parameters |
| fig7 | Attack waveform and detector comparison (ramp/step/sine, chi² vs CUSUM) |

---

## Requirements

- Python 3.10, 3.11, or 3.12
- `numpy>=1.24`
- `scipy>=1.10`
- `pandas>=2.0`
- `matplotlib>=3.7`
- `tqdm>=4.65`
- `PyYAML>=6.0`
- `pytest>=7.4`

Install with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use SPECTRE-SIM in your research, please cite:

```bibtex
@software{spectre_sim_2026,
  title  = {SPECTRE-SIM: Seeker-Phase EKF Covert Target Ramp Exploitation Simulator},
  year   = {2026},
  url    = {https://github.com/nareshdama/SPECTRE-SIM}
}
```

For structured metadata, see `CITATION.cff`.

---

## License

MIT License. See `LICENSE` for details.
