SPECTRE-SIM
Seeker-Phase EKF Covert Target Ramp Exploitation Simulator

A 2D missile-target engagement simulation framework for
studying adversarial false target injection attacks on
Extended Kalman Filter (EKF)-based proportional navigation
missile guidance systems.

Quick Start
bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/SPECTRE-SIM
cd SPECTRE-SIM
pip install -r requirements.txt
python run_all.py
Expected final output:

text
OVERALL VERDICT: ALL HYPOTHESES SUPPORTED
What SPECTRE-SIM Does
SPECTRE-SIM simulates a covert adversarial attack on a
missile seeker. The attacker injects a slowly ramping false
angular bearing measurement into the EKF measurement channel
after acquisition lock. Because the ramp rate stays below the
chi-squared innovation gate threshold, the attack is
statistically undetectable while systematically steering the
missile away from the true target.

The simulator validates three research hypotheses:

Hypothesis	Claim	Key Metric
Miss Distance Proportionality	D_m = Ca * I_dot	R² > 0.95
Covert Injection Threshold	I_dot* derivable analytically	Agreement < 15%
Kalman Gain Convergence + Directional Control	Miss vector is steerable	Pearson r > 0.95
Repository Structure
text
SPECTRE-SIM/
├── config/
│   └── sim_config.yaml          # All simulation parameters
├── src/
│   ├── engagement/
│   │   └── geometry.py          # RK4 engagement physics
│   ├── estimator/
│   │   └── ekf_seeker.py        # EKF bearing-only estimator
│   ├── guidance/
│   │   └── pn_guidance.py       # Proportional navigation law
│   ├── attacker/
│   │   └── injection_attacker.py # Adversarial ramp injector
│   ├── monitor/
│   │   └── chi2_monitor.py      # Chi-squared bad-data detector
│   └── simulation_runner.py     # Unified pipeline integrator
├── experiments/
│   ├── run_miss_distance_proportionality.py
│   ├── run_covert_threshold.py
│   └── run_gain_convergence_directional.py
├── tests/
│   ├── test_geometry.py
│   ├── test_ekf.py
│   ├── test_guidance.py
│   ├── test_attacker.py
│   ├── test_monitor.py
│   ├── test_runner.py
│   ├── test_miss_distance_proportionality.py
│   ├── test_covert_threshold.py
│   ├── test_gain_convergence_directional.py
│   └── test_qa_report.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   └── CONFIGURATION.md
├── results/
│   ├── data/                    # Generated CSVs and JSONs
│   └── figures/                 # Generated publication figures
├── run_all.py                   # Master execution script
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md

Running Individual Experiments
bash
# Full pipeline (recommended)
python run_all.py

# Verify existing outputs without re-running
python run_all.py --verify

# Individual experiments
python experiments/run_miss_distance_proportionality.py
python experiments/run_covert_threshold.py
python experiments/run_gain_convergence_directional.py

# Full test suite
python -m pytest tests/ -v
Attack Model
The injection adds a linear ramp to the angular bearing
measurement after EKF lock:

text
delta_z(t) = I_dot * (t - t_lock) * cos(theta_inj)
Where I_dot is the injection rate [rad/s²] and
theta_inj is the attack direction angle.

The critical covert threshold rate I_dot* is derived
analytically from steady-state EKF parameters:

text
I_dot* = sqrt(chi2_threshold * S_inf) * bandwidth_ekf
Test Suite
65 automated tests across 10 test files:

bash
python -m pytest tests/ -v --tb=short
# Expected: 65 passed, 0 failed
Generated Figures
Figure	Description
fig1	Miss distance vs injection rate (linear fit + R²)
fig2	Detection rate vs injection rate (covert zone)
fig3	Covert threshold: analytical vs empirical boundary
fig4	Kalman gain convergence after acquisition lock
fig5	Polar miss vector directional control map
Requirements
Python 3.10, 3.11, or 3.12

numpy >= 1.24

scipy >= 1.10

pandas >= 2.0

matplotlib >= 3.7

tqdm >= 4.65

PyYAML >= 6.0

pytest >= 7.4

Citation
If you use SPECTRE-SIM in your research, please cite:

text
@software{spectre_sim_2026,
  title  = {SPECTRE-SIM: Seeker-Phase EKF Covert Target
             Ramp Exploitation Simulator},
  year   = {2026},
  url    = {https://github.com/[YOUR_GITHUB_USERNAME]/SPECTRE-SIM}
}
License
MIT License — see LICENSE file for details.
