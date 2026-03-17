SPECTRE-SIM Configuration Reference
All simulation parameters are controlled by:
config/sim_config.yaml

Do NOT modify this file directly for experiments.
Use SPECTRESimulation.from_config_override() instead.

missile
Key	Default	Units	Description
x0	0.0	m	Initial x position
y0	0.0	m	Initial y position
vx0	300.0	m/s	Initial x velocity
vy0	0.0	m/s	Initial y velocity
target
Key	Default	Units	Description
x0	10000.0	m	Initial x position
y0	500.0	m	Initial y position
vx0	-200.0	m/s	Initial x velocity
vy0	0.0	m/s	Initial y velocity
simulation
Key	Default	Units	Description
dt	0.01	s	Timestep size
t_max	60.0	s	Maximum engagement duration
ekf
Key	Default	Units	Description
Q_diag	[0.1,0.1,1.0,1.0]	mixed	Process noise diagonal
R_scalar	0.0001	rad²	Measurement noise variance
lock_threshold	50.0	mixed	P trace threshold for lock
t_acquisition	2.0	s	Min time before lock check
guidance
Key	Default	Units	Description
N	4	—	Navigation constant
a_max	300.0	m/s²	Maximum acceleration command
attacker
Key	Default	Units	Description
injection_rate	0.0	rad/s²	Ramp rate (0 = clean run)
injection_angle_deg	90.0	degrees	Attack direction angle
physical_accel_max	100.0	m/s²	Kinematic consistency limit
active	false	bool	Attack enable flag
monitor
Key	Default	Units	Description
alpha	0.05	—	False alarm significance level
n_z	1	—	Measurement dimension (scalar)
window_size	10	steps	Rolling detection window size
Programmatic Override Example
python
from src.simulation_runner import SPECTRESimulation

sim = SPECTRESimulation.from_config_override(
    "config/sim_config.yaml",
    {
        "attacker.injection_rate":      0.02,
        "attacker.injection_angle_deg": 45.0,
        "attacker.active":              True,
        "ekf.R_scalar":                 0.0005
    }
)
results = sim.run(seed=42)
print(results["miss_distance"])
