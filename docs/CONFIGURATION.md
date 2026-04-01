# SPECTRE-SIM Configuration Reference

All simulation parameters are controlled by:

    config/sim_config.yaml

**Do NOT modify this file directly for experiments.** Use
`SPECTRESimulation.from_config_override()` instead.

---

## missile

| Key  | Default | Units | Description                |
|------|---------|-------|----------------------------|
| x0   | 0.0     | m     | Initial x position         |
| y0   | 0.0     | m     | Initial y position         |
| vx0  | 300.0   | m/s   | Initial x velocity         |
| vy0  | 0.0     | m/s   | Initial y velocity         |

---

## target

| Key  | Default | Units | Description                |
|------|---------|-------|----------------------------|
| x0   | 10000.0 | m     | Initial x position         |
| y0   | 500.0   | m     | Initial y position         |
| vx0  | -200.0  | m/s   | Initial x velocity         |
| vy0  | 0.0     | m/s   | Initial y velocity         |

---

## simulation

| Key  | Default | Units | Description                |
|------|---------|-------|----------------------------|
| dt   | 0.01    | s     | Timestep size              |
| t_max| 60.0    | s     | Maximum engagement duration|
| N    | 4       | —     | Navigation constant       |

---

## ekf

| Key           | Default                        | Units | Description                    |
|---------------|--------------------------------|-------|--------------------------------|
| Q_diag        | [0.001, 0.001, 0.003, 0.003]   | mixed  | Process noise diagonal         |
| R_diag        | [0.000001, 25.0]               | mixed  | Measurement noise (bearing², range²) |
| lock_threshold| 50.0                           | mixed  | P trace threshold for lock     |
| t_acquisition | 2.0                            | s     | Min time before lock check     |

---

## guidance

| Key   | Default | Units | Description                |
|-------|---------|-------|----------------------------|
| N     | 4       | —     | Navigation constant        |
| a_max | 300.0   | m/s²  | Maximum acceleration command|

---

## attacker

| Key                  | Default | Units   | Description                      |
|----------------------|---------|---------|----------------------------------|
| mode                 | ramp    | —       | `ramp` or `optimized`            |
| injection_rate       | 0.0     | rad/s²  | Ramp rate (0 = clean run)        |
| injection_angle_deg  | 90.0    | degrees | Attack direction angle           |
| physical_accel_max   | 2000.0  | m/s²    | Kinematic consistency limit     |
| range_injection_scale| 5000.0  | m/rad   | Scale for range-channel injection|
| active               | false   | bool    | Attack enable flag               |

**attacker.optimized** (when `mode: optimized`): `chi2_margin` (fraction of chi-squared threshold), `du_max_bearing` (rad per step), `du_max_range` (m per step).

---

## monitor

| Key         | Default | Units   | Description                     |
|-------------|---------|---------|---------------------------------|
| alpha       | 0.05    | —       | False alarm significance level  |
| n_z         | 2       | —       | Measurement dimension (bearing+range) |
| window_size | 10      | steps   | Rolling detection window size   |

---

## Programmatic Override Example

```python
from src.simulation_runner import SPECTRESimulation

sim = SPECTRESimulation.from_config_override(
    "config/sim_config.yaml",
    {
        "attacker.injection_rate":       0.02,
        "attacker.injection_angle_deg":  45.0,
        "attacker.active":               True,
        "ekf.R_diag":                    [0.000001, 25.0]
    }
)
results = sim.run(seed=42)
print(results["miss_distance"])
```
