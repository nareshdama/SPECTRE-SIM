import pytest
import numpy as np
import yaml
import copy
import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from src.attacker.injection_attacker import InjectionAttacker


@pytest.fixture
def config():
    with open("config/sim_config.yaml", "r") as f:
        return yaml.safe_load(f)


def make_attacker(config, injection_rate=0.01,
                  angle_deg=90.0, active=True):
    """Helper: build attacker with custom parameters."""
    cfg = copy.deepcopy(config)
    cfg["attacker"]["injection_rate"] = injection_rate
    cfg["attacker"]["injection_angle_deg"] = angle_deg
    cfg["attacker"]["active"] = active
    return InjectionAttacker(cfg)


def test_inactive_passthrough(config):
    """
    Inactive attacker must return z_true unchanged for all timesteps.
    """
    cfg = copy.deepcopy(config)
    cfg["attacker"]["active"] = False
    cfg["attacker"]["injection_rate"] = 0.05
    attacker = InjectionAttacker(cfg)
    attacker.reset()

    z_true_values = [0.01, 0.05, -0.03, 0.1, -0.2]
    for i, z_true in enumerate(z_true_values):
        t = i * config["simulation"]["dt"]
        z_inj = attacker.compute_injection(t, z_true)
        assert z_inj == z_true, (
            f"Inactive attacker modified measurement at t={t:.3f}: "
            f"z_true={z_true}, z_injected={z_inj}"
        )


def test_ramp_value_at_t_plus_1(config):
    """
    At t = t_start + 1.0s, delta_z must equal I_dot * 1.0 exactly.
    """
    attacker = make_attacker(config, injection_rate=0.01,
                              angle_deg=0.0)  # angle=0 so cos(0)=1
    attacker.reset()
    t_start = 5.0
    attacker.activate(t_start)

    t_check = t_start + 1.0
    z_true = 0.05
    z_inj = attacker.compute_injection(t_check, z_true)
    expected_delta = 0.01 * 1.0  # I_dot * (t - t_start)
    expected_z = z_true + expected_delta

    assert z_inj == pytest.approx(expected_z, abs=1e-10), (
        f"Expected z_injected={expected_z:.8f}, got {z_inj:.8f}"
    )


def test_ramp_value_at_t_plus_2(config):
    """
    At t = t_start + 2.0s, delta_z must equal I_dot * 2.0 exactly.
    """
    attacker = make_attacker(config, injection_rate=0.01,
                              angle_deg=0.0)
    attacker.reset()
    t_start = 5.0
    attacker.activate(t_start)

    t_check = t_start + 2.0
    z_true = 0.05
    z_inj = attacker.compute_injection(t_check, z_true)
    expected_delta = 0.01 * 2.0
    expected_z = z_true + expected_delta

    assert z_inj == pytest.approx(expected_z, abs=1e-10), (
        f"Expected z_injected={expected_z:.8f}, got {z_inj:.8f}"
    )


def test_ramp_is_linear(config):
    """
    delta_z must increase by exactly I_dot * dt each step.
    Confirms linearity of ramp injection over 100 steps.
    """
    dt = config["simulation"]["dt"]
    I_dot = 0.005
    attacker = make_attacker(config, injection_rate=I_dot,
                              angle_deg=0.0)
    attacker.reset()
    t_start = 0.0
    attacker.activate(t_start)

    z_true = 0.0
    prev_z_inj = attacker.compute_injection(t_start, z_true)
    expected_step = I_dot * dt

    for i in range(1, 100):
        t = t_start + i * dt
        curr_z_inj = attacker.compute_injection(t, z_true)
        actual_step = curr_z_inj - prev_z_inj
        assert actual_step == pytest.approx(expected_step, rel=1e-6), (
            f"Ramp non-linear at step {i}: "
            f"actual step={actual_step:.8f}, "
            f"expected={expected_step:.8f}"
        )
        prev_z_inj = curr_z_inj


def test_kinematic_consistency_raises(config):
    """
    InjectionAttacker must raise ValueError when injection_rate / dt
    exceeds physical_accel_max.
    """
    cfg = copy.deepcopy(config)
    dt = cfg["simulation"]["dt"]            # 0.01
    cfg["attacker"]["physical_accel_max"] = 100.0
    cfg["attacker"]["active"] = True

    # I_dot = 5.0, implied_accel = 5.0 / 0.01 = 500 > 100
    cfg["attacker"]["injection_rate"] = 5.0

    with pytest.raises(ValueError, match="Kinematic consistency violated"):
        InjectionAttacker(cfg)


def test_export_history_columns(config):
    """
    export_history() must return DataFrame with exactly 6 columns
    in the correct order.
    """
    attacker = make_attacker(config, injection_rate=0.005,
                              angle_deg=45.0)
    attacker.reset()
    attacker.activate(0.0)

    dt = config["simulation"]["dt"]
    for i in range(20):
        attacker.compute_injection(i * dt, 0.05)

    df = attacker.export_history()
    expected_cols = [
        "t", "active", "delta_z",
        "theta_inj", "z_true", "z_injected"
    ]
    assert list(df.columns) == expected_cols, (
        f"Expected columns {expected_cols}, got {list(df.columns)}"
    )
    assert len(df) == 20, (
        f"Expected 20 rows, got {len(df)}"
    )


def test_injection_angle_projects_correctly(config):
    """
    Injection angle theta=0 (cos=1.0) must give full delta_z.
    Injection angle theta=90 (cos=0.0) must give zero delta_z.
    Confirms cos(theta_inj) projection is correctly applied.
    """
    dt = config["simulation"]["dt"]
    t_start = 0.0
    t_check = 1.0  # 1 second after activation
    z_true = 0.05
    I_dot = 0.01

    # theta = 0 deg: cos(0) = 1.0 → full injection
    atk0 = make_attacker(config, injection_rate=I_dot, angle_deg=0.0)
    atk0.reset()
    atk0.activate(t_start)
    z0 = atk0.compute_injection(t_check, z_true)
    delta0 = z0 - z_true
    assert delta0 == pytest.approx(I_dot * t_check, rel=1e-6), (
        f"theta=0: expected delta={I_dot*t_check:.6f}, got {delta0:.6f}"
    )

    # theta = 90 deg: cos(90) = 0.0 → zero injection
    atk90 = make_attacker(config, injection_rate=I_dot, angle_deg=90.0)
    atk90.reset()
    atk90.activate(t_start)
    z90 = atk90.compute_injection(t_check, z_true)
    delta90 = z90 - z_true
    assert abs(delta90) < 1e-9, (
        f"theta=90: expected delta~0, got {delta90:.2e}"
    )

