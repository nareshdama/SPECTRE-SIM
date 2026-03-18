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
    cfg = copy.deepcopy(config)
    cfg["attacker"]["injection_rate"] = injection_rate
    cfg["attacker"]["injection_angle_deg"] = angle_deg
    cfg["attacker"]["active"] = active
    return InjectionAttacker(cfg)


def test_inactive_passthrough_scalar(config):
    """Inactive attacker returns scalar z_true unchanged."""
    cfg = copy.deepcopy(config)
    cfg["attacker"]["active"] = False
    cfg["attacker"]["injection_rate"] = 0.05
    attacker = InjectionAttacker(cfg)
    attacker.reset()

    for i, z_true in enumerate([0.01, 0.05, -0.03]):
        t = i * config["simulation"]["dt"]
        z_inj = attacker.compute_injection(t, z_true)
        assert z_inj == z_true


def test_inactive_passthrough_2channel(config):
    """Inactive attacker returns 2D z_true unchanged."""
    cfg = copy.deepcopy(config)
    cfg["attacker"]["active"] = False
    cfg["attacker"]["injection_rate"] = 0.05
    attacker = InjectionAttacker(cfg)
    attacker.reset()

    z_true = np.array([0.05, 10000.0])
    z_inj = attacker.compute_injection(0.1, z_true)
    np.testing.assert_array_almost_equal(z_inj, z_true)


def test_2channel_bearing_injection(config):
    """
    At theta=0 (cos=1, sin=0), bearing channel gets full injection,
    range channel gets zero.
    """
    attacker = make_attacker(config, injection_rate=0.01,
                              angle_deg=0.0)
    attacker.reset()
    attacker.activate(5.0)

    z_true = np.array([0.05, 10000.0])
    z_inj = attacker.compute_injection(6.0, z_true)

    delta_bearing = z_inj[0] - z_true[0]
    delta_range = z_inj[1] - z_true[1]
    expected_bearing = 0.01 * 1.0 * np.cos(0.0)  # 0.01

    assert delta_bearing == pytest.approx(expected_bearing, abs=1e-10)
    assert abs(delta_range) < 1e-6


def test_2channel_range_injection(config):
    """
    At theta=90 (cos=0, sin=1), bearing channel gets zero,
    range channel gets full injection * range_scale.
    """
    attacker = make_attacker(config, injection_rate=0.01,
                              angle_deg=90.0)
    attacker.reset()
    attacker.activate(5.0)

    z_true = np.array([0.05, 10000.0])
    z_inj = attacker.compute_injection(6.0, z_true)

    delta_bearing = z_inj[0] - z_true[0]
    delta_range = z_inj[1] - z_true[1]

    range_scale = config["attacker"].get("range_injection_scale", 5000.0)
    expected_range = 0.01 * 1.0 * range_scale * np.sin(np.pi / 2)

    assert abs(delta_bearing) < 1e-9
    assert delta_range == pytest.approx(expected_range, rel=1e-6)


def test_2channel_diagonal_injection(config):
    """At theta=45, both channels receive equal-magnitude injection."""
    attacker = make_attacker(config, injection_rate=0.01,
                              angle_deg=45.0)
    attacker.reset()
    attacker.activate(0.0)

    z_true = np.array([0.0, 5000.0])
    z_inj = attacker.compute_injection(1.0, z_true)

    delta_bearing = z_inj[0] - z_true[0]
    delta_range = z_inj[1] - z_true[1]

    assert abs(delta_bearing) > 1e-6
    assert abs(delta_range) > 1e-6


def test_ramp_is_linear_2channel(config):
    """delta_z bearing grows linearly with time."""
    dt = config["simulation"]["dt"]
    I_dot = 0.005
    attacker = make_attacker(config, injection_rate=I_dot,
                              angle_deg=0.0)
    attacker.reset()
    attacker.activate(0.0)

    z_true = np.array([0.0, 10000.0])
    prev = attacker.compute_injection(0.0, z_true)
    expected_step = I_dot * dt

    for i in range(1, 100):
        t = i * dt
        curr = attacker.compute_injection(t, z_true)
        actual_step = curr[0] - prev[0]
        assert actual_step == pytest.approx(expected_step, rel=1e-6)
        prev = curr


def test_kinematic_consistency_raises(config):
    """
    Attacker must raise ValueError when implied translational
    acceleration (I_dot * r0) exceeds physical_accel_max.
    """
    cfg = copy.deepcopy(config)
    cfg["attacker"]["physical_accel_max"] = 100.0
    cfg["attacker"]["active"] = True
    # I_dot = 0.05, r0 ≈ 10012m, implied = 500.6 > 100
    cfg["attacker"]["injection_rate"] = 0.05

    with pytest.raises(ValueError, match="Kinematic consistency"):
        InjectionAttacker(cfg)


def test_export_history_2channel(config):
    """export_history() columns must include bearing and range."""
    attacker = make_attacker(config, injection_rate=0.005,
                              angle_deg=45.0)
    attacker.reset()
    attacker.activate(0.0)

    dt = config["simulation"]["dt"]
    z_true = np.array([0.05, 10000.0])
    for i in range(20):
        attacker.compute_injection(i * dt, z_true)

    df = attacker.export_history()
    assert "delta_bearing" in df.columns
    assert "delta_range" in df.columns
    assert "z_true_bearing" in df.columns
    assert "z_injected_range" in df.columns
    assert len(df) == 20
