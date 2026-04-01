"""Tests for optimization-constrained stealth injection."""

import copy
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.attacker.optimized_stealth_attacker import OptimizedStealthAttacker
from src.estimator.ekf_seeker import EKFSeeker


@pytest.fixture
def config():
    with open("config/sim_config.yaml", "r") as f:
        return yaml.safe_load(f)


def _relative_x0(cfg):
    return np.array(
        [
            cfg["target"]["x0"] - cfg["missile"]["x0"],
            cfg["target"]["y0"] - cfg["missile"]["y0"],
            cfg["target"]["vx0"] - cfg["missile"]["vx0"],
            cfg["target"]["vy0"] - cfg["missile"]["vy0"],
        ],
        dtype=float,
    )


def test_innovation_statistics_matches_update_chi2(config):
    """innovation_statistics(z) chi2 must match update(z) chi2 after predict."""
    ekf = EKFSeeker(config)
    ekf.reset(_relative_x0(config))
    z = np.array([np.arctan2(500.0, 10000.0), np.sqrt(10000.0**2 + 500.0**2)])
    ekf.predict()
    r, S, chi2_pre = ekf.innovation_statistics(z)
    out = ekf.update(z)
    assert chi2_pre == pytest.approx(out["chi2_stat"], rel=1e-9)
    assert r.shape == (2,)


def test_optimized_requires_ekf_when_active(config):
    cfg = copy.deepcopy(config)
    cfg["attacker"]["active"] = True
    cfg["attacker"]["mode"] = "optimized"
    atk = OptimizedStealthAttacker(cfg)
    atk.reset()
    atk.activate(0.0)
    z = np.array([0.05, 10000.0])
    with pytest.raises(ValueError, match="requires ekf"):
        atk.compute_injection(0.01, z, ekf=None)


def test_solve_du_respects_tau_safe(config):
    """Single-step solve keeps Mahalanobis of (r_base + du) below tau_safe."""
    cfg = copy.deepcopy(config)
    cfg["attacker"]["active"] = True
    cfg["attacker"]["injection_angle_deg"] = 0.0
    atk = OptimizedStealthAttacker(cfg)
    ekf = EKFSeeker(cfg)
    ekf.reset(_relative_x0(cfg))
    z_noisy = np.array([0.05, 10000.0])
    ekf.predict()
    r0, S, _ = ekf.innovation_statistics(z_noisy)
    du = atk._solve_du(r0, S)
    chi2 = OptimizedStealthAttacker._mahalanobis_sq(r0 + du, S)
    assert chi2 <= atk._tau_safe + 1e-4
