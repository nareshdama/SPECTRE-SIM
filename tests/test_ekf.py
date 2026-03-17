import pytest
import numpy as np
import yaml
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.estimator.ekf_seeker import EKFSeeker
from src.engagement.geometry import EngagementGeometry


@pytest.fixture
def config():
    with open("config/sim_config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def ekf(config):
    return EKFSeeker(config)


@pytest.fixture
def geo(config):
    return EngagementGeometry(config)


def _get_initial_relative(config):
    """Helper: compute initial relative state from config."""
    return np.array([
        config["target"]["x0"] - config["missile"]["x0"],
        config["target"]["y0"] - config["missile"]["y0"],
        config["target"]["vx0"] - config["missile"]["vx0"],
        config["target"]["vy0"] - config["missile"]["vy0"]
    ])


def test_filter_converges(config, ekf):
    """
    After 200 update steps with clean measurements,
    P_trace must drop below 10.0 (filter has converged).
    """
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    np.random.seed(42)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]
    R_scalar = config["ekf"]["R_scalar"]

    for _ in range(200):
        # Propagate true state (constant velocity)
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt

        # Clean noisy measurement
        z_true = np.arctan2(true_state[1], true_state[0])
        z_noisy = z_true + np.random.normal(0, np.sqrt(R_scalar))

        ekf.predict()
        ekf.update(z_noisy)

    assert np.trace(ekf.P) < 10.0, (
        f"EKF failed to converge. P_trace = {np.trace(ekf.P):.4f}, "
        f"expected < 10.0 after 200 steps."
    )


def test_los_rate_estimate_accuracy(config, ekf):
    """
    After convergence, LOS rate estimate error must be < 0.01 rad/s.
    """
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    np.random.seed(0)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]
    R_scalar = config["ekf"]["R_scalar"]

    for _ in range(500):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_true = np.arctan2(true_state[1], true_state[0])
        z_noisy = z_true + np.random.normal(0, np.sqrt(R_scalar))
        ekf.predict()
        ekf.update(z_noisy)

    # True LOS rate
    r_x, r_y = true_state[0], true_state[1]
    v_rx, v_ry = true_state[2], true_state[3]
    r2 = r_x**2 + r_y**2 + 1e-9
    true_los_rate = (r_x * v_ry - r_y * v_rx) / r2

    est_los_rate = ekf.get_los_rate_estimate()
    error = abs(est_los_rate - true_los_rate)

    assert error < 0.01, (
        f"LOS rate estimate error {error:.6f} rad/s exceeds 0.01 rad/s. "
        f"True={true_los_rate:.6f}, Est={est_los_rate:.6f}"
    )


def test_chi2_false_alarm_rate(config, ekf):
    """
    With clean measurements, chi2_stat must be below chi2(0.05,1)=3.841
    for at least 93% of steps (allowing 7% false alarm margin).
    """
    from scipy.stats import chi2
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    np.random.seed(7)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]
    R_scalar = config["ekf"]["R_scalar"]
    threshold = chi2.ppf(0.95, df=1)

    alarms = 0
    steps = 500
    # Skip first 50 steps (filter warm-up)
    for i in range(steps):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_true = np.arctan2(true_state[1], true_state[0])
        z_noisy = z_true + np.random.normal(0, np.sqrt(R_scalar))
        ekf.predict()
        result = ekf.update(z_noisy)
        if i >= 50 and result["chi2_stat"] > threshold:
            alarms += 1

    evaluated = steps - 50
    alarm_rate = alarms / evaluated
    assert alarm_rate <= 0.07, (
        f"False alarm rate {alarm_rate:.4f} exceeds 0.07. "
        f"Chi2 threshold={threshold:.3f}."
    )


def test_kalman_gain_decreases_after_lock(config, ekf):
    """
    After acquisition lock, Kalman gain norm must be
    lower in second half of flight than first half.
    """
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    np.random.seed(3)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]
    R_scalar = config["ekf"]["R_scalar"]

    for _ in range(600):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_true = np.arctan2(true_state[1], true_state[0])
        z_noisy = z_true + np.random.normal(0, np.sqrt(R_scalar))
        ekf.predict()
        ekf.update(z_noisy)

    df = ekf.export_gain_history()
    mid = len(df) // 2
    mean_first = df["gain_norm"].iloc[:mid].mean()
    mean_second = df["gain_norm"].iloc[mid:].mean()

    assert mean_second < mean_first, (
        f"Kalman gain did not decrease after lock. "
        f"First half mean={mean_first:.6f}, "
        f"Second half mean={mean_second:.6f}"
    )


def test_joseph_form_symmetry(config, ekf):
    """
    Joseph form covariance update must keep P symmetric.
    Max element of |P - P^T| must be < 1e-10.
    """
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    np.random.seed(99)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]
    R_scalar = config["ekf"]["R_scalar"]

    max_asymmetry = 0.0
    for _ in range(300):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_true = np.arctan2(true_state[1], true_state[0])
        z_noisy = z_true + np.random.normal(0, np.sqrt(R_scalar))
        ekf.predict()
        ekf.update(z_noisy)
        asym = np.max(np.abs(ekf.P - ekf.P.T))
        max_asymmetry = max(max_asymmetry, asym)

    assert max_asymmetry < 1e-10, (
        f"P matrix asymmetry {max_asymmetry:.2e} exceeds 1e-10. "
        f"Joseph form not correctly implemented."
    )

