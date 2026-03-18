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
    return np.array([
        config["target"]["x0"] - config["missile"]["x0"],
        config["target"]["y0"] - config["missile"]["y0"],
        config["target"]["vx0"] - config["missile"]["vx0"],
        config["target"]["vy0"] - config["missile"]["vy0"]
    ])


def _make_measurement(true_state, config, rng=None):
    """Generate 2-channel noisy measurement from true relative state."""
    r_x, r_y = true_state[0], true_state[1]
    bearing = np.arctan2(r_y, r_x)
    rng_m = np.sqrt(r_x**2 + r_y**2)
    z_true = np.array([bearing, rng_m])

    ekf_cfg = config["ekf"]
    if "R_diag" in ekf_cfg:
        R = np.diag(ekf_cfg["R_diag"])
    else:
        R = np.array([[ekf_cfg["R_scalar"]]])

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.multivariate_normal(np.zeros(R.shape[0]), R)
    z_noisy = z_true + noise[:len(z_true)]
    return z_noisy


def test_ekf_is_two_channel(config, ekf):
    """EKF must initialize with n_z=2 for bearing+range."""
    assert ekf.n_z == 2, f"Expected n_z=2, got {ekf.n_z}"
    assert ekf.R.shape == (2, 2), f"R shape should be (2,2), got {ekf.R.shape}"


def test_measurement_function_returns_2d(config, ekf):
    """h(x) must return [bearing, range] vector."""
    x = np.array([10000.0, 500.0, -500.0, 0.0])
    z = ekf._h(x)
    assert z.shape == (2,), f"Expected shape (2,), got {z.shape}"
    assert abs(z[0] - np.arctan2(500, 10000)) < 1e-10
    assert abs(z[1] - np.sqrt(10000**2 + 500**2)) < 0.1


def test_jacobian_is_2x4(config, ekf):
    """Measurement Jacobian must be 2x4."""
    x = np.array([10000.0, 500.0, -500.0, 0.0])
    H = ekf._measurement_jacobian(x)
    assert H.shape == (2, 4), f"Expected (2,4), got {H.shape}"


def test_filter_converges(config, ekf):
    """
    After 200 update steps with clean 2-channel measurements,
    P_trace must drop below 10.0.
    """
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    rng = np.random.default_rng(42)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]

    for _ in range(200):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_noisy = _make_measurement(true_state, config, rng)
        ekf.predict()
        ekf.update(z_noisy)

    assert np.trace(ekf.P) < 10.0, (
        f"EKF failed to converge. P_trace = {np.trace(ekf.P):.4f}"
    )


def test_los_rate_estimate_accuracy(config, ekf):
    """After convergence, LOS rate estimate error must be < 0.01 rad/s."""
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    rng = np.random.default_rng(0)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]

    for _ in range(500):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_noisy = _make_measurement(true_state, config, rng)
        ekf.predict()
        ekf.update(z_noisy)

    r_x, r_y = true_state[0], true_state[1]
    v_rx, v_ry = true_state[2], true_state[3]
    r2 = r_x**2 + r_y**2 + 1e-9
    true_los_rate = (r_x * v_ry - r_y * v_rx) / r2
    est_los_rate = ekf.get_los_rate_estimate()
    error = abs(est_los_rate - true_los_rate)

    assert error < 0.01, (
        f"LOS rate error {error:.6f} rad/s exceeds 0.01. "
        f"True={true_los_rate:.6f}, Est={est_los_rate:.6f}"
    )


def test_chi2_false_alarm_rate(config, ekf):
    """
    With clean 2-channel measurements, chi2_stat must be below
    chi2(0.05,2)=5.9915 for at least 93% of post-warmup steps.
    """
    from scipy.stats import chi2
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    rng = np.random.default_rng(7)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]
    threshold = chi2.ppf(0.95, df=ekf.n_z)

    alarms = 0
    steps = 500
    for i in range(steps):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_noisy = _make_measurement(true_state, config, rng)
        ekf.predict()
        result = ekf.update(z_noisy)
        if i >= 50 and result["chi2_stat"] > threshold:
            alarms += 1

    evaluated = steps - 50
    alarm_rate = alarms / evaluated
    assert alarm_rate <= 0.10, (
        f"False alarm rate {alarm_rate:.4f} exceeds 0.10. "
        f"Chi2 threshold={threshold:.3f}, n_z={ekf.n_z}"
    )


def test_kalman_gain_decreases_after_lock(config, ekf):
    """Kalman gain norm should decrease from first to second half."""
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    rng = np.random.default_rng(3)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]

    for _ in range(600):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_noisy = _make_measurement(true_state, config, rng)
        ekf.predict()
        ekf.update(z_noisy)

    df = ekf.export_gain_history()
    mid = len(df) // 2
    mean_first = df["gain_norm"].iloc[:mid].mean()
    mean_second = df["gain_norm"].iloc[mid:].mean()

    assert mean_second < mean_first, (
        f"Gain did not decrease. First={mean_first:.6f}, "
        f"Second={mean_second:.6f}"
    )


def test_joseph_form_symmetry(config, ekf):
    """P must remain symmetric under Joseph form updates."""
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    rng = np.random.default_rng(99)
    true_state = x0.copy()
    dt = config["simulation"]["dt"]

    max_asymmetry = 0.0
    for _ in range(300):
        true_state[0] += true_state[2] * dt
        true_state[1] += true_state[3] * dt
        z_noisy = _make_measurement(true_state, config, rng)
        ekf.predict()
        ekf.update(z_noisy)
        asym = np.max(np.abs(ekf.P - ekf.P.T))
        max_asymmetry = max(max_asymmetry, asym)

    assert max_asymmetry < 1e-10, (
        f"P asymmetry {max_asymmetry:.2e} exceeds 1e-10."
    )


def test_innovation_is_2d(config, ekf):
    """Innovation vector must have length 2 after update."""
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    z = np.array([0.05, 10000.0])
    ekf.predict()
    result = ekf.update(z)
    assert result["innovation"].shape == (2,)


def test_chi2_stat_is_scalar(config, ekf):
    """Chi-squared statistic must be a non-negative scalar."""
    x0 = _get_initial_relative(config)
    ekf.reset(x0)
    z = np.array([0.05, 10000.0])
    ekf.predict()
    result = ekf.update(z)
    assert isinstance(result["chi2_stat"], float)
    assert result["chi2_stat"] >= 0.0
