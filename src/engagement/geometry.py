import numpy as np
import pandas as pd
from typing import Optional


class EngagementGeometry:
    """
    2D missile-target engagement geometry engine.
    Uses RK4 integration for state evolution.
    All units: meters, seconds, radians, m/s, m/s^2.
    """

    def __init__(self, config: dict):
        self.config = config
        self.dt = config["simulation"]["dt"]
        self.t_max = config["simulation"]["t_max"]

        # Initial conditions
        self.m0 = np.array([
            config["missile"]["x0"],
            config["missile"]["y0"],
            config["missile"]["vx0"],
            config["missile"]["vy0"]
        ], dtype=float)

        self.t0 = np.array([
            config["target"]["x0"],
            config["target"]["y0"],
            config["target"]["vx0"],
            config["target"]["vy0"]
        ], dtype=float)

        # Runtime state
        self.missile = self.m0.copy()
        self.target = self.t0.copy()
        self.t_current = 0.0
        self.accel_cmd = 0.0
        self.history = []
        self._min_range = np.inf
        self._min_range_t = 0.0
        self._intercepted = False

    def reset(self) -> None:
        self.missile = self.m0.copy()
        self.target = self.t0.copy()
        self.t_current = 0.0
        self.accel_cmd = 0.0
        self.history = []
        self._min_range = np.inf
        self._min_range_t = 0.0
        self._intercepted = False

    def _missile_dynamics(self, state: np.ndarray, accel: float) -> np.ndarray:
        """
        State: [x, y, vx, vy]
        Acceleration applied perpendicular to velocity vector (lateral).
        """
        vx, vy = state[2], state[3]
        speed = np.sqrt(vx**2 + vy**2) + 1e-9
        # Lateral acceleration: perpendicular to velocity
        ax = -accel * (vy / speed)
        ay =  accel * (vx / speed)
        return np.array([vx, vy, ax, ay])

    def _target_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Constant velocity target (no maneuver baseline)."""
        return np.array([state[2], state[3], 0.0, 0.0])

    def _rk4_step(self, state: np.ndarray, dynamics_fn, dt: float,
                  **kwargs) -> np.ndarray:
        """4th-order Runge-Kutta integrator."""
        k1 = dynamics_fn(state, **kwargs)
        k2 = dynamics_fn(state + 0.5 * dt * k1, **kwargs)
        k3 = dynamics_fn(state + 0.5 * dt * k2, **kwargs)
        k4 = dynamics_fn(state + dt * k3, **kwargs)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, accel_cmd: float) -> dict:
        """
        Advance simulation one timestep dt.
        Returns dict of current state observables.
        """
        self.accel_cmd = accel_cmd

        # Integrate states
        self.missile = self._rk4_step(
            self.missile, self._missile_dynamics,
            self.dt, accel=accel_cmd
        )
        self.target = self._rk4_step(
            self.target, self._target_dynamics, self.dt
        )
        self.t_current += self.dt

        # Compute observables
        los, los_rate = self.compute_los()
        r = self._range()
        Vc = self.compute_closing_velocity()

        # Track minimum range
        if r < self._min_range:
            self._min_range = r
            self._min_range_t = self.t_current

        # Intercept check (threshold accounts for discrete timestep)
        if r < 5.0:
            self._intercepted = True

        # Log
        record = {
            "t": self.t_current,
            "mx": self.missile[0], "my": self.missile[1],
            "mvx": self.missile[2], "mvy": self.missile[3],
            "tx": self.target[0], "ty": self.target[1],
            "tvx": self.target[2], "tvy": self.target[3],
            "los": los, "los_rate": los_rate,
            "range": r, "Vc": Vc
        }
        self.history.append(record)

        return record

    def compute_los(self):
        """
        True LOS angle (lambda) and LOS rate (lambda_dot).
        lambda_dot = (dx*dvy - dy*dvx) / r^2
        """
        dx = self.target[0] - self.missile[0]
        dy = self.target[1] - self.missile[1]
        dvx = self.target[2] - self.missile[2]
        dvy = self.target[3] - self.missile[3]
        r2 = dx**2 + dy**2 + 1e-9
        los = np.arctan2(dy, dx)
        los_rate = (dx * dvy - dy * dvx) / r2
        return los, los_rate

    def compute_closing_velocity(self) -> float:
        """
        Vc = -(r_vec dot v_rel) / |r|
        Positive when closing.
        """
        dx = self.target[0] - self.missile[0]
        dy = self.target[1] - self.missile[1]
        dvx = self.target[2] - self.missile[2]
        dvy = self.target[3] - self.missile[3]
        r = np.sqrt(dx**2 + dy**2) + 1e-9
        return -(dx * dvx + dy * dvy) / r

    def _range(self) -> float:
        dx = self.target[0] - self.missile[0]
        dy = self.target[1] - self.missile[1]
        return np.sqrt(dx**2 + dy**2)

    def is_intercept(self) -> bool:
        return self._intercepted or (self.t_current >= self.t_max)

    def get_miss_distance(self) -> float:
        """Returns minimum range achieved over entire flight."""
        return self._min_range

    def export_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

