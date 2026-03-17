import numpy as np
import pandas as pd


class InjectionAttacker:
    """
    Adversarial false target injection attacker.

    Attack model:
        delta_z(t) = I_dot * (t - t_start)   [ramp injection, radians]

    Injected measurement:
        z_injected = z_true + delta_z(t)

    The injection direction theta_inj rotates the ramp offset into the
    2D engagement plane, enabling directional control of the miss vector.

    Kinematic consistency constraint:
        The implied target acceleration from the ramp must not exceed
        physical_accel_max. If it does, a ValueError is raised at init.
    """

    def __init__(self, config: dict):
        self.config = config
        atk_cfg = config["attacker"]
        self.dt = config["simulation"]["dt"]

        # Attack parameters
        self.injection_rate = float(atk_cfg["injection_rate"])
        self.injection_angle = float(
            np.radians(atk_cfg["injection_angle_deg"])
        )
        self.physical_accel_max = float(atk_cfg["physical_accel_max"])
        self.enabled = bool(atk_cfg["active"])

        # Validate kinematic consistency at init
        self._validate_kinematic_consistency()

        # Runtime state
        self.t_start = None
        self._active = False
        self.history = []
        self.t_current = 0.0

    def _validate_kinematic_consistency(self) -> None:
        """
        Ensure injection ramp rate implies physically realizable
        target acceleration. Raises ValueError if violated.

        Implied acceleration = injection_rate / dt
        (worst-case single-step delta in measurement space,
         scaled to translational acceleration approximation)
        """
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        implied_accel = abs(self.injection_rate) / self.dt
        if implied_accel > self.physical_accel_max:
            raise ValueError(
                f"Kinematic consistency violated: "
                f"implied acceleration {implied_accel:.2f} m/s^2 "
                f"exceeds physical_accel_max "
                f"{self.physical_accel_max:.2f} m/s^2. "
                f"Reduce injection_rate or increase "
                f"physical_accel_max in config."
            )

    def reset(self) -> None:
        """Reset attacker to pre-activation state."""
        self.t_start = None
        self._active = False
        self.history = []
        self.t_current = 0.0

    def activate(self, t_lock: float) -> None:
        """
        Activate the attacker at EKF acquisition lock time.

        Args:
            t_lock: simulation time at which EKF acquired lock [s]
        """
        if not self.enabled:
            return
        self.t_start = t_lock
        self._active = True

    def get_current_offset(self) -> float:
        """
        Compute current injection ramp offset delta_z(t).
        Returns 0.0 if attacker is not active.
        """
        if not self._active or self.t_start is None:
            return 0.0
        return self.injection_rate * (self.t_current - self.t_start)

    def compute_injection(self, t: float, z_true: float) -> float:
        """
        Compute injected measurement for current timestep.

        The scalar ramp delta_z is projected onto the bearing
        measurement axis using injection_angle:
            delta_z_bearing = delta_z * cos(injection_angle)

        This allows directional control: varying theta_inj rotates
        the false target position in the engagement plane.

        Args:
            t      : current simulation time [s]
            z_true : true angular bearing measurement [rad]

        Returns:
            z_injected: bearing measurement seen by EKF [rad]
        """
        self.t_current = t

        if not self._active or not self.enabled:
            # Transparent pass-through when inactive
            self._log(t, False, 0.0, z_true, z_true)
            return z_true

        # Ramp offset — scalar
        delta_z = self.get_current_offset()

        # Project onto bearing axis using injection angle
        delta_z_bearing = delta_z * np.cos(self.injection_angle)

        # Injected measurement
        z_injected = z_true + delta_z_bearing

        self._log(t, True, delta_z_bearing, z_true, z_injected)
        return z_injected

    def _log(self, t: float, active: bool, delta_z: float,
             z_true: float, z_injected: float) -> None:
        """Append one record to history."""
        self.history.append({
            "t":            t,
            "active":       int(active),
            "delta_z":      delta_z,
            "theta_inj":    np.degrees(self.injection_angle),
            "z_true":       z_true,
            "z_injected":   z_injected
        })

    def is_active(self) -> bool:
        return self._active and self.enabled

    def export_history(self) -> pd.DataFrame:
        """
        Export full injection history as DataFrame.
        Columns: t, active, delta_z, theta_inj, z_true, z_injected
        """
        return pd.DataFrame(self.history)

