import numpy as np
import pandas as pd
import yaml
import json
import os
import time
from datetime import datetime

from src.engagement.geometry import EngagementGeometry
from src.estimator.ekf_seeker import EKFSeeker
from src.guidance.pn_guidance import PNGuidance
from src.attacker.injection_attacker import InjectionAttacker
from src.monitor.chi2_monitor import Chi2InnovationMonitor


class SPECTRESimulation:
    """
    SPECTRE-SIM unified simulation runner.

    Orchestrates one complete missile-target engagement:
        1. Engagement geometry (RK4 physics)
        2. EKF seeker (state estimation)
        3. PN guidance (acceleration command)
        4. Adversarial attacker (false measurement injection)
        5. Chi-squared monitor (bad-data detection)

    Usage:
        sim = SPECTRESimulation("config/sim_config.yaml")
        results = sim.run(seed=42)
        sim.save_results("results/data/")
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dt = self.config["simulation"]["dt"]
        self.t_max = self.config["simulation"]["t_max"]

        # Instantiate all modules
        self.geometry = EngagementGeometry(self.config)
        self.ekf = EKFSeeker(self.config)
        self.guidance = PNGuidance(self.config)
        self.attacker = InjectionAttacker(self.config)
        self.monitor = Chi2InnovationMonitor(self.config)

        # Results placeholder
        self.results = {}

    def _get_initial_relative_state(self) -> np.ndarray:
        """Compute initial EKF state from config."""
        cfg = self.config
        return np.array([
            cfg["target"]["x0"]  - cfg["missile"]["x0"],
            cfg["target"]["y0"]  - cfg["missile"]["y0"],
            cfg["target"]["vx0"] - cfg["missile"]["vx0"],
            cfg["target"]["vy0"] - cfg["missile"]["vy0"]
        ], dtype=float)

    def run(self, seed: int = 0) -> dict:
        """
        Execute one complete engagement simulation.

        Args:
            seed: random seed for measurement noise reproducibility

        Returns:
            results dict with all metrics and DataFrames
        """
        np.random.seed(seed)

        # Reset all modules
        self.geometry.reset()
        self.ekf.reset(self._get_initial_relative_state())
        self.guidance.reset()
        self.attacker.reset()
        self.monitor.reset()

        # Run engagement loop
        self._simulation_loop()

        # Collect and store results
        self.results = self._collect_results()
        return self.results

    def _simulation_loop(self) -> None:
        """
        Main timestep loop. Runs until intercept or t_max.

        Data flow each step:
            geometry → true state
            true state → noisy measurement z_true
            attacker → z_measured (injected or clean)
            ekf.predict + ekf.update(z_measured) → state estimate
            ekf.get_los_rate_estimate + geometry.Vc → guidance command
            monitor.check(chi2_stat) → alarm flag
            guidance command → geometry.step (next timestep)
        """
        R_scalar = self.config["ekf"]["R_scalar"]
        accel_cmd = 0.0
        t = 0.0

        while True:
            # 1. Step geometry forward with current command
            state = self.geometry.step(accel_cmd)
            t = state["t"]

            # 2. Compute true angular measurement + noise
            dx = state["tx"] - state["mx"]
            dy = state["ty"] - state["my"]
            z_true = np.arctan2(dy, dx)
            noise = np.random.normal(0.0, np.sqrt(R_scalar))
            z_noisy = z_true + noise

            # 3. Attacker injects (or passes through clean)
            z_measured = self.attacker.compute_injection(t, z_noisy)

            # 4. EKF predict and update
            self.ekf.predict()
            ekf_out = self.ekf.update(z_measured)

            # 5. Activate attacker at EKF acquisition lock
            if self.ekf.is_locked() and not self.attacker.is_active():
                self.attacker.activate(t)

            # 6. Compute PN guidance command
            los_rate_hat = self.ekf.get_los_rate_estimate()
            Vc = state["Vc"]
            accel_cmd = self.guidance.compute_command(los_rate_hat, Vc)

            # 7. Monitor chi-squared statistic
            self.monitor.check(ekf_out["chi2_stat"])

            # 8. Check termination
            if self.geometry.is_intercept():
                break

    def _collect_results(self) -> dict:
        """
        Assemble complete results dictionary from all module outputs.
        """
        geo_df      = self.geometry.export_history()
        ekf_df      = self.ekf.export_gain_history()
        guidance_df = self.guidance.export_history()
        attacker_df = self.attacker.export_history()
        monitor_df  = self.monitor.export_history()

        # Estimate Ca: miss_distance / injection_rate (if attack active)
        inj_rate = self.config["attacker"]["injection_rate"]
        miss = self.geometry.get_miss_distance()
        Ca_estimate = (miss / inj_rate) if inj_rate > 1e-9 else None

        # Lock time: first t where EKF is locked in gain history
        lock_time = None
        if not ekf_df.empty:
            locked_rows = ekf_df[ekf_df["locked"] == True]
            if not locked_rows.empty:
                lock_time = float(locked_rows["t"].iloc[0])

        return {
            # Scalar metrics
            "miss_distance":    miss,
            "t_final":          self.geometry.t_current,
            "detection_rate":   self.monitor.get_detection_rate(),
            "max_chi2":         float(
                monitor_df["chi2_stat"].max()
                if not monitor_df.empty else 0.0
            ),
            "Ca_estimate":      Ca_estimate,
            "lock_time":        lock_time,
            "clipping_events":  self.guidance.get_clip_count(),
            "total_alarms":     self.monitor.get_total_alarms(),
            "injection_rate":   inj_rate,
            "seed":             int(np.random.get_state()[1][0]),

            # DataFrames
            "geometry_df":      geo_df,
            "ekf_df":           ekf_df,
            "guidance_df":      guidance_df,
            "attacker_df":      attacker_df,
            "monitor_df":       monitor_df,
        }

    def save_results(self, output_dir: str,
                     run_id: str = "run") -> None:
        """
        Save all DataFrames as CSV and scalar metrics as JSON.
        Enforces: each file < 99MB.

        Args:
            output_dir : directory to write results
            run_id     : prefix for filenames (e.g. 'clean', 'attack')
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save DataFrames
        df_keys = [
            "geometry_df", "ekf_df", "guidance_df",
            "attacker_df", "monitor_df"
        ]
        size_limit = 99 * 1024 * 1024  # 99 MB in bytes

        for key in df_keys:
            df = self.results.get(key)
            if df is not None and not df.empty:
                fname = os.path.join(
                    output_dir, f"{run_id}_{key}.csv"
                )
                df.to_csv(fname, index=False)
                fsize = os.path.getsize(fname)
                assert fsize < size_limit, (
                    f"File {fname} exceeds 99MB limit: "
                    f"{fsize / 1024**2:.2f} MB"
                )

        # Save scalar summary
        summary = {
            k: v for k, v in self.results.items()
            if not isinstance(v, pd.DataFrame)
        }
        summary_path = os.path.join(
            output_dir, f"{run_id}_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    @classmethod
    def from_config_override(
        cls,
        config_path: str,
        overrides: dict
    ) -> "SPECTRESimulation":
        """
        Create simulation instance with config parameter overrides.
        Used by experiments to sweep injection rates and angles
        without modifying the base config file.

        Args:
            config_path : path to base sim_config.yaml
            overrides   : flat dict of dotted-key overrides, e.g.
                          {"attacker.injection_rate": 0.05,
                           "attacker.active": True}

        Returns:
            SPECTRESimulation instance with overrides applied
        """
        import copy
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply overrides using dot notation
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = config
            for k in keys[:-1]:
                d = d[k]
            # Convert numpy types to native Python for YAML safety
            if hasattr(value, 'item'):
                value = value.item()
            d[keys[-1]] = value

        # Write to temp config and instantiate
        # (avoids modifying base config file)
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml",
            delete=False
        ) as tmp:
            yaml.dump(config, tmp)
            tmp_path = tmp.name

        instance = cls(tmp_path)
        os.unlink(tmp_path)
        return instance
