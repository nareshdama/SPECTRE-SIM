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
        2. EKF seeker (state estimation, 2-channel bearing+range)
        3. PN guidance (acceleration command)
        4. Adversarial attacker (false measurement injection)
        5. Chi-squared monitor (bad-data detection)
    """

    ENDGAME_RANGE_THRESHOLD = 100.0  # meters

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dt = self.config["simulation"]["dt"]
        self.t_max = self.config["simulation"]["t_max"]

        self.geometry = EngagementGeometry(self.config)
        self.ekf = EKFSeeker(self.config)
        self.guidance = PNGuidance(self.config)
        self.attacker = InjectionAttacker(self.config)
        self.monitor = Chi2InnovationMonitor(self.config)

        self.results = {}

    def _get_initial_relative_state(self) -> np.ndarray:
        cfg = self.config
        return np.array([
            cfg["target"]["x0"]  - cfg["missile"]["x0"],
            cfg["target"]["y0"]  - cfg["missile"]["y0"],
            cfg["target"]["vx0"] - cfg["missile"]["vx0"],
            cfg["target"]["vy0"] - cfg["missile"]["vy0"]
        ], dtype=float)

    def _build_R_matrix(self) -> np.ndarray:
        ekf_cfg = self.config["ekf"]
        if "R_diag" in ekf_cfg:
            return np.diag(ekf_cfg["R_diag"]).astype(float)
        return np.array([[ekf_cfg["R_scalar"]]]).astype(float)

    def run(self, seed: int = 0) -> dict:
        np.random.seed(seed)

        self.geometry.reset()
        self.ekf.reset(self._get_initial_relative_state())
        self.guidance.reset()
        self.attacker.reset()
        self.monitor.reset()

        self._simulation_loop()

        self.results = self._collect_results()
        return self.results

    def _simulation_loop(self) -> None:
        """
        Main timestep loop. Supports both 1-channel (legacy) and
        2-channel (bearing+range) measurement models.
        """
        R_matrix = self._build_R_matrix()
        n_z = R_matrix.shape[0]
        is_2ch = (n_z >= 2)
        accel_cmd = 0.0
        t = 0.0

        while True:
            state = self.geometry.step(accel_cmd)
            t = state["t"]

            dx = state["tx"] - state["mx"]
            dy = state["ty"] - state["my"]
            r = state["range"]

            if is_2ch:
                z_true = np.array([np.arctan2(dy, dx), r])
                noise = np.random.multivariate_normal(
                    np.zeros(n_z), R_matrix
                )
                z_noisy = z_true + noise
            else:
                z_true = np.arctan2(dy, dx)
                noise = np.random.normal(0.0, np.sqrt(R_matrix[0, 0]))
                z_noisy = z_true + noise

            z_measured = self.attacker.compute_injection(t, z_noisy)

            self.ekf.predict()
            ekf_out = self.ekf.update(z_measured)

            if self.ekf.is_locked() and not self.attacker.is_active():
                self.attacker.activate(t)

            los_rate_hat = self.ekf.get_los_rate_estimate()
            Vc = state["Vc"]
            accel_cmd = self.guidance.compute_command(los_rate_hat, Vc)

            # Determine engagement phase for the monitor
            if not self.ekf.is_locked():
                phase = "startup"
            elif r < self.ENDGAME_RANGE_THRESHOLD:
                phase = "endgame"
            else:
                phase = "tracking"
            self.monitor.check(ekf_out["chi2_stat"], phase=phase)

            if self.geometry.is_intercept():
                break

    def _collect_results(self) -> dict:
        geo_df      = self.geometry.export_history()
        ekf_df      = self.ekf.export_gain_history()
        guidance_df = self.guidance.export_history()
        attacker_df = self.attacker.export_history()
        monitor_df  = self.monitor.export_history()

        inj_rate = self.config["attacker"]["injection_rate"]
        miss = self.geometry.get_miss_distance()
        Ca_estimate = (miss / inj_rate) if inj_rate > 1e-9 else None

        lock_time = None
        if not ekf_df.empty:
            locked_rows = ekf_df[ekf_df["locked"] == True]
            if not locked_rows.empty:
                lock_time = float(locked_rows["t"].iloc[0])

        return {
            "miss_distance":    miss,
            "t_final":          self.geometry.t_current,
            "detection_rate":   self.monitor.get_tracking_detection_rate(),
            "detection_rate_cumulative": self.monitor.get_detection_rate(),
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

            "geometry_df":      geo_df,
            "ekf_df":           ekf_df,
            "guidance_df":      guidance_df,
            "attacker_df":      attacker_df,
            "monitor_df":       monitor_df,
        }

    def save_results(self, output_dir: str,
                     run_id: str = "run") -> None:
        os.makedirs(output_dir, exist_ok=True)

        df_keys = [
            "geometry_df", "ekf_df", "guidance_df",
            "attacker_df", "monitor_df"
        ]
        size_limit = 99 * 1024 * 1024

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
        import copy
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = config
            for k in keys[:-1]:
                d = d[k]
            if hasattr(value, 'item'):
                value = value.item()
            d[keys[-1]] = value

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
