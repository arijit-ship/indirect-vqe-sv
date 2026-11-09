"""main.py: reads config.json, initializes Config, and executes VQE.
Acts strictly as an interface/orchestrator with zero core VQE/physics logic.

Usage:
    python run_vqe.py config-test.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.vqe import VQE


def get(d, *keys, default=None):
    """Safe nested lookup: get(raw, "system", "time", "total_time")."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


class Config:
    def __init__(self, raw):
        self.raw = raw                      # the full JSON, untouched

        # Raw configuration dictionaries required by VQE
        self.system = get(raw, "system", default={})
        self.noise = get(raw, "noise", default={})

        # ----- run / tasks -----
        self.task = get(raw, "run", "task")
        out = get(raw, "output", "output_dir", default=get(raw, "paths", "output_dir", default=get(raw, "run", "output_dir", default="../results")))
        self.output_dir = Path(out) if out is not None else None
        self.master_seed = get(raw, "run", "master_seed")
        self.verbosity = get(raw, "run", "verbosity", default=True)

        # ----- target observable -----
        self.target_observable_type = get(raw, "target_observable", "type")
        self.target_observable_coefficients = get(raw, "target_observable", "coefficients", default={})

        # ----- system properties -----
        self.n_qubits = get(self.system, "n_qubits")
        self.n_layers = get(self.system, "n_layers")
        self.control_unit_size = get(self.system, "control_unit_size")
        self.drift = get(self.system, "drift")
        self.ansatz = get(self.system, "ansatz")

        # ----- time -----
        time_eval = get(self.system, "time_evolution", default={})
        subdivide_dict = get(time_eval, "totter_subdivide", default=get(self.system, "time", "subdivide", default={}))
        
        self.total_time = get(time_eval, "total_time", default=get(self.system, "time", "total_time"))
        self.subdivide_enabled = get(subdivide_dict, "enabled", default=False)
        self.step_length = get(subdivide_dict, "step_length", default=0.1)

        # ----- noise -----
        drift_noise = get(self.noise, "drift_noise", default={})
        self.noise_type = get(drift_noise, "type", default=get(self.noise, "type"))
        strength = get(drift_noise, "strength", default=get(self.noise, "strength"))
        if strength is None:
            strength = get(drift_noise, "strenght", default=get(self.noise, "strenght"))
        self.noise_strength = strength

        # ----- control unit noise -----
        control_unit_noise = get(self.noise, "control_unit_noise", default={})
        self.control_noise_enabled = get(control_unit_noise, "enabled", default=False)
        self.control_noise_type = get(control_unit_noise, "type")
        self.control_noise_prob = get(control_unit_noise, "probability")

        # ----- parameters -----
        self.parameter_values = get(raw, "parameters", "values")
        
        # Keep random_init as a dictionary to support vqe.py's .get() usage
        rand_init = get(raw, "parameters", "random_initialization")
        if isinstance(rand_init, dict):
            self.random_init = rand_init
        elif isinstance(rand_init, bool):
            self.random_init = {"enabled": rand_init}
        else:
            self.random_init = {"enabled": True}

        self.monotonic_time = get(raw, "parameters", "monotonically_increasing_time_enabled", default=True)

        # ----- optimization -----
        self.optimization = get(raw, "optimization", default={})

    def parameter_source(self):
        """'values' if parameters are passed, otherwise 'random'."""
        if self.parameter_values is not None:
            return "values"
        
        enabled = self.random_init.get("enabled", True) if isinstance(self.random_init, dict) else self.random_init
        if not enabled:
            raise ValueError("parameters.values is empty, so parameters.random_initialization is needed")
        return "random"


def load_json(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Config file not found: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"Config file is not valid JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="Indirect Control VQE Execution Script")
    parser.add_argument("config", help="path to config.json", nargs="?", default="config-test.json")
    args = parser.parse_args()

    # 1. Load configuration
    raw_config = load_json(args.config)
    cfg = Config(raw_config)

    # 2. Setup output directories and save a copy of the config used
    if cfg.output_dir is not None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(cfg.output_dir / f"config_used_{timestamp}.json", "w") as f:
            json.dump(raw_config, f, indent=2)

    if cfg.verbosity >= 1:
        print("=" * 50)
        print(f"Starting Task: {cfg.task} | Qubits: {cfg.n_qubits}")
        print("=" * 50)

    # 3. Execution Setup
    start_time = time.time()

    # Initialize VQE solver instance
    vqe_solver = VQE(config=cfg)

    # Run optimization
    execution_output = vqe_solver.run()

    execution_output = {
        "min_cost": None
    }

    total_run_time = time.time() - start_time

    # 4. Save Final Output to files
    if cfg.output_dir is not None:
        summary_file = cfg.output_dir / f"summary_output_{timestamp}.json"
        summary_data = {
            "config": raw_config,
            "total_run_time_sec": total_run_time,
            "results": execution_output
        }
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        print("\n" + "=" * 50)
        print(f"Execution complete. Summary saved to: {summary_file.resolve()}")
        print("=" * 50)


if __name__ == "__main__":
    main()