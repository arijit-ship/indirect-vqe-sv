"""main.py: reads config.json, initializes Config, and executes VQE.
Acts strictly as an interface/orchestrator with zero core VQE/physics logic.

Usage:
    python main.py config-test.json
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

        # ----- run / tasks -----
        self.task = get(raw, "run", "task")
        out = get(raw, "paths", "output_dir", default=get(raw, "run", "output_dir", default="../results"))
        self.output_dir = Path(out) if out is not None else None
        self.master_seed = get(raw, "run", "master_seed")
        self.verbose = get(raw, "run", "verbose", default=True)

        # ----- target observable -----
        self.target_type = get(raw, "target_observable", "type")
        self.coefficients = get(raw, "target_observable", "coefficients", default={})

        # ----- system -----
        self.n_qubits = get(raw, "system", "n_qubits")
        self.n_layers = get(raw, "system", "n_layers")
        self.control_unit_size = get(raw, "system", "control_unit_size")
        self.drift = get(raw, "system", "drift")
        self.ansatz = get(raw, "system", "ansatz")

        # ----- time -----
        self.total_time = get(raw, "system", "time", "total_time")
        self.subdivide_enabled = get(raw, "system", "time", "subdivide", "enabled", default=False)
        self.step_length = get(raw, "system", "time", "subdivide", "step_length", default=0.1)

        # ----- noise -----
        self.noise_type = get(raw, "noise", "type")
        strength = get(raw, "noise", "strength")
        if strength is None:
            strength = get(raw, "noise", "strenght")
        self.noise_strength = strength

        # ----- control unit noise -----
        self.control_noise_enabled = get(raw, "noise", "control_unit_noise", "enabled", default=False)
        self.control_noise_type = get(raw, "noise", "control_unit_noise", "type")
        self.control_noise_prob = get(raw, "noise", "control_unit_noise", "probability")

        # ----- parameters -----
        self.parameter_values = get(raw, "parameters", "values")
        self.random_init = get(raw, "parameters", "random_initialization")
        self.monotonic_time = get(raw, "parameters", "monotonically_increasing_time_enabled", default=True)

        # ----- optimization -----
        self.optimization = get(raw, "optimization", default={})

    def parameter_source(self):
        """'values' if parameters are passed, otherwise 'random'."""
        if self.parameter_values is not None:
            return "values"
        if not self.random_init:
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
    parser.add_argument("config", help="path to config.json", nargs="?", default="config.json")
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

    if cfg.verbose:
        print("=" * 50)
        print(f"Starting Task: {cfg.task} | Qubits: {cfg.n_qubits}")
        print("=" * 50)

    # 3. Execution Setup
    start_time = time.time()

    # Initialize VQE solver instance
    vqe_solver = VQE(config=cfg)

    # Manually invoke workflow steps as you implement them:
    # initial_params = vqe_solver.initialize_parameters()
    # execution_output = vqe_solver.run_optimization(initial_params)

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