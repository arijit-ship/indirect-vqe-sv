"""main.py: reads config.json, initializes Config, and executes VQE.

Usage:
    python run_vqe.py config-test.json
"""

import argparse
import logging
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from scripts.vqe import VQE
from scripts.config import Config



logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Keep DEBUG for our own code, but silence noisy third-party libraries
for noisy in ("matplotlib", "PIL"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

def load_json(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"Config file not found: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"Config file is not valid JSON: {e}")

# Helper to convert NumPy types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    parser = argparse.ArgumentParser(description="Indirect Control VQE Execution Script")
    parser.add_argument("config", nargs="?", default="config.json", help="path to config.json")
    args = parser.parse_args()

    # Loading the configuration
    raw_config = load_json(args.config)
    cfg = Config(raw_config)

    # Setup timestamps and experiment IDs
    experiment_id = f"vqe_{int(time.time())}"
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_index = getattr(cfg, "run_index", 0)

    # File name prefix to be used in output log
    file_name_prefix = cfg.output.get("output_prefix", None)

    # 2. Setup output directories and save a copy of the config used
    if cfg.output_dir is not None:
        
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_dir / f"{file_name_prefix}_config_used_{experiment_id}_{timestamp_file}.json", "w") as f:
            json.dump(raw_config, f, indent=2)

    if cfg.verbosity >= 1:
        print("=" * 50)
        print(f"Starting Task: {cfg.task} | Qubits: {cfg.n_qubits}")
        print("=" * 50)

    # VQE execution with timer
    start_time = time.time()
    vqe_solver = VQE(config=cfg)
    execution_output = vqe_solver.run()
    total_run_time = time.time() - start_time
    timestamp_finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if cfg.verbosity>2:
       logging.info(f"VQE execution log: {execution_output}")

    # Saving the the vqe log in a JSON file
    if cfg.output_dir is not None:
        summary_file = cfg.output_dir / f"{file_name_prefix}_summary_output_{experiment_id}_{timestamp_file}.json"
        
        summary_data = {
            "meta": {
                "id": experiment_id,
                "datetime": timestamp_start,
                "datetime_finished": timestamp_finished,
                "run_index": run_index,
                "python_version": sys.version,
                "os": f"{platform.system()} {platform.release()}",
                "hostname": platform.node(),
                "cpu_logical_cores": os.cpu_count(),
                "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
            },
            "config": raw_config,
            "total_run_time_sec": total_run_time,
            "output": execution_output
        }
        
        # Recursively clean any lingering NumPy types before dumping
        clean_summary_data = convert_numpy_types(summary_data)

        with open(summary_file, "w") as f:
            json.dump(clean_summary_data, f, indent=2)

         # Draw the circuit (needs a file path, so it lives inside the output_dir block)
        if cfg.draw_cricuit_enabled:
            circuit_file = cfg.output_dir / f"{file_name_prefix}_circuit_{experiment_id}_{timestamp_file}"
            vqe_solver.draw_circuit(str(circuit_file), dpi=cfg.draw_cricuit_dpi)   
        print("\n" + "=" * 50)
        print(f"Execution complete. Summary saved to: {summary_file.resolve()}")
        print("=" * 50)



if __name__ == "__main__":
    main()