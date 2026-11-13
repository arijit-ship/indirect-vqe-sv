"""main.py: reads config.json, initializes Config, and executes VQE.

Usage:
    python run_vqe.py config-test.json
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

from scripts.config import Config
from scripts.vqe import VQE

# Constants
# Standardized file name chunks
CONFIG_FILE_NAME_CHUNK: str = "source_config"
VQE_LOG_FILE_NAME_CHUNK: str = "vqe_logs"
CIRCUIT_FIGURE_FILE_NAME_CHUNK: str = "circuit_figure"
INIT_QUANTUM_STATE_FILE_NAME_CHUNK: str = "quantum_state"
FINAL_QUANTUM_STATE_FILE_NAME_CHUNK: str = "final_state"
CIRCUIT_JSON_FILE_NAME_CHUNK: str = "circuit_json"

# File name prefix length safety cap
MAX_FILE_NAME_PREFIX_LENGTH: int = 30

# Repeating symbol in print
REPEAT_SYMBOL: int = 50

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


# Helper function for JSON dumping
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle NumPy booleans explicitly
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
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
    experiment_id = f"vqe_{uuid.uuid4().hex[:8]}"
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_index = getattr(cfg, "run_index", 0)

    # File name prefix to be used in output log
    # Retrieve prefix and trim to a reasonable maximum length (e.g., 30 chars)
    raw_prefix = cfg.output.get("output_prefix", "experiment")
    if raw_prefix:
        # Sanitize and trim: keeps alphanumeric characters and underscores, capped at 30 chars
        file_name_prefix = "".join(c for c in raw_prefix if c.isalnum() or c in ("_", "-"))[
            :MAX_FILE_NAME_PREFIX_LENGTH
        ]
    else:
        file_name_prefix = "vqe_run"

    # Setup output directory
    if cfg.output_dir is not None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checking for ourput directory")
        # with open(
        #     cfg.output_dir
        #     / f"{file_name_prefix}_{CONFIG_FILE_NAME_CHUNK}_{experiment_id}_{timestamp_file}.json",
        #     "w",
        # ) as f:
        #     json.dump(raw_config, f, indent=2)

    if cfg.verbosity >= 1:
        print("=" * REPEAT_SYMBOL)
        print(f"Starting Task: {cfg.task} | Qubits: {cfg.n_qubits}")
        print("=" * REPEAT_SYMBOL)

    # VQE execution with timer
    start_time = time.time()
    vqe_solver = VQE(config=cfg)
    execution_output = vqe_solver.run()
    total_run_time = time.time() - start_time
    timestamp_finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if cfg.verbosity > 2:
        logging.info(f"VQE execution log: {execution_output}")

    # Meta
    experiment_meta = {
        "id": experiment_id,
        "datetime": timestamp_start,
        "datetime_finished": timestamp_finished,
        # "run_index": run_index,
        "python_version": sys.version,
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }
    # Saving the the vqe log in a JSON file
    if cfg.output_dir is not None:
        summary_file = (
            cfg.output_dir / f"{file_name_prefix}_{VQE_LOG_FILE_NAME_CHUNK}_{experiment_id}_{timestamp_file}.json"
        )

        summary_data = {
            "meta": experiment_meta,
            "config": raw_config,
            "total_run_time_sec": total_run_time,
            "output": execution_output,
        }

        # Recursively clean any lingering NumPy types before dumping
        clean_summary_data = convert_numpy_types(summary_data)

        with open(summary_file, "w") as f:
            json.dump(clean_summary_data, f, indent=2)

        # Draw the circuit (needs a file path, so it lives inside the output_dir block)
        if cfg.draw_cricuit_enabled:
            circuit_file_path = (
                cfg.output_dir / f"{file_name_prefix}_{CIRCUIT_FIGURE_FILE_NAME_CHUNK}_{experiment_id}_{timestamp_file}"
            )
            vqe_solver.draw_circuit(str(circuit_file_path), dpi=cfg.draw_cricuit_dpi)

        if cfg.export_circut_json:
            circuit_json_str = vqe_solver.export_circuit_json(export_json=True)
            if circuit_json_str:
                circuit_json_file_path = (
                    cfg.output_dir / f"{CIRCUIT_JSON_FILE_NAME_CHUNK}_{experiment_id}_{timestamp_file}.json"
                )

                # Load the string back to a dict so json.dump indents the entire file properly
                try:
                    parsed_circuit_data = json.loads(circuit_json_str)
                except (json.JSONDecodeError, TypeError):
                    parsed_circuit_data = circuit_json_str  # Fallback if it's already a dict or raw text

                circuit_json_with_meta = {
                    "meta": experiment_meta,
                    "ansatz_circuit_optimized": parsed_circuit_data,
                }

                with open(circuit_json_file_path, "w") as f:
                    json.dump(circuit_json_with_meta, f, indent=2)

                logging.info(f"Circuit JSON saved to: {circuit_json_file_path.resolve()}")

        if cfg.export_initial_state or cfg.export_final_state:
            quantum_states = vqe_solver.export_quantum_states(
                export_initial_state=cfg.export_initial_state,
                export_final_state=cfg.export_final_state,
            )

            for key, name_chunk in (
                ("initial_state", INIT_QUANTUM_STATE_FILE_NAME_CHUNK),
                ("final_state", FINAL_QUANTUM_STATE_FILE_NAME_CHUNK),
            ):
                data = quantum_states.get(key)
                if data is None:
                    continue
                path = cfg.output_dir / f"{file_name_prefix}_{name_chunk}_{experiment_id}_{timestamp_file}.json"
                payload = convert_numpy_types({"meta": experiment_meta, key: data})
                path.write_text(json.dumps(payload, indent=2))
                logging.info(f"{key} saved to: {path.resolve()}")

        print("\n" + "=" * REPEAT_SYMBOL)
        print(f"Execution complete. Summary saved to: {summary_file.resolve()}")
        print("=" * REPEAT_SYMBOL)


if __name__ == "__main__":
    main()
