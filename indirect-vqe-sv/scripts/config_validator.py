import numpy as np
from qulacs import QuantumCircuit


def validate_system_config(system: dict) -> dict:
    """Validates and extracts parameters from the system configuration dictionary."""
    if system is None:
        raise ValueError("[MISSING PARAMETER]: system configuration dictionary is missing.")

    n_qubits = system.get("n_qubits")
    if n_qubits is None:
        raise ValueError("[MISSING PARAMETER]: n_qubits is missing. Check the configuration file.")

    n_layers = system.get("n_layers")
    if n_layers is None:
        raise ValueError("[MISSING PARAMETER]: n_layers is missing. Check the configuration file.")

    control_unit_size = system.get("control_unit_size")
    if control_unit_size is None:
        raise ValueError("[MISSING PARAMETER]: control_unit_size is missing. Check the configuration file.")

    drift = system.get("drift")
    if drift is None:
        raise ValueError("[MISSING PARAMETER]: drift is missing. Check the configuration file.")

    time_config = system.get("time", {})
    total_time = time_config.get("total_time")
    if total_time is None:
        raise ValueError("[MISSING PARAMETER]: total_time is missing. Check the configuration file.")

    subdivide_config = time_config.get("subdivide", {})
    subdivide_enabled = subdivide_config.get("enabled", False)
    step_length = subdivide_config.get("step_length", 0)

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "control_unit_size": control_unit_size,
        "drift": drift,
        "total_time": total_time,
        "subdivide_enabled": subdivide_enabled,
        "step_length": step_length
    }


def validate_noise_config(noise: dict) -> dict:
    """Validates and extracts parameters from the noise configuration dictionary."""
    if noise is None:
        raise ValueError("[MISSING PARAMETER]: noise configuration dictionary is missing.")

    # 1. Drift noise
    drift_noise_cfg = noise.get("drift_noise")
    if drift_noise_cfg is None:
        raise ValueError("[MISSING PARAMETER]: 'drift_noise' is missing in noise config.")
    
    drift_enabled = drift_noise_cfg.get("enabled")
    if drift_enabled is None:
        raise ValueError("[MISSING PARAMETER]: 'drift_noise.enabled' is missing.")
        
    drift_type = drift_noise_cfg.get("type")
    if drift_type is None:
        raise ValueError("[MISSING PARAMETER]: 'drift_noise.type' is missing.")
        
    drift_strength = drift_noise_cfg.get("strength")
    if drift_strength is None:
        raise ValueError("[MISSING PARAMETER]: 'drift_noise.strength' is missing.")

    # 2. Control unit noise
    control_noise_cfg = noise.get("control_unit_noise")
    if control_noise_cfg is None:
        raise ValueError("[MISSING PARAMETER]: 'control_unit_noise' is missing in noise config.")
        
    control_enabled = control_noise_cfg.get("enabled")
    if control_enabled is None:
        raise ValueError("[MISSING PARAMETER]: 'control_unit_noise.enabled' is missing.")
        
    control_type = control_noise_cfg.get("type")
    if control_type is None:
        raise ValueError("[MISSING PARAMETER]: 'control_unit_noise.type' is missing.")
        
    control_prob = control_noise_cfg.get("probability")
    if control_prob is None:
        raise ValueError("[MISSING PARAMETER]: 'control_unit_noise.probability' is missing.")

    # 3. CZ gate noise
    cz_noise_cfg = noise.get("cz_gate_noise")
    if cz_noise_cfg is None:
        raise ValueError("[MISSING PARAMETER]: 'cz_gate_noise' is missing in noise config.")
        
    cz_enabled = cz_noise_cfg.get("enabled")
    if cz_enabled is None:
        raise ValueError("[MISSING PARAMETER]: 'cz_gate_noise.enabled' is missing.")
        
    cz_type = cz_noise_cfg.get("type")
    if cz_type is None:
        raise ValueError("[MISSING PARAMETER]: 'cz_gate_noise.type' is missing.")
        
    cz_prob = cz_noise_cfg.get("probability")
    if cz_prob is None:
        raise ValueError("[MISSING PARAMETER]: 'cz_gate_noise.probability' is missing.")

    return {
        "drift_enabled": drift_enabled,
        "drift_type": drift_type,
        "drift_strength": drift_strength,
        "control_enabled": control_enabled,
        "control_type": control_type,
        "control_prob": control_prob,
        "cz_enabled": cz_enabled,
        "cz_type": cz_type,
        "cz_prob": cz_prob
    }
