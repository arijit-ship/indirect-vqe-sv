"""Ansatz: control unit (RX/RY on the first k qubits, CZ on qubits 1-2) followed by drift time evolution.

Layer structure (repeated n_layers times):
    1. RX, RY on qubits 0 .. k-1          (k = control_unit_size)
    2. CZ(0, 1), always, regardless of k
    3. Drift evolution exp(-i H t) on all qubits, optionally Trotter-subdivided,
       with time-dependent noise after each (sub)step.

parameterseter layout (length = n_layers + 2 * k * n_layers):
    parameterseters[0 : n_layers]      -> evolution times t_1 ... t_L (absolute; layer l evolves t_{l-1} -> t_l, t_{-1} = 0)
    parameterseters[n_layers : ]       -> per layer, per control qubit i: [RX angle, RY angle]
"""

from typing import List, Tuple

import numpy as np
import logging
from qulacs import QuantumCircuit, Observable
from qulacs.gate import CZ, RX, RY, Identity

from .config_validator import validate_noise_config, validate_system_config
from .hamiltonian import create_xy_hamiltonian
from .time_evolution_gate import create_time_evo_unitary


from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def create_ansatz_drift_xy(
    system: dict,
    noise: dict,
    parameters: List,
    verbosity: int = 1,
    return_layer_details=False,
) -> dict:

    try:
        # Qubit and layer
        num_qubits: int = system["n_qubits"]
        num_layers: int = system["n_layers"]

        # Time evolution
        total_evolution_time: float = system["time_evolution"]["total_time"]
        trotter_subdivide_enabled: bool = system["time_evolution"]["totter_subdivide"][
            "enabled"
        ]
        trotter_subdivide_step_length: float = system["time_evolution"][
            "totter_subdivide"
        ]["step_length"]

        # Noise in control unit
        rotation_gate_noise_enabled: bool = noise["control_unit_noise"]["enabled"]
        rotation_gate_noise_probability: float = noise["control_unit_noise"][
            "probability"
        ]
        rotation_gate_noise_type: str = noise["control_unit_noise"]["type"]

        # Noise in CZ
        cz_gate_noise_enabled: bool = noise["cz_gate_noise"]["enabled"]
        cz_gate_noise_probability: float = noise["cz_gate_noise"]["probability"]
        cz_gate_noise_type: str = noise["cz_gate_noise"]["type"]

        # Noise in drift
        drift_noise_enabled: bool = noise["drift_noise"]["enabled"]
        drift_noise_strength: float = noise["drift_noise"]["strength"]
        drift_noise_type: str = noise["drift_noise"]["type"]

    except KeyError as e:
        raise KeyError(f"[ EROR ] Misconfiguration: {e}")

    # Drift Hamiltonian is XY
    xy_drift_hamiltonian: Observable = create_xy_hamiltonian(num_qubits=num_qubits)
    circuit: QuantumCircuit = QuantumCircuit(num_qubits)
    flag: int = num_layers  # Tracking angles in param ndarrsy

    # Initialize layer details container if tracking is enabled
    all_layer_details = [] if return_layer_details else None

    for layer in range(num_layers):

        # Control Unit
        if rotation_gate_noise_enabled:
            try:
                # Add Rx to first and second qubits
                circuit.add_noise_gate(
                    RX(0, parameters[flag]),
                    rotation_gate_noise_type,
                    rotation_gate_noise_probability,
                )
                circuit.add_noise_gate(
                    RX(1, parameters[flag + 1]),
                    rotation_gate_noise_type,
                    rotation_gate_noise_probability,
                )
                # Add Ry to first and second qubits
                circuit.add_noise_gate(
                    RY(0, parameters[flag + 2]),
                    rotation_gate_noise_type,
                    rotation_gate_noise_probability,
                )
                circuit.add_noise_gate(
                    RY(1, parameters[flag + 3]),
                    rotation_gate_noise_type,
                    rotation_gate_noise_probability,
                )

            except Exception as e:
                raise ValueError(f"[ EROR ] Misconfiguration: {e}") from e
        else:
            try:
                # Add Rx to first and second qubits
                circuit.add_gate(RX(0, parameters[flag]))
                circuit.add_gate(RX(1, parameters[flag + 1]))
                # Add Ry to first and second qubits
                circuit.add_gate(RY(0, parameters[flag + 2]))
                circuit.add_gate(RY(1, parameters[flag + 3]))

            except Exception as e:
                raise ValueError(f"[ EROR ] Misconfiguration: {e}") from e

        # CZ Gate
        if cz_gate_noise_enabled:
            try:
                circuit.add_CZ_gate(0, 1)
                circuit.add_noise_gate(
                    Identity(0), cz_gate_noise_type, cz_gate_noise_probability
                )
                circuit.add_noise_gate(
                    Identity(1), cz_gate_noise_type, cz_gate_noise_probability
                )

            except Exception as e:
                raise ValueError(f"[ EROR ] Misconfiguration: {e}") from e

        else:
            circuit.add_CZ_gate(0, 1)

        # Drift evolution
        ## Add multi-qubit U gate
        # 2. Determine time boundaries
        if layer == 0:
            ti = 0.0
            tf = parameters[0]
            if verbosity >= 3:
                logger.info(f"Layer-{layer} stating time: {ti}")
        else:
            ti = parameters[layer - 1]
            tf = parameters[layer]

        if not trotter_subdivide_enabled:
            # logger.info(f"Layer-{layer}")
            time_evo_gate = create_time_evo_unitary(xy_drift_hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)
            if drift_noise_enabled:
                drift_noise_probability: float = drift_noise_strength * (tf - ti)
                for i in range(num_qubits):
                    circuit.add_noise_gate(
                        Identity(i), drift_noise_type, drift_noise_probability
                    )
                if return_layer_details:
                    current_details = {
                        "layer_index": layer,
                        "time": [ti, tf],
                        "noise_probability": drift_noise_probability,
                    }
                    all_layer_details.append(current_details)
                if verbosity >= 3:
                    logger.info(
                        f"Layer-{layer} noise-{drift_noise_type} probability: {drift_noise_probability}"
                    )
        else:

            logger.info(f"Trotter subdivisions of layer-{layer}")
            layer_details = None
            circuit, layer_details = lie_trotter_evo(
                num_qubits=num_qubits,
                circuit=circuit,
                tf=tf,
                ti=ti,
                delta_t=trotter_subdivide_step_length,
                drift_hamiltonian=xy_drift_hamiltonian,
                noise_enabled=drift_noise_enabled,
                noise_type=drift_noise_type,
                noise_strength=drift_noise_strength,
                curent_layer_index=layer,
                verbosity=verbosity,
                return_layer_details=False,
            )

    ansatz_dict = {
        "circuit": circuit,
        "layer_details": all_layer_details if all_layer_details else None,
    }
    return ansatz_dict


def lie_trotter_evo(
    num_qubits,
    circuit,
    tf,
    ti,
    delta_t,
    drift_hamiltonian,
    curent_layer_index=0,
    noise_enabled=False,
    noise_type="Depolarizing",
    noise_strength=0.0,
    verbosity: int = 3,
    return_layer_details: bool = False,
):
    n_i = int(np.ceil(abs(tf - ti) / delta_t)) if delta_t > 0 else 1
    if n_i == 0:
        n_i = 1

    noise_probability = noise_strength * abs(tf - ti) / n_i
    ti_sub = ti / n_i
    tf_sub = tf / n_i

    time_evo_gate = create_time_evo_unitary(
        hamiltonian=drift_hamiltonian, ti=ti_sub, tf=tf_sub
    )

    # Logging is cheap—safe to run whenever verbosity >= 3
    if verbosity >= 3:
        logger.info(f"Number of subdivisions of layer-{curent_layer_index}: {n_i}")
        logger.info(
            f"Subdivision layer-{curent_layer_index} noise-{noise_type} probability: {noise_probability}"
        )

    # Only allocate the heavy trotter tracking dictionary & list if requested
    layer_details = None
    if verbosity >= 3 and return_layer_details:
        layer_details = {
            "total_steps": n_i,
            "noise_probability": noise_probability,
            "steps": [],
        }

    # Fast loop execution
    for i in range(n_i):
        circuit.add_gate(time_evo_gate)
        if noise_enabled:
            for q in range(num_qubits):
                circuit.add_noise_gate(Identity(q), "Depolarizing", noise_probability)

        # Only append to the list if tracing is explicitly turned on
        if layer_details is not None:
            layer_details["steps"].append(
                {"deep_trotter": {"step_loc": i, "time": [ti_sub, tf_sub]}}
            )

    return circuit, layer_details


def create_ansatz_drift_heisenberg(): ...
