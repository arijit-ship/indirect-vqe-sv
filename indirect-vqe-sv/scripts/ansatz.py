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
    return_layer_details: bool=False,
    drift_type: str = "xy"
) -> dict:

    circuit_chunk = None

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
    drift_hamiltonian: Observable = create_xy_hamiltonian(num_qubits=num_qubits)
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

                if verbosity >= 3:
                    logger.info(f"{rotation_gate_noise_probability}-noise added to RX & RY gates with probability: {rotation_gate_noise_probability}")

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

                if verbosity >= 3:
                    logging.info(f"RX, RY gates added without any noise")

            except Exception as e:
                raise ValueError(f"[ EROR ] Misconfiguration: {e}") from e

        # Increment of the flag depends on the size of the control unit.
        # For XY drift, size of the control unit is 4.
        # For Heisenberg drift, the size of the control unit is 2.
        flag += {"xy": 4, "heisenberg": 2}.get(drift_type.lower(), 0)

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
                logger.info(f"{cz_gate_noise_type}-noise added to CZ gate with probability: {cz_gate_noise_probability}")

            except Exception as e:
                raise ValueError(f"[ EROR ] Misconfiguration: {e}") from e

        else:
            circuit.add_CZ_gate(0, 1)
            logger.info(f"CZ gate added without any noise")

        # Drift evolution
        ##################
        # Add multi-qubit U gate
        # Determine time boundaries
        if layer == 0:
            ti = 0.0
            tf = parameters[0]
        else:
            ti = parameters[layer - 1]
            tf = parameters[layer]

                # Current layer details
        current_layer_details = {
            "layer_index": layer,
            "time": [ti, tf],
            "drift_noise_probability": None,
            "trotter_subdivision": None,
        }

        # Drift evolution: single step, no Trotter subdivision
        if not trotter_subdivide_enabled:
            time_evo_gate = create_time_evo_unitary(drift_hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

            if drift_noise_enabled:
                drift_noise_probability: float = drift_noise_strength * (tf - ti)
                for i in range(num_qubits):
                    circuit.add_noise_gate(
                        Identity(i), drift_noise_type, drift_noise_probability
                    )
                current_layer_details["drift_noise_probability"] = drift_noise_probability

        # Drift evolution: Lie-Trotter subdivision
        else:
            circuit, trotter_details = lie_trotter_evo(
                num_qubits=num_qubits,
                circuit=circuit,
                tf=tf,
                ti=ti,
                delta_t=trotter_subdivide_step_length,
                drift_hamiltonian=drift_hamiltonian,
                noise_enabled=drift_noise_enabled,
                noise_type=drift_noise_type,
                noise_strength=drift_noise_strength,
                curent_layer_index=layer,
                verbosity=verbosity,
                return_layer_details=return_layer_details,
            )
            # Stored under its own key. It is None unless verbosity >= 3
            # and return_layer_details, because of how lie_trotter_evo builds it.
            current_layer_details["trotter_subdivision"] = trotter_details

        if return_layer_details:
            all_layer_details.append(current_layer_details)
        if verbosity >= 3:
            logger.info(f"Layer-{layer} details: {current_layer_details}")

        if circuit_chunk is None:
            circuit_chunk = circuit.copy()
            if verbosity >= 3:
                logger.info(f"Circuit chunk copied at layer: {layer}")

    ansatz_dict = {
        "circuit": circuit,
        "layer_details": all_layer_details,
        "drift_hamiltonian_string": str(drift_hamiltonian),
        "circuit_chunk": circuit_chunk
    }
    return ansatz_dict


def lie_trotter_evo(
    num_qubits: int,
    circuit: QuantumCircuit,
    tf: float,
    ti: float,
    delta_t: float,
    drift_hamiltonian: Observable,
    curent_layer_index: int = 0,
    noise_enabled: bool = False,
    noise_type: str = "Depolarizing",
    noise_strength: float = 0.0,
    verbosity: int = 3,
    return_layer_details: bool = False,
):
    n_steps = int(np.ceil(abs(tf - ti) / delta_t)) if delta_t > 0 else 1
    if n_steps == 0:
        n_steps = 1

    #noise_probability = noise_strength * abs(tf - ti) / n_steps
    ti_sub = ti/n_steps
    tf_sub = tf/n_steps
    noise_probability_sub = noise_strength * abs(tf_sub - ti_sub)

    time_evo_gate = create_time_evo_unitary(
        hamiltonian=drift_hamiltonian, ti=ti_sub, tf=tf_sub
    )

    if verbosity >= 3:
        logger.info(f"Number of subdivisions of layer-{curent_layer_index}: {n_steps}")

    # Details for this layer, built only when requested (the cost function never asks)
    trotter_dict = None
    if return_layer_details:
        trotter_dict = {
            "total_steps": n_steps,
            "step_time": [ti_sub, tf_sub],
            "noise_prob": noise_probability_sub if noise_enabled else None,
        }

    for i in range(n_steps):
        circuit.add_gate(time_evo_gate)
        if noise_enabled:
            for q in range(num_qubits):
                circuit.add_noise_gate(Identity(q), noise_type, noise_probability_sub)
            if verbosity >= 3:
                logger.info(f"Subdivision layer-{curent_layer_index} noise-{noise_type} probability: {noise_probability_sub}")

    return circuit, trotter_dict


def create_ansatz_drift_heisenberg(): ...
