import os
import logging

from functools import cached_property
import numpy as np
from qulacs import DensityMatrix, QuantumCircuit
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

import json
import matplotlib.pyplot as plt
from .ansatz import create_ansatz_drift_heisenberg, create_ansatz_drift_xy
from .config import Config
from .constraint import create_tf_fixed_constraint, create_time_constraints_with_mingap
from .createparam import create_random_params
from .hamiltonian import create_tf_ising_hamiltonian

logger = logging.getLogger(__name__)  # logging is configured in run_vqe.py


class VQE:
    """Variational optimization of a time-evolution ansatz for a target Hamiltonian.

    All settings live in `self.cfg`. Only values derived from it are stored here.

    Parameter vector layout: [time params (n_layers) | angle params].
    Both constraint builders assume the time parameters come first.
    """

    # Number of angle parameters per layer for each drift type
    ANGLE_PARAMS_PER_LAYER = {"xy": 4, "heisenberg": 2}

    def __init__(self, config) -> None:
        # Accept either a Config object or the raw JSON dict
        self.cfg = config if isinstance(config, Config) else Config(config)
        self._check_config()

        # Derived values
        self.drift = self.cfg.drift.lower()
        if self.drift not in self.ANGLE_PARAMS_PER_LAYER:
            raise ValueError(f"Unsupported drift type: {self.drift}")
        self.per_layer_angle_params = self.ANGLE_PARAMS_PER_LAYER[self.drift]
        self.n_time_params = self.cfg.n_layers
        self.n_angle_params = self.per_layer_angle_params * self.cfg.n_layers
        self.n_total_params = self.n_time_params + self.n_angle_params

        # Seed for random parameters
        # Seed can be none and an int
        self.seed = self.cfg.random_init.get("seed")
        # A new random seed will be created if self.seed == None
        self.new_random_seed = None

        self.target_observable = self._build_observable()

        # Runtime state
        self.cost_fn_calling_counter = 0
        self.iteration_count = 0

        # Circuit with optimized param
        self.ansatz_final: QuantumCircuit = None
        # circuit chunk
        self.circuit_chunk = None

        # Quantum States
        self.init_state = None
        self.final_state = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _check_config(self) -> None:
        """Catch missing config entries early with a readable message."""
        required = [
            "n_qubits",
            "n_layers",
            "drift",
            "total_time",
            "target_observable_type",
        ]
        missing = [k for k in required if getattr(self.cfg, k, None) is None]
        if not self.cfg.optimization.get("method"):
            missing.append("optimization.method")
        if missing:
            raise ValueError(f"Config is missing required entries: {missing}")

    def _build_observable(self):
        if self.cfg.target_observable_type.lower() != "tfih":
            raise ValueError(
                f"Unsupported observable type: {self.cfg.target_observable_type}"
            )
        coeffs = self.cfg.target_observable_coefficients
        return create_tf_ising_hamiltonian(
            num_qubits=self.cfg.n_qubits, j=coeffs["j"], h=coeffs["h"]
        )

    def reset_counters(self) -> None:
        self.cost_fn_calling_counter = 0
        self.iteration_count = 0

    # Exact solution
    @cached_property
    def min_eigenvalue(self) -> float:
        """Exact ground-state energy of the target observable (computed on first use).

        Dense diagonalization of a 2^n x 2^n matrix: fine up to roughly n = 12.
        """
        matrix = self.target_observable.get_matrix().toarray()
        return float(np.linalg.eigvalsh(matrix)[0])  # eigenvalues come sorted ascending

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _initialize_parameters(self) -> np.ndarray:
        """Use `parameters.values` if given, otherwise random initialization."""
        if self.cfg.parameter_source() == "values":
            params = np.array(self.cfg.parameter_values, dtype=float)
            if len(params) != self.n_total_params:
                raise ValueError(
                    f"parameters.values has length {len(params)}, "
                    f"expected {self.n_total_params} "
                    f"({self.n_time_params} time + {self.n_angle_params} angle)"
                )
            return params

        random_seed = None
        random_params = None

        if self.seed is not None:
            random_seed = self.seed
            #np.random.seed(self.seed)
        else:
            random_seed = int(np.random.randint(0, 2**31 - 1))
            self.new_random_seed = random_seed
            #np.random.seed(random_seed)

        random_params = create_random_params(
            per_layer_angle_params=self.per_layer_angle_params,
            n_layers=self.cfg.n_layers,
            time=self.cfg.total_time,
            seed=random_seed,
        )

        if self.cfg.verbosity >= 2:
            logger.info(f"Random init, seed={self.seed}")
            logger.info(
                f"For drift: {self.drift}, per layer angle param count: {self.per_layer_angle_params}"
            )
            logger.info(
                f"Time params ({len(random_params['time_params'])}): "
                f"{random_params['time_params']}"
            )
            logger.info(
                f"Angle params ({len(random_params['angle_params'])}): "
                f"{random_params['angle_params']}"
            )

        return np.array(random_params["combined_params"], dtype=float)

    # ------------------------------------------------------------------
    # Circuit and cost
    # ------------------------------------------------------------------
    def _create_ansatz(self, params: np.ndarray, return_layer_details: bool = False):
        # drift = self.cfg.drift.lower()
        if self.drift == "xy":
            return create_ansatz_drift_xy(
                system=self.cfg.system,
                noise=self.cfg.noise,
                parameters=params,
                verbosity=self.cfg.verbosity,
                return_layer_details=return_layer_details,
            )
        if self.drift == "heisenberg":
            return create_ansatz_drift_heisenberg(
                system=self.cfg.system,
                noise=self.cfg.noise,
                parameters=params,
            )
        raise ValueError(f"Unsupported drift type: {self.cfg.drift}")

    def _cost_function(
        self, params: np.ndarray, return_layer_details: bool = False
    ) -> float:
        state = DensityMatrix(self.cfg.n_qubits)
        circuit = self._create_ansatz(params, return_layer_details)["circuit"]
        circuit.update_quantum_state(state)
        cost = self.target_observable.get_expectation_value(state)

        self.cost_fn_calling_counter += 1
        if self.init_state is not None:
            self.init_state = self.init_state = state.copy()
        else:
            pass
        if self.cfg.verbosity >= 1:
            logger.info(
                f"Current cost: {cost} with counter: {self.cost_fn_calling_counter}"
            )
        return cost

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def _build_constraint(self, time_param_length=None, all_param_length=None):
        """Build scipy constraints from the config flags.

        The arguments default to this ansatz's sizes; pass them only to override.
        """
        if time_param_length is None:
            time_param_length = self.n_time_params
        if all_param_length is None:
            all_param_length = self.n_total_params

        constraints = []
        if self.cfg.monotonically_increasing_time:
            constraints.append(
                create_time_constraints_with_mingap(
                    time_params_length=time_param_length,
                    all_params_length=all_param_length,
                )
            )
        if self.cfg.fixed_final_time:
            constraints.append(
                create_tf_fixed_constraint(
                    tf_index=time_param_length - 1,  # last time parameter
                    all_params_length=all_param_length,
                    T_max=self.cfg.total_time,
                )
            )
        return constraints

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def run_optimization(self, parameters, constraint=None):
        """Minimize the cost. Returns (min_cost, optimized_params as a list)."""
        if constraint is None:
            constraint = self._build_constraint()

        opt_cfg = self.cfg.optimization
        opt = minimize(
            self._cost_function,
            parameters,
            method=opt_cfg["method"],
            constraints=constraint,
            tol=opt_cfg.get("tol"),
            options=dict(opt_cfg.get("options") or {}),
        )

        return opt

    def run(self) -> dict:
        initial_params = self._initialize_parameters()
        opt_solution = self.run_optimization(initial_params, self._build_constraint())
        optimized_parameters = opt_solution.x.tolist()
        min_cost = float(opt_solution.fun)

        # After optimization, excute the circuit again with optimized parameters
        ansatz = self._create_ansatz(
            params=optimized_parameters, return_layer_details=True
        )
        self.ansatz_final = ansatz["circuit"]
        layer_details = ansatz["layer_details"]
        drift_hamiltonian_string = ansatz["drift_hamiltonian_string"]
        self.circuit_chunk = ansatz["circuit_chunk"]
        # Exact solution
        exact_min_eigh_val = self.min_eigenvalue
        # Final state
        final_density_matrix = DensityMatrix(self.cfg.n_qubits)
        self.ansatz_final.update_quantum_state(final_density_matrix)
        self.final_state = final_density_matrix.copy()

        output_dict = {
            "exact_sol": exact_min_eigh_val,
            "min_cost": min_cost,
            "optimized_params": optimized_parameters,
            "new_seed": self.new_random_seed,
            "initial_params": initial_params.tolist(),
            "n_cost_evaluations": self.cost_fn_calling_counter,
            "opmizer_artifacts": opt_solution,
            "target_observable_string": str(self.target_observable),
            "drift_hamiltonian_string": drift_hamiltonian_string,
            "ansatz_layer_details_optimized": layer_details,
        }
        return output_dict

    def draw_circuit(
        self, file_path: str, dpi: int = 300, file_type: str | None = None
    ) -> None:
        """Save a drawing of the first ansatz layer (call after run())."""
        if self.circuit_chunk is None:
            logger.warning("No circuit chunk available. Run the optimization first.")
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            circuit_drawer(self.circuit_chunk, "mpl")
            plt.savefig(file_path, dpi=dpi, format=file_type, bbox_inches="tight")
            logger.info(f"Circuit figure saved to: {os.path.abspath(file_path)}")
        except Exception as e:
            logger.error(f"Could not draw the circuit: {e}")
        finally:
            plt.close("all")

    def export_circuit_json(self, export_json: bool = False) -> str | None:
        """Return the final ansatz circuit serialized as a JSON string."""
        if not export_json:
            return None

        if self.ansatz_final is None:
            logger.warning("No final ansatz circuit available to export. Run the optimization first.")
            return None

        try:
            circuit_data = self.ansatz_final.to_json()
            
            # If to_json() returns a raw JSON string, load it and re-dump with indent for readability
            if isinstance(circuit_data, str):
                try:
                    circuit_data = json.loads(circuit_data)
                except json.JSONDecodeError:
                    return circuit_data  # Return as-is if it's already a plain string

            return json.dumps(circuit_data, indent=2)
            
        except Exception as e:
            logger.error(f"Could not serialize circuit to JSON: {e}")
            return None

    @staticmethod
    def _matrix_to_json(state) -> dict:
        """Density matrix -> JSON-safe dict (JSON has no complex type)."""
        m = np.asarray(state.get_matrix())
        return {"real": m.real.tolist(), "imag": m.imag.tolist()}

    def export_quantum_states(self, export_initial_state: bool = False,
                            export_final_state: bool = False) -> dict:
        """Return the requested states as {"real": ..., "imag": ...} dicts (None if unavailable)."""
        states = {}
        for wanted, key, attr in (
            (export_initial_state, "initial_state", "init_state"),
            (export_final_state, "final_state", "final_state"),
        ):
            if not wanted:
                continue
            state = getattr(self, attr, None)
            if state is None:
                logger.warning(f"{key} is not available. Run the optimization first.")
                states[key] = None
            else:
                states[key] = self._matrix_to_json(state)
        return states


