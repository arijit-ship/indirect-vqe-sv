import logging

from functools import cached_property
import numpy as np
from qulacs import DensityMatrix
from scipy.optimize import minimize

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
        drift = self.cfg.drift.lower()
        if drift not in self.ANGLE_PARAMS_PER_LAYER:
            raise ValueError(f"Unsupported drift type: {self.cfg.drift}")
        self.per_layer_angle_params = self.ANGLE_PARAMS_PER_LAYER[drift]
        self.n_time_params = self.cfg.n_layers
        self.n_angle_params = self.per_layer_angle_params * self.cfg.n_layers
        self.n_total_params = self.n_time_params + self.n_angle_params

        self.target_observable = self._build_observable()

        # Runtime state
        self.cost_fn_calling_counter = 0
        self.iteration_count = 0

    
    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _check_config(self) -> None:
        """Catch missing config entries early with a readable message."""
        required = ["n_qubits", "n_layers", "drift", "total_time", "target_observable_type"]
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

        seed = self.cfg.random_init.get("seed")
        if seed is not None:
            np.random.seed(seed)

        random_params = create_random_params(
            per_layer_angle_params=self.per_layer_angle_params,
            n_layers=self.cfg.n_layers,
            time=self.cfg.total_time,
            seed=seed,
        )

        if self.cfg.verbosity >= 2:
            logger.info(f"Random init, seed={seed}")
            logger.info(f"Time params ({len(random_params['time_params'])}): "
                        f"{random_params['time_params']}")
            logger.info(f"Angle params ({len(random_params['angle_params'])}): "
                        f"{random_params['angle_params']}")

        return np.array(random_params["combined_params"], dtype=float)

    # ------------------------------------------------------------------
    # Circuit and cost
    # ------------------------------------------------------------------
    def _create_ansatz(self, params: np.ndarray, return_layer_details: bool = False):
        drift = self.cfg.drift.lower()
        if drift == "xy":
            return create_ansatz_drift_xy(
                system=self.cfg.system,
                noise=self.cfg.noise,
                parameters=params,
                verbosity=self.cfg.verbosity,
                return_layer_details=return_layer_details,
            )
        if drift == "heisenberg":
            return create_ansatz_drift_heisenberg(
                system=self.cfg.system,
                noise=self.cfg.noise,
                parameters=params,
            )
        raise ValueError(f"Unsupported drift type: {self.cfg.drift}")

    def _cost_function(self, params: np.ndarray, return_layer_details: bool = False) -> float:
        state = DensityMatrix(self.cfg.n_qubits)
        circuit = self._create_ansatz(params, return_layer_details)["circuit"]
        circuit.update_quantum_state(state)
        cost = self.target_observable.get_expectation_value(state)

        self.cost_fn_calling_counter += 1
        if self.cfg.verbosity >= 1:
            logger.info(f"Current cost: {cost}")    
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
        opt_solution = self.run_optimization(
            initial_params, self._build_constraint()
        )
        optimized_parameters = opt_solution.x.tolist()
        min_cost = float(opt_solution.fun)

        # After optimization, excute the circuit again with optimized parameters
        ansatz = self._create_ansatz(params=optimized_parameters, return_layer_details=True)
        layer_details = ansatz["layer_details"] 

        # Exact solution
        exact_min_eigh_val = self.min_eigenvalue
        return {
            "exact_sol": exact_min_eigh_val,
            "min_cost": min_cost,
            "optimized_params":optimized_parameters,
            "initial_params": initial_params.tolist(),
            "n_cost_evaluations": self.cost_fn_calling_counter,
            "opmizer_artifacts": opt_solution,
            "ansatz_layer_details_optimized": layer_details
        }

    def draw_circuit(self): ...