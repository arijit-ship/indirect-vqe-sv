import numpy as np
from qulacs import DensityMatrix, Observable, QuantumCircuit, QuantumState
from .createparam import create_random_params
from .ansatz import create_ansatz_drift_xy, create_ansatz_drift_heisenbrg
from .hamiltonian import create_xy_hamiltonian, create_tf_ising_hamiltonian
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class VQE:
    # Class-level constant declaration
    NO_ROTATION_GATES_PER_CONTROL_UNIT = 2

    def __init__(self, config) -> None:
        """Boilerplate VQE constructor.

        Stores configuration and target observable without executing any
        logic.
        """
        self.cfg = config
        self.system = config.system
        self.noise = config.noise
     
        self.observable_type = config.target_observable_type
        
        # System & Ansatz properties extracted from config
        self.nqubits = self.cfg.n_qubits
        self.n_layers = self.cfg.n_layers

        # Tracking counters
        self.cost_fn_calling_counter = 0
        self.iteration_count = 0

        # Verbosity
        self.verbosity: int = self.cfg.verbosity

        # Creating the targte observable
        try:
            obs_type = self.observable_type.lower()
            if obs_type == "tfih":
                logger.info(f"Target observable: {self.observable_type}")
                self.target_observable = create_tf_ising_hamiltonian(
                    num_qubits=self.nqubits,
                    j=config.target_observable_coefficients["j"],
                    h=config.target_observable_coefficients["h"]
                )
                logger.info(f"Target observable-{self.observable_type} created successfully.")
            else:
                raise ValueError(f"Unsupported observable type: {self.observable_type}")
        except Exception as e:
            raise ValueError(f"[ Error ] Misconfiguration in target observable: {e}")

        # Counter
        self.cost_fn_calling_counter: int = 0

    def run(self):
        """Run VQE.

        This method orchestrates the VQE process:
        1. Initializes parameters based on the configuration.
        2. Creates the ansatz circuit using the initialized parameters.
        3. Evaluates the cost function for the ansatz.
        4. Runs the optimization loop to minimize the cost function.

        Returns:
            dict: A dictionary containing optimization results and metadata.
        """
        # Step 1: Initialize parameters
        initial_params = self._initialize_parameters()

        # Step 2: Create ansatz circuit
        ansatz_circuit = self._create_ansatz(initial_params)

        # Step 3: Evaluate cost function (placeholder)
        cost_value = self._cost_function(initial_params)

        # Step 4: Run optimization loop (placeholder)
        optimization_results = self.run_optimization(initial_params)

       # return {
        #     "initial_params": initial_params,
        #     "ansatz_circuit": ansatz_circuit,
        #     "cost_value": cost_value,
        #     "optimization_results": optimization_results,
        #     "metadata": {
        #         "nqubits": self.nqubits,
        #         "n_layers": self.n_layers,
        #         "verbosity": self.verbosity
        #     }
        # }

    def _initialize_parameters(self) -> np.ndarray:
        """Initializes parameters based on the config settings:
        - If `values` is a list, uses those values directly.
        - If `values` is null and `random_initialization.enabled` is true, 
          sets up the seed and calls `create_random_params`.
        """
        parameter_source = self.cfg.parameter_source()

        if parameter_source == "values":
            return np.array(self.cfg.parameter_values, dtype=float)

        elif parameter_source == "random":
            random_init_config = self.cfg.random_init or {}
            
            seed = None
            if random_init_config.get("enabled", False):
                seed = random_init_config.get("seed")
                if seed is not None:
                    np.random.seed(seed)

            # Assuming control unit size (gateset) and total time are stored in config
            control_unit_size = getattr(self.cfg, "control_unit_size", 1)
            total_time = getattr(self.cfg, "total_time", 1.0)

            # Total number of angle parameters per layer in control unit
            per_layer_angle_params = control_unit_size * self.NO_ROTATION_GATES_PER_CONTROL_UNIT

            param__dict = create_random_params(
                per_layer_angle_params=per_layer_angle_params,
                n_layers=self.n_layers,
                time=total_time,
                seed=seed
            )
            initial_random_params = param__dict["combined_params"]
            initial_random_time_params = param__dict["time_params"]
            initial_random_angle_params = param__dict["angle_params"]

            if self.verbosity:
                # print(f"Randomly initialized parameters with seed: {seed}")
                # print(f"Time parameters (length {len(initial_random_time_params)}):", initial_random_time_params)
                # print(f"Angle parameters (length {len(initial_random_angle_params)}):", initial_random_angle_params)
                #print("Combined parameters:", initial_random_params)  

                logger.info(f"Randomly initialized parameters with seed: {seed}")

                logger.info(
                    f"Time parameters (length {len(initial_random_time_params)}): "
                    f"{initial_random_time_params}"
                )

                logger.info(
                    f"Angle parameters (length {len(initial_random_angle_params)}): "
                    f"{initial_random_angle_params}"
                )

            return np.array(initial_random_params, dtype=float)
        
        else:
            raise ValueError("Invalid parameter configuration source specified.")

        
    def _create_ansatz(self, params: np.ndarray):
        try:
            drift: str = self.cfg.drift
        except AttributeError:
            raise ValueError("Configuration is missing the 'drift' attribute.")

        drift_lower = drift.lower()

        if drift_lower == "xy":
            # XY drift have control unit size 2
            ansatz_dict = create_ansatz_drift_xy(
                system=self.system,
                noise=self.noise,
                parameters=params,
                verbosity=self.verbosity
            )
        elif drift_lower == "heisenberg":
            # # Heisenberg drift have control unit size 1
            ansatz_dict = create_ansatz_drift_heisenberg(
                            system=self.system,
                            noise=self.noise,
                            parameters=params,
                        )
        else:
            raise ValueError("Misconfigured 'drift'. Valid drifts are: XY, Heisenberg")
            
        return ansatz_dict
    

    def _cost_function(self, params: np.ndarray) -> float:

        quantum_state = DensityMatrix(self.nqubits)
        ansatz_dict = self._create_ansatz(params=params)
        circuit: QuantumCircuit = ansatz_dict["circuit"]
        circuit.update_quantum_state(quantum_state)
        cost_value: float = self.target_observable.get_expectation_value(quantum_state)
        if self.verbosity >= 1:
            logger.info(f"Current cost: {cost_value}")

        self.cost_fn_calling_counter += 1

        return cost_value

    def run_optimization(self, initial_params: np.ndarray) -> dict:
        """Boilerplate step: Plug in your optimization wrapper logic here."""
        # TODO: Implement optimization loop (e.g., scipy.optimize.minimize or custom optimizer)
        raise NotImplementedError("Implement your optimization loop here.")

    def draw_circuit(self):
        ...