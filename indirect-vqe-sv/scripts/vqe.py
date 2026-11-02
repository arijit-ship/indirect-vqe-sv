import numpy as np
from .createparam import create_random_params

class VQE:
    # Class-level constant declaration
    NO_ROTATION_GATES_PER_CONTROL_UNIT = 2

    def __init__(self, config, observable=None) -> None:
        """Boilerplate VQE constructor.

        Stores configuration and target observable without executing any
        logic.
        """
        self.cfg = config
        self.system = config.system
        self.noise = config.noise
     
        self.observable = observable

        # System & Ansatz properties extracted from config
        self.nqubits = self.cfg.n_qubits
        self.n_layers = self.cfg.n_layers

        # Tracking counters
        self.cost_fn_calling_counter = 0
        self.iteration_count = 0

    def initialize_parameters(self) -> np.ndarray:
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
            
            return create_random_params(
                per_layer_angle_params=per_layer_angle_params,
                n_layers=self.n_layers,
                time=total_time,
                seed=seed
            )
        else:
            raise ValueError("Invalid parameter configuration source specified.")

        
    def create_ansatz(self, params: np.ndarray):

        ansatz = self.create_ansatz(
            system=self.system,
            noise=self.noise,
            parameters=params,
        )
        return ansatz

    def cost_function(self, params: np.ndarray) -> float:
        """Boilerplate step: Plug in your cost evaluation logic here."""
        # TODO: Implement state initialization, ansatz application, and expectation value calculation
        raise NotImplementedError(
            "Implement your cost function evaluation logic here."
        )

    def run_optimization(self, initial_params: np.ndarray) -> dict:
        """Boilerplate step: Plug in your optimization wrapper logic here."""
        # TODO: Implement optimization loop (e.g., scipy.optimize.minimize or custom optimizer)
        raise NotImplementedError("Implement your optimization loop here.")