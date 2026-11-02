import os
import time
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
from qulacs import DensityMatrix, QuantumCircuit, QuantumState
from scipy.optimize import minimize

from .ansatz import create_noisy_ansatz, noiseless_ansatz
from .constraint import create_time_constraints_with_mingap, create_tf_fixed_constraint
from .createparam import create_param
from .hamiltonian import create_ising_hamiltonian, create_xy_hamiltonian  # Adjust based on your module imports


class VQE:
    def __init__(self, config, trial: int = 0, observable=None) -> None:
        self.cfg = config
        self.trial = trial
        self.observable = observable

        # System & Ansatz parameters
        self.nqubits = self.cfg.n_qubits
        self.n_layers = self.cfg.n_layers
        self.state_type = "dmatrix" if self.cfg.control_noise_enabled else "statevector" # Or configure via config
        
        # Optimization settings
        self.optimizer_method = self.cfg.optimization.get("method", "SLSQP")
        self.opt_options = self.cfg.optimization.get("options", {})
        self.maxiter = self.cfg.optimization.get("maxiter", 1000)
        self.tol = self.cfg.optimization.get("tol", 1e-6)

        # Counters & tracking
        self.cost_fn_calling_counter = 0
        self.iteration_count = 0

        # Setup initial parameters
        self.initial_params = self._initialize_parameters()

    def _initialize_parameters(self) -> np.ndarray:
        """Determines whether to use passed values or generate random parameters based on config logic."""
        source = self.cfg.parameter_source()
        
        if source == "values":
            return np.array(self.cfg.parameter_values, dtype=float)
        elif source == "random":
            # Set seed for reproducibility per trial
            seed = self.cfg.init_seed(self.trial)
            np.random.seed(seed)
            
            # Generate random parameters (adjust arguments to match your create_param signature)
            return create_param(
                layers=self.n_layers,
                gateset=self.cfg.control_unit_size,
                ti=0.0,
                tf=self.cfg.total_time
            )
        else:
            raise ValueError(f"Unknown parameter source: {source}")

    def create_ansatz(self, param: np.ndarray) -> QuantumCircuit:
        """Constructs the ansatz circuit (noiseless or noisy)."""
        if self.cfg.noise_strength and self.cfg.noise_strength > 0:
            ansatz_dict = create_noisy_ansatz(
                nqubits=self.nqubits,
                layers=self.n_layers,
                gateset=self.cfg.control_unit_size,
                ugateH=self.observable, # or drift Hamiltonian
                ansatz_noise_type=self.cfg.noise_type,
                ansatz_noise_prob=self.cfg.noise_strength,
                param=param,
                identity_factors=[0, 0, 0, 0] # Adjust as needed
            )
        else:
            ansatz_dict = noiseless_ansatz(
                nqubits=self.nqubits,
                layers=self.n_layers,
                gateset=self.cfg.control_unit_size,
                ugateH=self.observable,
                param=param
            )
        return ansatz_dict["circuit"]

    def cost_function(self, param: np.ndarray) -> float:
        """Evaluates the VQE cost function (expectation value)."""
        state = DensityMatrix(self.nqubits) if self.state_type.lower() == "dmatrix" else QuantumState(self.nqubits)
        
        circuit = self.create_ansatz(param=param)
        circuit.update_quantum_state(state)
        
        cost = self.observable.get_expectation_value(state)
        self.cost_fn_calling_counter += 1

        if self.cfg.verbose:
            print(f"[Trial {self.trial}] Cost Call #{self.cost_fn_calling_counter} | Cost: {cost:.6f}", flush=True)

        return float(cost)

    def run_optimization(self) -> Dict:
        """Runs the optimization loop using SciPy minimize."""
        cost_history = []

        def _tracked_cost(param):
            cost = self.cost_function(param)
            cost_history.append(cost)
            return cost

        def _iteration_callback(param):
            self.iteration_count += 1
            if self.cfg.verbose:
                print(f"[Trial {self.trial}] Iteration completed: {self.iteration_count}", flush=True)

        # Initial evaluation
        initial_cost = self.cost_function(self.initial_params)

        # Build constraints if monotonically increasing time is enabled
        constraints = None
        if self.cfg.monotonic_time and self.optimizer_method == "SLSQP":
            # Example constraint hook (adjust to your project's constraint builder functions)
            constraints = create_time_constraints_with_mingap(self.n_layers, len(self.initial_params))

        # Run SciPy optimization
        opt_result = minimize(
            _tracked_cost,
            self.initial_params,
            method=self.optimizer_method,
            constraints=constraints,
            callback=_iteration_callback,
            options=self.opt_options
        )

        optimized_cost = opt_result.fun
        optimized_params = opt_result.x.tolist()

        # Optional HDF5 logging if output directory is configured
        if self.cfg.output_dir is not None:
            self._save_h5_results(initial_cost, optimized_cost, cost_history, opt_result)

        return {
            "initial_cost": initial_cost,
            "min_cost": optimized_cost,
            "initial_params": self.initial_params.tolist(),
            "optimized_params": optimized_params,
            "cost_callings_total": self.cost_fn_calling_counter,
            "optimizer_result": opt_result
        }

    def _save_h5_results(self, init_cost: float, final_cost: float, cost_history: List[float], opt_result):
        """Helper to save trial history into HDF5 format."""
        h5_path = self.cfg.output_dir / f"trial_{self.trial}_results.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("initial_params", data=self.initial_params)
            hf.create_dataset("optimized_params", data=opt_result.x)
            hf.create_dataset("cost_history", data=np.array(cost_history))
            hf.attrs["initial_cost"] = init_cost
            hf.attrs["final_cost"] = final_cost
            hf.attrs["success"] = bool(opt_result.success)
            hf.attrs["message"] = str(opt_result.message)