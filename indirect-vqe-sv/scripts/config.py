from pathlib import Path


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


class Config:
    def __init__(self, raw: dict):
        self.raw = raw
        # Support both {"config": {...}} and legacy top-level keys
        cfg = raw["config"] if isinstance(raw.get("config"), dict) else raw

        self.system = get(cfg, "system", default={})
        self.noise = get(cfg, "noise", default={})

        self.task = get(cfg, "run", "task", default=get(raw, "run", "task"))
        self.output = get(cfg, "output", default={})

        out = get(
            self.output,
            "output_dir",
            default=get(
                cfg,
                "paths",
                "output_dir",
                default=get(
                    cfg,
                    "run",
                    "output_dir",
                    default=get(raw, "run", "output_dir", default="../results"),
                ),
            ),
        )
        self.output_dir = Path(out) if out is not None else None

        self.output_prefix = get(self.output, "output_prefix")
        self.draw_cricuit_enabled = get(self.output, "draw_circut_layer", "enabled", default=False)
        self.draw_cricuit_type = get(self.output, "draw_circut_layer", "type", default="png")
        self.draw_cricuit_dpi = get(self.output, "draw_circut_layer", "dpi", default=300)
        self.export_circut_json = get(self.output, "export_circuit_json", default=False)
        # self.export_optimizer_artifact = get(self.output, "export_optimizer_artifact", default=False)
        self.export_initial_state = get(self.output, "expot_quantum_states", "initial", default=False)
        self.export_final_state = get(self.output, "expot_quantum_states", "final", default=False)

        self.verbosity = get(self.output, "verbosity", default=1)
        self.hdf5_enabled = get(self.output, "hdf5_storing", "enabled", default=False)

        self.target_observable_type = get(cfg, "target_observable", "type")
        self.target_observable_coefficients = get(cfg, "target_observable", "coefficients", default={})

        self.n_qubits = get(self.system, "n_qubits")
        self.n_layers = get(self.system, "n_layers")
        self.drift = get(self.system, "drift")
        time_eval = get(self.system, "time_evolution", default={})
        self.total_time = get(time_eval, "total_time", default=get(self.system, "time", "total_time"))

        self.parameter_values = get(cfg, "parameters", "values")
        rand = get(cfg, "parameters", "random_initialization")
        self.random_init = (
            rand if isinstance(rand, dict) else {"enabled": rand} if isinstance(rand, bool) else {"enabled": True}
        )

        # New location first, legacy location as fallback
        cons = get(cfg, "parameters", "constraints", default={})
        self.monotonically_increasing_time = get(
            cons,
            "monotonically_increasing_time",
            default=get(cfg, "parameters", "monotonically_increasing_time", default=True),
        )
        self.fixed_final_time = get(
            cons,
            "fixed_final_time",
            default=get(cfg, "parameters", "fixed_final_time", default=False),
        )

        self.optimization = get(cfg, "optimization", default={})

    def parameter_source(self):
        if self.parameter_values is not None:
            return "values"
        if not self.random_init.get("enabled", True):
            raise ValueError("parameters.values is empty, so parameters.random_initialization is needed")
        return "random"