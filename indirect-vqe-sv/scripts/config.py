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
        self.system = get(raw, "system", default={})
        self.noise = get(raw, "noise", default={})

        self.task = get(raw, "run", "task")
        self.verbosity = get(raw, "run", "verbosity", default=1)
       # config.py, in Config.__init__
        self.output = get(raw, "output", default={})       # the dict, like system / noise

        out = get(self.output, "output_dir",
                default=get(raw, "paths", "output_dir",
                            default=get(raw, "run", "output_dir", default="../results")))
        self.output_dir = Path(out) if out is not None else None

        self.output_prefix = get(self.output, "output_prefix")
        self.hdf5_enabled = get(self.output, "hdf5_storing", "enabled", default=False)
        
        self.target_observable_type = get(raw, "target_observable", "type")
        self.target_observable_coefficients = get(raw, "target_observable", "coefficients", default={})

        self.n_qubits = get(self.system, "n_qubits")
        self.n_layers = get(self.system, "n_layers")
        self.drift = get(self.system, "drift")
        time_eval = get(self.system, "time_evolution", default={})
        self.total_time = get(time_eval, "total_time",
                              default=get(self.system, "time", "total_time"))

        self.parameter_values = get(raw, "parameters", "values")
        rand = get(raw, "parameters", "random_initialization")
        self.random_init = (rand if isinstance(rand, dict)
                            else {"enabled": rand} if isinstance(rand, bool)
                            else {"enabled": True})

        # New location first, legacy location as fallback
        cons = get(raw, "parameters", "constraints", default={})
        self.monotonically_increasing_time = get(
            cons, "monotonically_increasing_time",
            default=get(raw, "parameters", "monotonically_increasing_time", default=True))
        self.fixed_final_time = get(
            cons, "fixed_final_time",
            default=get(raw, "parameters", "fixed_final_time", default=False))

        self.optimization = get(raw, "optimization", default={})

    def parameter_source(self):
        if self.parameter_values is not None:
            return "values"
        if not self.random_init.get("enabled", True):
            raise ValueError("parameters.values is empty, so parameters.random_initialization is needed")
        return "random"