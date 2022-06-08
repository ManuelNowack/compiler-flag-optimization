from BOCS.BOCS import BOCS
from .base_tuner import Tuner
import numpy as np


class BOCSTuner(Tuner):
    def __init__(self, search_space, evaluator, default_optimization):
        super().__init__(search_space, evaluator, "BOCSTuner", default_optimization)
        self.binary_flags = []
        self.parametric_flags = []
        for flag, configs in self.search_space.items():
            if flag == "stdOptLv":
                continue
            assert len(configs) > 1
            if configs != (False, True):
                for val in configs:
                    self.parametric_flags.append((flag, val))
            else:
                self.binary_flags.append(flag)

    def subset_to_optimization_(self, subset: np.ndarray):
        optimization = {"stdOptLv": 3}
        subset_it = iter(subset)
        for flag, enabled in zip(self.binary_flags, subset_it):
            optimization[flag] = bool(enabled)
        for (flag, val), enabled in zip(self.parametric_flags, subset_it):
            if enabled:
                optimization[flag] = val
        return optimization

    def tune(self, budget, batch_size=1, file=None):
        inputs = {}
        inputs["n_vars"] = len(self.binary_flags) + len(self.parametric_flags)
        inputs["evalBudget"] = budget
        inputs["n_init"] = budget // 5
        inputs["lambda"] = 1e-4

        def objective_function(x):
            def evaluate(subset):
                return self.evaluator.evaluate(
                    self.subset_to_optimization_(subset))
            return np.apply_along_axis(evaluate, 1, x)

        inputs["model"] = objective_function
        inputs["penalty"] = lambda x: inputs["lambda"] * np.sum(x, axis=1)

        rng = np.random.default_rng()
        inputs["x_vals"] = rng.random(
            (inputs["n_init"], inputs["n_vars"])).round()
        inputs["y_vals"] = inputs["model"](inputs["x_vals"])

        (BOCS_SA_model, BOCS_SA_obj) = BOCS(inputs.copy(), 2, "SA", file)
        # (BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, "SDP-l1")

        best_subset = BOCS_SA_model[BOCS_SA_obj.argmin()]
        best_optimization = self.subset_to_optimization_(best_subset)
        best_perf = self.evaluator.evaluate(best_optimization, 10)

        return best_optimization, best_perf
