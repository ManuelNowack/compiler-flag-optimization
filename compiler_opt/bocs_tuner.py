from typing import TextIO

import numpy as np
from BOCS.BOCS import BOCS

from . import base_tuner
from . import evaluator
from . import powerset
from .typing import Optimization, SearchSpace


class BOCSTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization,
            acquisition_function: str):
        super().__init__(
            search_space,
            evaluator,
            f"BOCS-{acquisition_function}",
            default_optimization)
        self.acquisition_function = acquisition_function
        self.powerset = powerset.PowerSet(self.search_space)

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        inputs = {}
        inputs["n_vars"] = self.powerset.num_elements
        inputs["evalBudget"] = budget
        inputs["n_init"] = budget // 5
        inputs["lambda"] = 1e-4

        def objective_function(x):
            def evaluate(subset):
                return self.evaluator.evaluate(
                    self.powerset.subset_to_optimization_(subset))
            return np.apply_along_axis(evaluate, 1, x)

        inputs["model"] = objective_function
        inputs["penalty"] = lambda x: inputs["lambda"] * np.sum(x, axis=1)

        rng = np.random.default_rng()
        inputs["x_vals"] = rng.random(
            (inputs["n_init"], inputs["n_vars"])).round()
        inputs["y_vals"] = inputs["model"](inputs["x_vals"])

        BOCS_model, BOCS_obj = BOCS(
            inputs.copy(), 2, self.acquisition_function, file)
        best_subset = BOCS_model[BOCS_obj.argmin()]

        return self.powerset.subset_to_optimization_(best_subset)


class BOCSSATuner(BOCSTuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(
            search_space,
            evaluator,
            default_optimization,
            "SA")


class BOCSSDPTuner(BOCSTuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(
            search_space,
            evaluator,
            default_optimization,
            "SDP-l1")
