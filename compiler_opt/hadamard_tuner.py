from typing import TextIO

import numpy as np
import ssftapprox
import ssftapprox.minimization
import swht

from . import base_tuner
from . import evaluator
from . import powerset
from .typing import Optimization, SearchSpace


class HadamardTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(search_space, evaluator, "HadamardTuner", default_optimization)
        self.powerset = powerset.PowerSet(self.search_space)

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:

        def evaluate(subset):
            return self.evaluator.evaluate(
                self.powerset.subset_to_optimization_(subset))

        if file is not None:
            file.write(f"{budget}\n")
        ret = swht.swht(
            signal=evaluate,
            cs_algorithm=swht.NAIVE,
            n=self.powerset.num_elements,
            K=self.powerset.num_elements,
            robust_iterations=10)
        est = ssftapprox.SparseWHTFunction(np.array(list(ret.keys())),
                                           np.array(list(ret.values())))
        min_subset = ssftapprox.minimization.minimize_wht(est)
        return self.powerset.subset_to_optimization_(min_subset)
