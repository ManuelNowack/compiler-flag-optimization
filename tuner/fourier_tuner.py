from .base_tuner import Tuner
from .evaluator import Evaluator
import numpy as np
from ssftapprox import ElasticNetEstimator
from ssftapprox.minimization import minimize_dsft3
import time
from .types import Optimization, SearchSpace
from typing import TextIO


class FourierTuner(Tuner):
    def __init__(self, search_space: SearchSpace,
                 evaluator: Evaluator, default_optimization: Optimization):
        super().__init__(search_space, evaluator, "FourierTuner", default_optimization)
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

    def subset_to_optimization_(self, subset: np.ndarray) -> Optimization:
        optimization = {"stdOptLv": 3}
        subset_it = iter(subset)
        for flag, enabled in zip(self.binary_flags, subset_it):
            optimization[flag] = bool(enabled)
        for (flag, val), enabled in zip(self.parametric_flags, subset_it):
            if enabled:
                optimization[flag] = val
        return optimization

    def tune(self, budget: int, batch_size: int = 1,
             file: TextIO = None) -> tuple[Optimization, float]:
        n = len(self.binary_flags) + len(self.parametric_flags)
        rng = np.random.default_rng()
        X_train = rng.random((budget, n)).round()

        def evaluate(subset):
            return self.evaluator.evaluate(
                self.subset_to_optimization_(subset))
        Y_train = np.apply_along_axis(evaluate, axis=1, arr=X_train)

        start = time.perf_counter()
        est = ElasticNetEstimator(enet_alpha=0.00001, standardize=False)
        est.fit(X_train, Y_train)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Fit duration: {end - start} s\n")

        start = time.perf_counter()
        argmin, minval = minimize_dsft3(
            est.est, cardinality_constraint=lambda x: x == 7, C=1000)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Minimize duration: {end - start} s\n")
        best_optimization = self.subset_to_optimization_(argmin)
        best_perf = self.evaluator.evaluate(best_optimization, 10)
        return best_optimization, best_perf
