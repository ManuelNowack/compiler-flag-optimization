import time
from typing import TextIO

import numpy as np
import ssftapprox
import ssftapprox.minimization

from . import base_tuner
from . import evaluator
from .typing import Optimization, SearchSpace


class FourierTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
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

    def tune(self, budget: int,
             file: TextIO = None) -> tuple[Optimization, float]:
        n = len(self.binary_flags) + len(self.parametric_flags)
        rng = np.random.default_rng()
        X_train = rng.random((budget, n)).round()

        def evaluate(subset):
            return self.evaluator.evaluate(
                self.subset_to_optimization_(subset))
        Y_train = np.apply_along_axis(evaluate, axis=1, arr=X_train)

        if file is not None:
            file.write(f"best train runtime: {Y_train.min():.3e} s\n")

        start = time.perf_counter()
        est = ssftapprox.ElasticNetEstimator(enet_alpha=1e-5, standardize=True)
        if file is not None:
            file.write(f"Alpha: {est.enet_alpha}\n")
        est.fit(X_train, Y_train)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Num coefs: {len(est.est.coefs)}\n")
            file.write(f"Fit duration: {end - start} s\n")
        start = time.perf_counter()
        argmin, minval = ssftapprox.minimization.minimize_dsft3(est.est)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Minimize duration: {end - start} s\n")
        best_optimization = self.subset_to_optimization_(argmin)
        best_perf = self.evaluator.evaluate(best_optimization, 10)
        return best_optimization, best_perf
