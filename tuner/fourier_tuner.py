from .base_tuner import Tuner
import numpy as np
from ssftapprox import ElasticNetEstimator
from ssftapprox.minimization import minimize_dsft3
import time


class FourierTuner(Tuner):
    def __init__(self, search_space, evaluator, default_setting):
        super().__init__(search_space, evaluator, "FourierTuner", default_setting)
        self.binary_flags = []
        self.parametric_flags = []
        for flag, configs in self.search_space.items():
            if flag == "stdOptLv":
                continue
            assert len(configs) > 1
            if configs != [False, True]:
                for val in configs:
                    self.parametric_flags.append((flag, val))
            else:
                self.binary_flags.append(flag)

    def subset_to_opt_setting_(self, subset: np.ndarray):
        opt_setting = {"stdOptLv": 3}
        subset_it = iter(subset)
        for flag, enabled in zip(self.binary_flags, subset_it):
            opt_setting[flag] = bool(enabled)
        for (flag, val), enabled in zip(self.parametric_flags, subset_it):
            if enabled:
                opt_setting[flag] = val
        return opt_setting

    def tune(self, budget, batch_size=1, file=None):
        n = len(self.binary_flags) + len(self.parametric_flags)
        rng = np.random.default_rng()
        X_train = rng.random((budget, n)).round()

        def evaluate(subset):
            return self.evaluator.evaluate(
                self.subset_to_opt_setting_(subset))
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
        best_opt_setting = self.subset_to_opt_setting_(argmin)
        best_perf = self.evaluator.evaluate(best_opt_setting, 10)
        return best_opt_setting, best_perf
