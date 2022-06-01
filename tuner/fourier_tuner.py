from .base_tuner import Tuner
import numpy as np
from ssftapprox import ElasticNetEstimator
from ssftapprox.minimization import minimize_dsft3


class FourierTuner(Tuner):
    def __init__(self, search_space, evaluator, default_setting):
        super().__init__(search_space, evaluator, "FourierTuner", default_setting)
        self.binary_flags = []
        self.parametric_flags = []
        for flag, info in self.search_space.items():
            if flag == "stdOptLv":
                continue
            if info.isParametric:
                for val in info.configs:
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

        est = ElasticNetEstimator(enet_alpha=0.00001, standardize=False)
        est.fit(X_train, Y_train)

        argmin, minval = minimize_dsft3(
            est.est, cardinality_constraint=lambda x: x == 7, C=1000)
        best_opt_setting = self.subset_to_opt_setting_(argmin)
        best_perf = self.evaluator.evaluate(best_opt_setting)
        return best_opt_setting, best_perf
