import time
from typing import TextIO

import numpy as np
import pandas as pd
import ssftapprox
import ssftapprox.minimization

from . import base_tuner
from . import evaluator
from . import flag_info
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

    def optimization_to_subset_(
            self, optimization: Optimization) -> np.ndarray:
        n = len(self.binary_flags) + len(self.parametric_flags)
        # TODO: Remove empty configs from input file
        # assert optimization.keys() == self.search_space.keys()
        subset = np.zeros(n)
        for flag_name, config in optimization.items():
            if flag_name == "stdOptLv":
                # assert config == 3
                continue
            elif config is True:
                index = self.binary_flags.index(flag_name)
                subset[index] = 1.0
            elif config is False:
                pass  # Already initialized to zero
            else:
                index = self.parametric_flags.index((flag_name, config))
                subset[len(self.binary_flags) + index] = 1.0
        return subset

    def str_to_subset_(self, flags: str) -> np.ndarray:
        optimization = flag_info.str_to_optimization(flags, self.search_space)
        return self.optimization_to_subset_(optimization)

    def tune(self, budget: int,
             file: TextIO = None) -> tuple[Optimization, float]:
        if False:
            n = len(self.binary_flags) + len(self.parametric_flags)
            rng = np.random.default_rng()
            X_train = rng.random((budget, n)).round()

            def evaluate(subset):
                return self.evaluator.evaluate(
                    self.subset_to_optimization_(subset))
            Y_train = np.apply_along_axis(evaluate, axis=1, arr=X_train)
        else:
            x, y = self.load_training_data("samples/5000.csv")
            rng = np.random.default_rng()
            train_indices = rng.choice(len(x), size=budget, replace=False)
            X_train = x[train_indices]
            Y_train = y[train_indices]
            assert np.all(Y_train)

        self.best_perf_train = Y_train.min()

        start = time.perf_counter()
        est = ssftapprox.ElasticNetEstimator(enet_alpha=1e-5, standardize=True)
        if file is not None:
            file.write(f"Alpha: {est.enet_alpha}\n")
        est.fit(X_train, Y_train)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Num coefs: {len(est.est.coefs)}\n")
            file.write(f"Fit duration: {end - start} s\n")
            file.write(f"Train score: {est.score(X_train, Y_train)}\n")
            file.write(f"Validate score: {est.score(x, y)}\n")
        start = time.perf_counter()
        argmin, minval = ssftapprox.minimization.minimize_dsft3(est.est)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Minimize duration: {end - start} s\n")
        best_optimization = self.subset_to_optimization_(argmin)
        best_perf = self.evaluator.evaluate(best_optimization, 10)
        return best_optimization, best_perf

    def load_training_data(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path, index_col=0)
        x = np.array([self.str_to_subset_(flags) for flags in df.index])
        if self.evaluator.command == "":
            program_name = self.evaluator.program
        else:
            program_name = f"{self.evaluator.program}-{self.evaluator.command}"
        y = df[program_name].to_numpy()
        return x, y
