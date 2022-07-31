import time
from typing import TextIO

import numpy as np
import pandas as pd
import ssftapprox
import ssftapprox.minimization

from . import base_tuner
from . import evaluator
from . import flag_info
from . import powerset
from . import simulator
from .typing import Optimization, SearchSpace


class FourierTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(search_space, evaluator, "FourierTuner", default_optimization)
        self.powerset = powerset.PowerSet(self.search_space)

    def str_to_subset_(self, flags: str) -> np.ndarray:
        optimization = flag_info.str_to_optimization(flags, self.search_space)
        return self.powerset.optimization_to_subset_(optimization)

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        if isinstance(self.evaluator, simulator.Simulator):
            rng = np.random.default_rng(42)
            x = rng.random((10000, self.powerset.num_elements)).round()

            def evaluate(subset):
                return self.evaluator.evaluate(
                    self.powerset.subset_to_optimization_(subset))
            y = np.apply_along_axis(evaluate, axis=1, arr=x)
            x_train = x[:budget]
            y_train = y[:budget]
        else:
            x, y = self.load_training_data("samples/10000.csv")
            rng = np.random.default_rng()
            train_indices = rng.choice(len(x), size=budget, replace=False)
            x_train = x[train_indices]
            y_train = y[train_indices]
            assert np.all(y_train)

        self.train_runtime = y_train.min()

        start = time.perf_counter()
        est = ssftapprox.ElasticNetEstimator(
            enet_alpha=1e-5, n_threads=1, standardize=True)
        if file is not None:
            file.write(f"Alpha: {est.enet_alpha}\n")
        est.fit(x_train, y_train)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Num coefs: {len(est.est.coefs)}\n")
            file.write(f"Fit duration: {end - start} s\n")
            file.write(f"Train score: {est.score(x_train, y_train)}\n")
            file.write(f"Validate score: {est.score(x, y)}\n")
        start = time.perf_counter()
        argmin, minval = ssftapprox.minimization.minimize_dsft3(est.est)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Minimize duration: {end - start} s\n")
        return self.powerset.subset_to_optimization_(argmin)

    def load_training_data(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path, index_col=0)
        x = np.array([self.str_to_subset_(flags) for flags in df.index])
        module = (f"{self.evaluator.program}:{self.evaluator.dataset}"
                  f":{self.evaluator.command}")
        y = df[module].to_numpy()
        return x, y
