import time
from typing import TextIO, Union

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


class FourierTunerBase(base_tuner.Tuner):
    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: evaluator.Evaluator,
                 default_optimization: Optimization,
                 estimator: Union[ssftapprox.ElasticNetEstimator,
                                  ssftapprox.LowDegreeEstimator],
                 train_fraction: float = 1.0):
        if isinstance(estimator, ssftapprox.ElasticNetEstimator):
            name = "Fourier"
        elif isinstance(estimator, ssftapprox.LowDegreeEstimator):
            name = "LowDegree"
        else:
            raise ValueError("Unsupported estimator")
        if train_fraction < 1.0:
            name = "Active" + name
        super().__init__(search_space, evaluator, name, default_optimization)
        self.estimator = estimator
        self.train_fraction = train_fraction
        self.powerset = powerset.PowerSet(self.search_space)

    def str_to_subset_(self, flags: str) -> np.ndarray:
        optimization = flag_info.str_to_optimization(flags, self.search_space)
        return self.powerset.optimization_to_subset_(optimization)

    def evaluate_subset_(self, subset):
        return self.evaluator.evaluate(
            self.powerset.subset_to_optimization_(subset))

    def fit_and_minimize_(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_validate: np.ndarray,
            y_validate: np.ndarray,
            file: TextIO = None) -> np.ndarray:
        start = time.perf_counter()
        self.estimator.fit(x_train, y_train)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Num coefs: {len(self.estimator.est.coefs)}\n")
            file.write(f"Fit duration: {end - start} s\n")
            train_score = self.estimator.score(x_train, y_train)
            self.validate_score = self.estimator.score(x_validate, y_validate)
            file.write(f"Train score: {train_score}\n")
            file.write(f"Validate score: {self.validate_score}\n")

        if isinstance(self.estimator.est, ssftapprox.SparseDSFT3Function):
            minimize_function = ssftapprox.minimization.minimize_dsft3
        elif isinstance(self.estimator.est, ssftapprox.SparseWHTFunction):
            minimize_function = ssftapprox.minimization.minimize_wht
        else:
            raise ValueError("Unsupported estimator")
        start = time.perf_counter()
        min_feature, _ = minimize_function(self.estimator.est)
        end = time.perf_counter()
        if file is not None:
            file.write(f"Minimize duration: {end - start} s\n")
        return min_feature

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        initial_budget = int(budget * self.train_fraction)
        if isinstance(self.evaluator, simulator.Simulator):
            rng = np.random.default_rng(42)
            x = rng.random((10000, self.powerset.num_elements)).round()
            y = np.apply_along_axis(self.evaluate_subset_, axis=1, arr=x)
            x_train = x[:initial_budget]
            y_train = y[:initial_budget]
        else:
            samples_path = f"samples/10000_{len(self.search_space) - 1}.csv"
            x, y = self.load_training_data_(samples_path)
            rng = np.random.default_rng()
            train_indices = rng.choice(
                len(x), size=initial_budget, replace=False)
            x_train = x[train_indices]
            y_train = y[train_indices]
            assert np.all(y_train)

        if file is not None:
            file.write(f"Alpha: {self.estimator.enet_alpha}\n")
        if initial_budget == budget:
            self.train_runtime = y_train.min()
            if file is not None:
                file.write("\n")
            min_feature = self.fit_and_minimize_(x_train, y_train, x, y, file)
            return self.powerset.subset_to_optimization_(min_feature)
        for i in range(initial_budget, budget):
            if file is not None:
                file.write("\n")
                file.write(f"Query {i + 1}\n")
            min_feature = self.fit_and_minimize_(x_train, y_train, x, y, file)
            next_feature = min_feature.astype(x_train.dtype)
            for prev_feature in x_train:
                if np.array_equal(prev_feature, next_feature):
                    next_feature = rng.random(
                        self.powerset.num_elements).round()
                    if file is not None:
                        file.write(f"Take random feature\n")
                    break
            x_train = np.vstack((x_train, next_feature))
            y_train = np.append(y_train, self.evaluate_subset_(next_feature))
        return self.powerset.subset_to_optimization_(x_train[y_train.argmin()])

    def load_training_data_(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path, index_col=0)
        x = np.array([self.str_to_subset_(flags) for flags in df.index])
        module = (f"{self.evaluator.program}:{self.evaluator.dataset}"
                  f":{self.evaluator.command}")
        y = df[module].to_numpy()
        return x, y


class ActiveFourierTuner(FourierTunerBase):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(
            search_space,
            evaluator,
            default_optimization,
            ssftapprox.ElasticNetEstimator(
                enet_alpha=1e-2,
                n_threads=1,
                standardize=True),
            0.2)


class ActiveLowDegreeTuner(FourierTunerBase):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(
            search_space,
            evaluator,
            default_optimization,
            ssftapprox.LowDegreeEstimator(
                enet_alpha=1e-2,
                n_threads=1,
                standardize=True),
            0.2)


class FourierTuner(FourierTunerBase):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(
            search_space,
            evaluator,
            default_optimization,
            ssftapprox.ElasticNetEstimator(
                enet_alpha=1e-2,
                n_threads=1,
                standardize=True))


class LowDegreeTuner(FourierTunerBase):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(
            search_space,
            evaluator,
            default_optimization,
            ssftapprox.LowDegreeEstimator(
                enet_alpha=1e-2,
                n_threads=1,
                standardize=True))
