import hashlib
import itertools
import random

import numpy as np

from . import powerset
from .typing import Optimization, SearchSpace


class Simulator():
    def __init__(
            self,
            program: str,
            dataset: str,
            command: str,
            search_space: SearchSpace):
        self.program = program
        self.dataset = dataset
        self.command = command
        self.search_space = search_space
        self.powerset = powerset.PowerSet(search_space)

        default_runtime = 1.0
        best_speedup = 1.4
        best_runtime_improvement = default_runtime - default_runtime / best_speedup
        coefficients = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2}

        module = f"{self.program}:{self.dataset}:{self.command}"
        seed = bytes.fromhex(hashlib.sha256(module.encode()).hexdigest())

        self.rng = random.Random(seed)
        self.simulation = {0: default_runtime}
        for degree, num_coefficients in coefficients.items():
            # Major coefficients that either improve or impair the program
            max_contribution = best_runtime_improvement / len(coefficients)
            max_coefficient = 4 * max_contribution / num_coefficients
            target_len = len(self.simulation) + num_coefficients
            while len(self.simulation) < target_len:
                flags_indices = self.rng.sample(
                    range(self.powerset.num_elements), degree)
                feature = 0
                for i in flags_indices:
                    feature += 1 << i
                if feature not in self.simulation:
                    self.simulation[feature] = self.rng.uniform(
                        -max_coefficient, max_coefficient)
            # Minor coefficients that improve the program, i.e., disabling them
            # is detrimental; We assume they improve runtime by x1.5 in total
            max_contribution = default_runtime * 0.5 / len(coefficients)
            max_coefficient = 2 * max_contribution / num_coefficients
            target_len = len(self.simulation) + num_coefficients
            while len(self.simulation) < target_len:
                flags_indices = self.rng.sample(
                    range(self.powerset.num_elements), degree)
                feature = 0
                for i in flags_indices:
                    feature += 1 << i
                if feature not in self.simulation:
                    self.simulation[feature] = self.rng.triangular(
                        0.0, max_coefficient)

    def evaluate(self, optimization: Optimization,
                 num_repeats: int = 1) -> float:
        subset = self.powerset.optimization_to_subset_(optimization)
        feature = subset_to_int_(subset)
        runtime = self.evaluate_in_feature_space_(feature)
        # Abuse the num_repeats parameter as a request for accurate
        # measurements
        #if num_repeats == 1:
        #    runtime += self.rng.gauss(0.0, 0.005)
        return runtime

    def evaluate_in_feature_space_(self, otpimization: int) -> float:
        runtime = 0.0
        for feature, coefficient in self.simulation.items():
            if feature & otpimization == feature:
                runtime += coefficient
        return runtime

    def min(self) -> float:
        beneficial_features = [
            feature for feature, coefficient in self.simulation.items()
            if coefficient < 0.0]
        features_powerset = powerset_(beneficial_features)
        min_runtime = float("inf")
        for feature_set in features_powerset:
            combo = 0
            for feature in feature_set:
                combo |= feature
            runtime = self.evaluate_in_feature_space_(combo)
            if min_runtime > runtime:
                min_runtime = runtime
        return min_runtime


def subset_to_int_(subset: np.ndarray) -> int:
    value = 0
    for i, bit in enumerate(subset):
        if bit:
            value += 1 << i
    return value


def powerset_(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1))
