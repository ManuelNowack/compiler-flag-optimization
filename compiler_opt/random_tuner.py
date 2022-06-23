import random
import time
from typing import TextIO

from . import base_tuner
from . import evaluator
from .typing import Optimization, SearchSpace


class RandomTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(search_space, evaluator, "RandomTuner", default_optimization)
        self.visited = set()

    def generate_candidates(self, batch_size: int = 1) -> list[Optimization]:
        random.seed(time.time())
        candidates = []
        for _ in range(batch_size):
            while True:
                optimization = {}
                for flag_name, domain in self.search_space.items():
                    num = len(domain)
                    rv = random.randint(0, num - 1)
                    optimization[flag_name] = domain[rv]

                # Avoid duplication
                if str(optimization) not in self.visited:
                    self.visited.add(str(optimization))
                    candidates.append(optimization)
                    break

        return candidates

    def evaluate_candidates(
            self, candidates: list[Optimization]) -> list[float]:
        return [self.evaluator.evaluate(optimization)
                for optimization in candidates]

    def reflect_feedback(self, perfs):
        # Random search. Do nothing
        pass

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        best_optimization, best_runtime = None, float("inf")
        i = 0
        while i < budget:
            candidates = self.generate_candidates()
            perfs = self.evaluate_candidates(candidates)

            i += len(candidates)
            for optimization, perf in zip(candidates, perfs):
                if perf < best_runtime:
                    best_runtime = perf
                    best_optimization = optimization

            print(best_runtime, file=file)
            self.reflect_feedback(perfs)
        return best_optimization
