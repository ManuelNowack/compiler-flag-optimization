from .base_tuner import Tuner
from .evaluator import Evaluator
import random
import time
from .types import Optimization, SearchSpace
from typing import TextIO


class RandomTuner(Tuner):
    def __init__(self, search_space: SearchSpace,
                 evaluator: Evaluator, default_optimization: Optimization):
        super().__init__(search_space, evaluator, "RandomTuner", default_optimization)
        self.visited = set()

    def generate_candidates(self, batch_size: int = 1) -> list[Optimization]:
        random.seed(time.time())
        candidates = []
        for _ in range(batch_size):
            while True:
                optimization = {}
                for flag_name, configs in self.search_space.items():
                    num = len(configs)
                    rv = random.randint(0, num - 1)
                    optimization[flag_name] = configs[rv]

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

    def tune(self, budget: int, batch_size: int = 1,
             file: TextIO = None) -> tuple[Optimization, float]:
        best_optimization, best_perf = None, float("inf")
        i = 0
        while i < budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)

            i += len(candidates)
            for optimization, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_optimization = optimization

            print(best_perf, file=file)
            self.reflect_feedback(perfs)
        best_perf = self.evaluator.evaluate(best_optimization, 10)
        return best_optimization, best_perf
