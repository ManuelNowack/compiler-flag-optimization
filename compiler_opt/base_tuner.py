from typing import TextIO

from . import evaluator
from .typing import Optimization, SearchSpace


class Tuner:
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            name: str,
            default_optimization: Optimization):
        self.search_space = search_space
        self.evaluator = evaluator
        self.name = name
        self.default_optimization = default_optimization

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        raise NotImplementedError

    def tune(self, budget: int, file: TextIO = None) -> None:
        self.best_optimization = self.find_best_optimization(budget, file)
        num_repeats = 10
        self.default_runtime = self.evaluator.evaluate(
            self.default_optimization, num_repeats)
        self.best_runtime = self.evaluator.evaluate(
            self.best_optimization, num_repeats)
