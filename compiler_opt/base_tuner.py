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
        self.default_perf = evaluator.evaluate(default_optimization, 10)

    def tune(self, budget: int,
             file: TextIO = None) -> tuple[Optimization, float]:
        raise NotImplementedError