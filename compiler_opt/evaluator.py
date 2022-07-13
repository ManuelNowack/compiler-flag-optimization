import shutil
import statistics

from . import benchmark
from . import flag_info
from .typing import Optimization, SearchSpace


class Evaluator():
    def __init__(self, program: str, num_repeats: int,
                 search_space: SearchSpace, dataset: str, command: str):
        self.program = program
        self.num_repeats = num_repeats
        self.search_space = search_space
        self.dataset = dataset
        self.command = command

    def evaluate(self, optimization: Optimization,
                 num_repeats: int = 1) -> float:
        flags = flag_info.optimization_to_str(optimization, self.search_space)
        dir = benchmark.compile(self.program, "-w " + flags, True)
        run_times = [benchmark.run(self.program, self.dataset, self.command, dir)
                     for _ in range(num_repeats)]
        shutil.rmtree(dir)
        return statistics.median(run_times)
