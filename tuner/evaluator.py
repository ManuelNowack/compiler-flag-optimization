import benchmark
from .flag_info import optimization_to_str
import statistics
from .types import Optimization, SearchSpace


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
        flags = optimization_to_str(optimization, self.search_space)
        dir = benchmark.compile(self.program, "-w " + flags, "-fopenmp", True)
        run_times = [benchmark.run(self.program, self.dataset, self.command, dir)
                     for _ in range(num_repeats)]
        return statistics.median(run_times)
