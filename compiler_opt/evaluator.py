import shutil
import statistics

from . import benchmark
from . import flag_info
from .typing import Optimization, SearchSpace


class Evaluator():
    def __init__(self, program: str, num_repeats: int,
                 search_space: SearchSpace, dataset: str, command: str):
        self.program = program
        self.search_space = search_space
        self.dataset = dataset
        self.command = command
        self.repeat = benchmark.get_repeat(program, dataset, command, "-w -O3")

    def evaluate(self, optimization: Optimization,
                 num_repeats: int = 1) -> float:
        flags = flag_info.optimization_to_str(optimization, self.search_space)
        tmp_dir = benchmark.compile(self.program, "-w " + flags, True)
        run_times = [
            benchmark.run(
                self.program,
                self.dataset,
                self.command,
                tmp_dir,
                self.repeat) for _ in range(num_repeats)]
        shutil.rmtree(tmp_dir)
        return statistics.median(run_times)
