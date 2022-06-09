import benchmark
from tuner.flag_info import optimization_to_str
import statistics


class Evaluator():
    def __init__(self, path, num_repeats, search_space, dataset, command):
        self.path = path
        self.num_repeats = num_repeats
        self.search_space = search_space
        self.dataset = dataset
        self.command = command

    def evaluate(self, optimization, num_repeats=1):
        flags = optimization_to_str(optimization, self.search_space)
        dir = benchmark.compile(self.path, "-w " + flags, "-fopenmp", True)
        run_times = [benchmark.run(self.path, self.dataset, self.command, dir)
                     for _ in range(num_repeats)]
        return statistics.median(run_times)
