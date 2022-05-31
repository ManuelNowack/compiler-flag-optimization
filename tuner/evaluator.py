import benchmark
from tuner.flag_info import convert_to_str
import statistics

class Evaluator():
    def __init__(self, path, num_repeats, search_space, dataset, command):
        self.path = path
        self.num_repeats = num_repeats
        self.search_space = search_space
        self.dataset = dataset
        self.command = command

    def evaluate(self, opt_setting, num_repeats=1):
        flags = convert_to_str(opt_setting, self.search_space)
        benchmark.compile(self.path, flags, "-fopenmp")
        run_times = [benchmark.run(self.path, self.dataset, self.command)
                     for _ in range(num_repeats)]
        return statistics.median(run_times)
