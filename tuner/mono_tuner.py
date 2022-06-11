from .base_tuner import Tuner
from .flag_info import optimization_to_str, read_gcc_flags


class MonoTuner(Tuner):
    def __init__(self, search_space, evaluator, default_optimization):
        super().__init__(search_space, evaluator, "MonoTuner", default_optimization)
        self.default_flags = read_gcc_flags(
            self.evaluator.path,
            optimization_to_str(self.default_optimization, self.search_space))

    def search_space_size(self):
        size = 0
        for flag, configs in self.search_space.items():
            if flag == "stdOptLv":
                continue
            size += len(configs)
        return size

    def tune(self, budget, batch_size=1, file=None):
        repeats = budget // self.search_space_size()
        assert repeats > 0

        best_optimization = self.default_optimization
        best_perf = self.default_perf
        for flag_name, configs in self.search_space.items():
            if flag_name == "stdOptLv":
                continue
            for config in configs:
                if config is True:
                    if flag_name in self.default_flags:
                        continue
                elif config is False:
                    if flag_name not in self.default_flags:
                        continue
                else:
                    flag = f"{flag_name}={config}"
                    if flag in self.default_flags:
                        continue
                curr_optimization = self.default_optimization.copy()
                curr_optimization[flag_name] = config
                curr_perf = self.evaluator.evaluate(curr_optimization, repeats)
                if file is not None:
                    speedup = self.default_perf / curr_perf
                    file.write(f"{flag_name}: {speedup:.3f}\n")
                if best_perf > curr_perf:
                    best_perf = curr_perf
                    best_optimization = curr_optimization
        return best_optimization, best_perf
