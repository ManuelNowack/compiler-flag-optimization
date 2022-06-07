from SRTuner import SRTunerModule
from .base_tuner import Tuner

# SRTuner as a standalone tuner
class SRTuner(Tuner):
    def __init__(self, search_space, evaluator, default_setting):
        super().__init__(search_space, evaluator, "SRTuner", default_setting)
        self.visited = set()

        # User can customize reward func as Python function and pass to module.
        # In this demo, we use the default reward func. 
        self.mod = SRTunerModule(convert_search_space(search_space), default_perf = self.default_perf)

    def generate_candidates(self, batch_size=1):
        return self.mod.generate_candidates(batch_size)

    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(opt_setting) for opt_setting in candidates]

    def reflect_feedback(self, perfs):
        self.mod.reflect_feedback(perfs)

    def tune(self, budget, batch_size=1, file=None):
        best_opt_setting, best_perf = None, float("inf")
        i = 0
        while i<budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)
        
            i += len(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            print(best_perf, file=file)
            self.reflect_feedback(perfs)
        best_perf = self.evaluator.evaluate(best_opt_setting, 10)
        return best_opt_setting, best_perf


class FlagInfo:
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs


class GCCFlagInfo(FlagInfo):
    def __init__(self, name, configs, isParametric, stdOptLv):
        super().__init__(name, configs)
        self.isParametric = isParametric
        self.stdOptLv = stdOptLv


def convert_search_space(search_space):
    search_space_new = dict()
    for flag_name, configs in search_space.items():
        isParametric = configs != [False, True]
        search_space_new[flag_name] = GCCFlagInfo(flag_name, configs, isParametric, None)
    return search_space_new
