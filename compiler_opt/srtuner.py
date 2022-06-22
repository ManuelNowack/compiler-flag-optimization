from SRTuner import SRTunerModule

from . import base_tuner


class SRTuner(base_tuner.Tuner):
    def __init__(self, search_space, evaluator, default_optimization):
        super().__init__(search_space, evaluator, "SRTuner", default_optimization)
        self.visited = set()

        # User can customize reward func as Python function and pass to module.
        # In this demo, we use the default reward func.
        self.mod = SRTunerModule(convert_search_space(search_space))

    def generate_candidates(self, batch_size=1):
        return self.mod.generate_candidates(batch_size)

    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(optimization)
                for optimization in candidates]

    def reflect_feedback(self, perfs):
        self.mod.reflect_feedback(perfs)

    def find_best_optimization(self, budget, file=None):
        best_optimization, best_runtime = None, float("inf")
        i = 0
        while i < budget:
            candidates = self.generate_candidates()
            perfs = self.evaluate_candidates(candidates)

            i += len(candidates)
            for optimization, perf in zip(candidates, perfs):
                if perf < best_runtime:
                    best_runtime = perf
                    best_optimization = optimization

            print(best_runtime, file=file)
            self.reflect_feedback(perfs)
        return best_optimization


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
        isParametric = configs != (False, True)
        search_space_new[flag_name] = GCCFlagInfo(
            flag_name, configs, isParametric, None)
    return search_space_new
