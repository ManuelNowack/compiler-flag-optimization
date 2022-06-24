import operator
from typing import TextIO

from . import base_tuner
from . import evaluator
from . import flag_info
from .typing import Optimization, SearchSpace


class MonoTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(search_space, evaluator, "MonoTuner", default_optimization)

    def mutate_optimization_by_individual_flags(
            self, optimization: Optimization) -> list[tuple[Optimization, float]]:
        base_flags = flag_info.read_gcc_flags(
            self.evaluator.program, flag_info.optimization_to_str(
                optimization, self.search_space))
        mutated_optimizations = []
        for flag_name, domain in self.search_space.items():
            if flag_name == "stdOptLv":
                continue
            for value in domain:
                if optimization.get(flag_name) == value:
                    continue
                elif value is True:
                    if flag_name in base_flags:
                        continue
                elif value is False:
                    if flag_name not in base_flags:
                        continue
                else:
                    flag = f"{flag_name}={value}"
                    if flag in base_flags:
                        continue
                new_optimization = optimization.copy()
                new_optimization[flag_name] = value
                run_time = self.evaluator.evaluate(new_optimization)
                mutated_optimizations.append((new_optimization, run_time))
        mutated_optimizations.sort(key=operator.itemgetter(1), reverse=True)
        return mutated_optimizations

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        best_optimization = self.default_optimization
        best_runtime = self.evaluator.evaluate(best_optimization)
        while budget > 0:
            candidates = self.mutate_optimization_by_individual_flags(
                best_optimization)
            budget -= len(candidates)
            if budget < 0:
                del candidates[budget:]
            next_optimization = best_optimization.copy()
            for optimization, run_time in candidates[:1]:
                if run_time > best_runtime:
                    curr_optimization_set = set(optimization.items())
                    best_optimization_set = set(best_optimization.items())
                    new_flags = curr_optimization_set - best_optimization_set
                    removed_flags = best_optimization_set - curr_optimization_set
                    (flag_name, value), = new_flags
                    if flag_name not in next_optimization:
                        next_optimization[flag_name] = value
                        assert not removed_flags
                    else:
                        (flag_name2, old_value), = removed_flags
                        assert flag_name == flag_name2
                        if next_optimization[flag_name] == old_value:
                            next_optimization[flag_name] = value
            best_optimization = next_optimization
        return best_optimization
