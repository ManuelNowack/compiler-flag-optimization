import multiprocessing
import os
import tempfile

import pandas as pd

from . import base_tuner
from . import flag_info
from .typing import SearchSpace


class Experiment():
    def __init__(
            self,
            modules: list[str],
            tuner_types: list[type],
            search_space: SearchSpace,
            budget: int,
            evaluator_type: type,
            parallel: int = None):
        self.modules = modules
        self.tuner_types = tuner_types
        self.search_space = search_space
        self.budget = budget
        self.evaluator_type = evaluator_type
        self.parallel = parallel
        self.default_optimization = {"stdOptLv": 3}
        for i in range(100):
            try:
                file_name = (f"results/n_{len(self.search_space) - 1:03d}_"
                             f"budget_{self.budget:04d}_{i:02d}")
                with open(file_name, "x"):
                    self.base_path_ = file_name
                    break
            except Exception:
                pass
        else:
            raise ValueError("No nonce available")
        self.run_()
        self.write_results_()

    def latch_arrive(self, module: str) -> None:
        with open(os.path.join(self.sync_dir, module), "x"):
            pass

    def latch_finished(self) -> None:
        for module in self.modules:
            if not os.path.isfile(os.path.join(self.sync_dir, module)):
                return False
        return True

    def tuning_thread_(self, module: str) -> list[base_tuner.Tuner]:
        program, dataset, command = module.split(":")
        evaluator = self.evaluator_type(
            program, dataset, command, self.search_space)
        tuners: list[base_tuner.Tuner] = [
            tuner_type(
                self.search_space,
                evaluator,
                self.default_optimization) for tuner_type in self.tuner_types]
        for tuner in tuners:
            file_name = (f"{self.base_path_}_{module}_{tuner.name}.txt")
            with open(file_name, "x", buffering=1) as fh:
                tuner.tune(self.budget, file=fh)
        if self.parallel is not None:
            self.latch_arrive(module)
            while not self.latch_finished():
                evaluator.evaluate(self.default_optimization)
        return tuners

    def run_(self) -> None:
        if self.parallel is not None:
            with tempfile.TemporaryDirectory() as sync_dir:
                self.sync_dir = sync_dir
                with multiprocessing.Pool(processes=self.parallel) as pool:
                    self.results = pool.map(self.tuning_thread_, self.modules)
        else:
            self.results = [self.tuning_thread_(module)
                            for module in self.modules]

    def write_results_(self) -> None:
        with open(f"{self.base_path_}.txt", "x") as fh:
            fh.write(f"search space size: {len(self.search_space) - 1}\n")
            fh.write(f"budget: {self.budget}\n")
            row_names = []
            for tuner in self.results[0]:
                row_names.append("Default")
                try:
                    tuner.train_runtime
                    row_names.append("Train")
                except AttributeError:
                    pass
                row_names.append(tuner.name)
                try:
                    tuner.evaluator.min()
                    row_names.append("Optimal")
                except AttributeError:
                    pass
            df = pd.DataFrame(index=row_names)
            for module, tuners in zip(self.modules, self.results):
                runtimes = []
                for tuner in tuners:
                    fh.write("\n")
                    fh.write(f"{module} with {tuner.name}\n")
                    # Baseline runtime
                    runtimes.append(tuner.default_runtime)
                    # Best runtime encountered in training data
                    try:
                        runtimes.append(tuner.train_runtime)
                        speedup_train = tuner.default_runtime / tuner.train_runtime
                        fh.write(f"speedup train: {speedup_train:.3f}\n")
                    except AttributeError:
                        pass
                    # Best runtime learned by tuner
                    runtimes.append(tuner.best_runtime)
                    speedup = tuner.default_runtime / tuner.best_runtime
                    fh.write(f"speedup: {speedup:.3f}\n")
                    # Optimal runtime
                    try:
                        min_runtime = tuner.evaluator.min()
                        runtimes.append(min_runtime)
                        speedup_optimal = tuner.default_runtime / min_runtime
                        fh.write(f"speedup optimal: {speedup_optimal:.3f}\n")
                    except AttributeError:
                        pass

                    fh.write(f"default runtime: {tuner.default_runtime:.3e}\n")
                    fh.write(f"best runtime: {tuner.best_runtime:.3e}\n")
                    default_flags = flag_info.optimization_to_str(
                        tuner.default_optimization, tuner.search_space)
                    best_flags = flag_info.optimization_to_str(
                        tuner.best_optimization, tuner.search_space)
                    fh.write(f"default flags: {default_flags}\n")
                    fh.write(f"best flags: {best_flags}\n")
                df[module] = runtimes
        df.to_csv(f"{self.base_path_}.csv")
        # Validate scores
        row_names = [tuner.name for tuner in self.results[0]
                     if hasattr(tuner, "validate_score")]
        df = pd.DataFrame(index=row_names)
        for module, tuners in zip(self.modules, self.results):
            df[module] = [tuner.validate_score for tuner in tuners
                          if hasattr(tuner, "validate_score")]
        df.to_csv(f"{self.base_path_}_validate_score.csv")
