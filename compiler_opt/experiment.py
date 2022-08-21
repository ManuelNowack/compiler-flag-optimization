import multiprocessing
import os
import tempfile

import pandas as pd

from . import base_tuner
from . import flag_info
from . import simulator
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
        self.nonce_()
        self.run_()
        self.write_results_()

    def nonce_(self) -> None:
        for i in range(100):
            try:
                with open(f"results/tuning_{i:02d}.txt", "x"):
                    self.nonce = i
                    break
            except Exception:
                pass
        else:
            raise ValueError("No nonce available")

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
            file_name = (f"results/tuning_{self.nonce:02d}_{module}"
                         f"_{tuner.name}.txt")
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
        with open(f"results/tuning_{self.nonce:02d}.txt", "a") as fh:
            fh.write(f"search space size: {len(self.search_space) - 1}\n")
            fh.write(f"budget: {self.budget}\n")
        if issubclass(self.evaluator_type, simulator.Simulator):
            row_names = [name for tuner in self.results[0]
                         for name in ("Default", tuner.name, "Optimal")]
        else:
            row_names = [name for tuner in self.results[0]
                         for name in ("Default", tuner.name)]
        df = pd.DataFrame(index=row_names)
        for module, tuners in zip(self.modules, self.results):
            if issubclass(self.evaluator_type, simulator.Simulator):
                df[module] = [
                    runtime for tuner in tuners for runtime in (
                        tuner.default_runtime,
                        tuner.best_runtime,
                        tuner.evaluator.min())]
            else:
                df[module] = [
                    runtime for tuner in tuners for runtime in (
                        tuner.default_runtime,
                        tuner.best_runtime)]
            for tuner in tuners:
                default_flags = flag_info.optimization_to_str(
                    tuner.default_optimization, tuner.search_space)
                best_flags = flag_info.optimization_to_str(
                    tuner.best_optimization, tuner.search_space)
                speedup = tuner.default_runtime / tuner.best_runtime
                try:
                    speedup_train = tuner.default_runtime / tuner.train_runtime
                except AttributeError:
                    speedup_train = None
                if issubclass(self.evaluator_type, simulator.Simulator):
                    speedup_optimal = tuner.default_runtime / tuner.evaluator.min()
                else:
                    speedup_optimal = None
                with open(f"results/tuning_{self.nonce:02d}.txt", "a") as fh:
                    fh.write("\n")
                    fh.write(f"{module} with {tuner.name}\n")
                    fh.write(f"speedup: {speedup:.3f}\n")
                    if speedup_train is not None:
                        fh.write(f"speedup train: {speedup_train:.3f}\n")
                    if speedup_optimal is not None:
                        fh.write(f"speedup optimal: {speedup_optimal:.3f}\n")
                    fh.write(f"default runtime: {tuner.default_runtime:.3e}\n")
                    fh.write(f"best runtime: {tuner.best_runtime:.3e}\n")
                    fh.write(f"default flags: {default_flags}\n")
                    fh.write(f"best flags: {best_flags}\n")
        df.to_csv(f"results/tuning_{self.nonce:02d}.csv")
