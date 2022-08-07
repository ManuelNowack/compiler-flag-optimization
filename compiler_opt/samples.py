import multiprocessing
import os
import random
import sys
import tempfile

import numpy as np
import pandas as pd

from . import evaluator
from . import flag_info
from .typing import Optimization, SearchSpace


class Samples():
    def __init__(
            self,
            modules: list[str],
            search_space: SearchSpace,
            samples: int,
            parallel: int = None):
        self.modules = modules
        self.search_space = search_space
        self.samples = samples
        self.parallel = parallel
        self.rng = random.Random(42)
        self.optimizations = [self.random_optimization_()
                              for _ in range(self.samples)]
        self.run_()
        self.write_results_()

    def random_optimization_(self) -> Optimization:
        optimization = {"stdOptLv": 3}
        for flag_name, domain in self.search_space.items():
            if flag_name == "stdOptLv":
                continue
            optimization[flag_name] = self.rng.choice(domain)
        return optimization

    def latch_arrive(self, module: str) -> None:
        with open(os.path.join(self.sync_dir, module), "x"):
            pass

    def latch_finished(self) -> None:
        for module in self.modules:
            if not os.path.isfile(os.path.join(self.sync_dir, module)):
                return False
        return True

    def sample_thread_(self, module: str) -> list[float]:
        program, dataset, command = module.split(":")
        sample_evaluator = evaluator.Evaluator(
            program, dataset, command, self.search_space)
        run_times = []
        for opt in self.optimizations:
            try:
                run_times.append(sample_evaluator.evaluate(opt))
            except Exception as e:
                run_times.append(0.0)
                opt_str = flag_info.optimization_to_str(opt, self.search_space)
                print(f"{e} at {opt_str}", file=sys.stderr)
        if self.parallel is not None:
            self.latch_arrive(module)
            while not self.latch_finished():
                sample_evaluator.evaluate(opt)
        return run_times

    def run_(self) -> None:
        if self.parallel is not None:
            with tempfile.TemporaryDirectory() as sync_dir:
                self.sync_dir = sync_dir
                with multiprocessing.Pool(processes=self.parallel) as pool:
                    self.results = pool.map(self.sample_thread_, self.modules)
        else:
            self.results = [self.sample_thread_(module)
                            for module in self.modules]

    def write_results_(self) -> None:
        flags = [flag_info.optimization_to_str(opt, self.search_space)
                 for opt in self.optimizations]
        data = np.array(self.results).transpose()
        df = pd.DataFrame(data=data, index=flags, columns=self.modules)
        df.to_csv(f"samples/{self.samples}_{len(self.search_space)}.csv")
