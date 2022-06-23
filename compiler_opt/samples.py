import multiprocessing
import random
import sys

import numpy as np
import pandas as pd

from . import Evaluator
from . import flag_info
from .typing import Optimization


class Samples():
    def __init__(
            self,
            programs: tuple[str, str, str],
            samples: int,
            parallel: int = None):
        self.programs = programs
        self.samples = samples
        self.parallel = parallel
        self.search_space = flag_info.read_gcc_search_space("gcc_opts.txt")
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

    def sample_thread_(self, program: str, dataset: str,
                       command: str) -> list[float]:
        evaluator = Evaluator(program, 1, self.search_space, dataset, command)
        run_times = []
        for opt in self.optimizations:
            try:
                run_times.append(evaluator.evaluate(opt))
            except RuntimeError as e:
                run_times.append(0.0)
                opt_str = flag_info.optimization_to_str(opt, self.search_space)
                print(f"{e} at {opt_str}", file=sys.stderr)
        return run_times

    def run_(self) -> None:
        if self.parallel is not None:
            with multiprocessing.Pool(processes=self.parallel) as pool:
                self.results = pool.starmap(self.sample_thread_, self.programs)
        else:
            self.results = [self.sample_thread_(program, dataset, command)
                            for program, dataset, command in self.programs]

    def write_results_(self) -> None:
        program_names = [program if command == "" else f"{program}-{command}"
                         for program, _, command in self.programs]
        flags = [flag_info.optimization_to_str(opt, self.search_space)
                 for opt in self.optimizations]
        data = np.array(self.results).transpose()
        df = pd.DataFrame(data=data, index=flags, columns=program_names)
        df.to_csv(f"samples/{self.samples}.csv")
