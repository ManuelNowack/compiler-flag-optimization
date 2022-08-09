from typing import TextIO

import numpy as np
import pandas as pd

from . import base_tuner
from . import evaluator
from . import flag_info
from . import simulator
from .typing import Optimization, SearchSpace


class RandomTuner(base_tuner.Tuner):
    def __init__(
            self,
            search_space: SearchSpace,
            evaluator: evaluator.Evaluator,
            default_optimization: Optimization):
        super().__init__(search_space, evaluator, "RandomTuner", default_optimization)
        self.visited = set()

    def find_best_optimization(
            self,
            budget: int,
            file: TextIO = None) -> Optimization:
        if isinstance(self.evaluator, simulator.Simulator):
            rng = np.random.default_rng(42)

            def random_optimization() -> Optimization:
                optimization = {"stdOptLv": 3}
                for flag_name, domain in self.search_space.items():
                    if flag_name == "stdOptLv":
                        continue
                    optimization[flag_name] = rng.choice(domain)
                return optimization
            best_runtime = float("inf")
            for _ in range(budget):
                optimization = random_optimization()
                runtime = self.evaluator.evaluate(optimization)
                if file is not None:
                    print(runtime, file=file)
                if best_runtime > runtime:
                    best_runtime = runtime
                    best_optimization = optimization
        else:
            samples_path = f"samples/10000_{len(self.search_space) - 1}.csv"
            df = pd.read_csv(samples_path, index_col=0)
            module = (f"{self.evaluator.program}:{self.evaluator.dataset}"
                      f":{self.evaluator.command}")
            samples = df[module].sample(budget)
            if file is not None:
                samples.cummin().to_string(file, index=False)
            best_flags = samples.idxmin()
            best_optimization = flag_info.str_to_optimization(
                best_flags, self.search_space)
        return best_optimization
