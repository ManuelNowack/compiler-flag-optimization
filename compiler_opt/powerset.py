import numpy as np

from .typing import Optimization, SearchSpace

class PowerSet:
    def __init__(self, search_space: SearchSpace):
        self.binary_flags = []
        self.parametric_flags = []
        for flag, configs in search_space.items():
            if flag == "stdOptLv":
                continue
            assert len(configs) > 1
            if configs == (False, True):
                self.binary_flags.append(flag)
            else:
                for val in configs:
                    self.parametric_flags.append((flag, val))
        self.num_elements = len(self.binary_flags) + len(self.parametric_flags)

    def subset_to_optimization_(self, subset: np.ndarray) -> Optimization:
        optimization = {"stdOptLv": 3}
        subset_it = iter(subset)
        for flag, enabled in zip(self.binary_flags, subset_it):
            optimization[flag] = bool(enabled)
        for (flag, val), enabled in zip(self.parametric_flags, subset_it):
            if enabled:
                optimization[flag] = val
        return optimization

    def optimization_to_subset_(
            self, optimization: Optimization) -> np.ndarray:
        # TODO: Remove empty configs from input file
        # assert optimization.keys() == self.search_space.keys()
        subset = np.zeros(self.num_elements)
        for flag_name, config in optimization.items():
            if flag_name == "stdOptLv":
                # TODO: Ensure assertion
                # assert config == 3
                continue
            elif config is True:
                index = self.binary_flags.index(flag_name)
                subset[index] = 1.0
            elif config is False:
                pass  # Already initialized to zero
            else:
                index = self.parametric_flags.index((flag_name, config))
                subset[len(self.binary_flags) + index] = 1.0
        return subset
