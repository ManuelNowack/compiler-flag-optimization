import numpy as np

from .typing import Optimization, SearchSpace


class PowerSet:
    def __init__(self, search_space: SearchSpace):
        self.binary_flags = []
        self.parametric_flags = []
        for flag_name, domain in search_space.items():
            if flag_name == "stdOptLv":
                continue
            assert len(domain) > 1
            if domain == (False, True):
                self.binary_flags.append(flag_name)
            else:
                for value in domain:
                    self.parametric_flags.append((flag_name, value))
        self.num_elements = len(self.binary_flags) + len(self.parametric_flags)

    def subset_to_optimization_(self, subset: np.ndarray) -> Optimization:
        optimization = {"stdOptLv": 3}
        subset_it = iter(subset)
        for flag_name, is_flag_active in zip(self.binary_flags, subset_it):
            optimization[flag_name] = bool(is_flag_active)
        for (flag_name, value), is_flag_active in zip(
                self.parametric_flags, subset_it):
            if is_flag_active:
                optimization[flag_name] = value
        return optimization

    def optimization_to_subset_(
            self, optimization: Optimization) -> np.ndarray:
        # TODO: Remove empty configs from input file
        # assert optimization.keys() == self.search_space.keys()
        subset = np.zeros(self.num_elements)
        for flag_name, value in optimization.items():
            if flag_name == "stdOptLv":
                # TODO: Ensure assertion
                # assert value == 3
                continue
            elif value is True:
                index = self.binary_flags.index(flag_name)
                subset[index] = 1.0
            elif value is False:
                pass  # Already initialized to zero
            else:
                index = self.parametric_flags.index((flag_name, value))
                subset[len(self.binary_flags) + index] = 1.0
        return subset
