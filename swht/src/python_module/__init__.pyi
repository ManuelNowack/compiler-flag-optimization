#====================
# SWHT lib stub file
#====================
"""
SWHT package
------------

This package offers the swht function for fast sparse Walsh-Hadamard transforms.
It also defines a overridable binary_signal class and the NAIVE, RANDOM_BINNING
and REED_SOLOMON constants.
"""

from typing import Callable, Dict, Tuple
from typing_extensions import Literal


NAIVE: str = ...
RANDOM_BINNING: str = ...
REED_SOLOMON: str = ...


class binary_signal:
    """
    Binary signal
    --

    Abstract class representing a signal with bit-sequence indexes and real
    responses.
    
    Usage
    -----
    Overload and provide the __call__(self, Tuple[Literal[0, 1]]) method.
    """

    def __call__(self, index: Tuple[Literal[0, 1]]) -> float:
        """
        __call__(self, index)

        Queries the signal at the given index.

        Arguments
        ---------
        index: tuple[int]
            queried index as a tuple of 0-1 ints

        Returns
        -------
        Signal value at the queried index as a float.
        """
        ...


def swht(signal: Callable[[Tuple[Literal[0, 1]]], float], cs_algorithm: str,
        n: int, K: int, C: float = 1.3, ratio: float = 1.4,
        robust_iterations: int = 1, cs_bins: int = None, cs_iterations: int = 1,
        cs_ratio: float = 2.0, degree: int = None) -> Dict[Tuple[Literal[0, 1]], float]:
    """
    swht(signal, cs_algorithm, n, K, C, ratio, /, robust_iterations=1,
        cs_bins=None, cs_iterations=1, cs_ratio=2.0, degree=None)
    --

    Performs a non-robust Sparse Walsh-Hadamard Transform of the input signal.
    
    Parameters
    ----------
    signal: Python callable
        signal to be transformed
    cs_algorithm: str
        compressive sensing algorithm to be used for frequency retrieval
        (NAIVE, RANDOM_BINNING or REED_SOLOMON)
    n: int
        index size of the input signal
    K: int
        signal sparsity
    C: float (optional)
        bucket constant
    ratio: float
        buckets reduction ratio
    robust_iterations: int (optional)
        number of robust iterations to perform (1 = default non-robust)
    cs_bins: int (required - random binning)
        number of bins to hash into before binary search
    cs_iterations: int (optional - random binning)
        number of hashing and binary search rounds to perform
    cs_ratio: int (optional - random binning)
        reduction ratio of the random binning bins across iterations
    degree: int (required - Reed-Solomon)
        degree of the signal

    Returns
    -------
    A dictionary containing the frequency-amplitude mapping of the transform.
    """
    ...
