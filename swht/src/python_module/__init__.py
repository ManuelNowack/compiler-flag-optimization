#=======================
# SWHT lib base package
#=======================
"""
SWHT package
------------

This package offers the swht function for fast sparse Walsh-Hadamard transforms.
It also defines a overridable binary_signal class and the NAIVE, RANDOM_BINNING
and REED_SOLOMON constants.
"""

from typing import Tuple
from typing_extensions import Literal
from .swht import *


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
        raise NotImplementedError("The callable must be overriden with an actual signal.")


__all__ = [
    "swht",
    "binary_signal",
    "NAIVE",
    "RANDOM_BINNING",
    "REED_SOLOMON"
]
