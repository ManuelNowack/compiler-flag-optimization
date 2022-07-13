from math import isclose
from numpy import array, float64, logical_xor, logical_and
from numpy.random import default_rng
from typing import Tuple, List
from typing_extensions import Literal
from math import factorial
from fractions import Fraction


# This class has all the properties of RandomFunction except that it returns
# fourier coefficients instead of actual function value

class RandomFunction(object):
    """
    Creates a signal with k randomly generated Fourier coefficients

    Usage
    -----
    Generate a random signal with 3 frequencies of size 5 and degree 2:
    >>> f = RandomFunction(5, 3, 2)

    Query the signal at a time index:
    >>> f[(0, 0, 1, 1, 0)]
    """

    def __init__(self, n: int, k: int, degree: int = None) -> None:
        """
        RandomFunction(n, k, degree=None)

        Constructor

        Parameters
        ----------
        n: int
            size of the frequency tuples
        k: int
            number of random coefficients to generate
        degree: int
            degree of the generated frequencies (default = n)
        """
        self.n = n
        self.k = k
        self.prng = default_rng()
        self.degree = n if degree is None else degree
        self.dict = {}
        self.cache = []
        self.cache_iter = None
        self.binomials = array(self.__binom(n, self.degree + 1))
        self.binomials = (self.binomials / self.binomials.sum()).astype(float64)
        for _ in range(k):
            self.add_random_coeff()
    
    def add_random_coeff(self) -> None:
        """Adds a randomly generated Fourier coefficient."""
        while True:
            freq_degree = self.prng.choice(self.degree + 1, p=self.binomials)
            sampled_coordinates = self.prng.choice(self.n, size=freq_degree, replace=False)
            freq = tuple(int(i in sampled_coordinates) for i in range(self.n))
            if freq not in self.dict:
                break
        self.dict[freq] = self.prng.integers(1, 10)
    
    def add_coeff(self, freq: tuple, x: float) -> None:
        """
        Adds a fourier coefficient to the signal.

        Parameters
        ----------
        freq: tuple of binary ints
            frequency at which to add a coefficient
        x: float
            amplitude of the given frequency
        """
        if freq not in self.dict:
            self.k += 1
        self.dict[freq] = x
    
    def ready_cache_run(self):
        self.cache_iter = iter(self.cache)

    def __call__(self, t: Tuple[Literal[0, 1]]) -> float:
        """
        random_function(t)

        Queries the signal.

        Parameters
        ----------
        t: tuple of binary ints
            time index at which to query

        Returns
        -------
        Value at the queried time index with small additive gaussian noise.
        """
        if self.cache_iter:
            return self.cache_iter.__next__()
        value = 0
        for freq, amp in self.dict.items():
            if logical_xor.reduce(logical_and(freq, t)):
                value -= amp
            else:
                value += amp
        self.cache.append(value)
        return value# + 0.001 * self.prng.standard_normal()

    def __str__(self) -> str:
        """String representation
        
        Returns
        -------
        Stringified internal dictionary of coefficients.
        """
        return str(self.dict)

    def __eq__(self, other: 'RandomFunction') -> bool:
        """Verifies closeness between signals. They must have the
        same frequencies and close amplitudes."""
        if not isinstance(other, RandomFunction):
            return False
        count = 0
        for freq, x in self.dict.items():
            try:
                if isclose(x, other.dict[freq], rel_tol=0.1):
                    count += 1
            except KeyError:
                return False
        return count == len(self.dict) and count == len(other.dict)

    @staticmethod
    def create_from_FT(n, fourier: dict) -> 'RandomFunction':
        """
        Generates a signal from a pre-existing dictionary of coefficients.

        Parameters
        ----------
        fourier: dict of floats (amplitudes) with tuples of binary ints as keys (frequencies)
            Fourier coefficients of the signal
        """
        f = RandomFunction(n, 0)
        for freq, x in fourier.items():
            f.add_coeff(freq, x)
        return f

    @staticmethod
    def __binom(n: int, d: int) -> List[Fraction]:
        """Generates binomial coefficients from 0 to d among n."""
        n_factorial = factorial(n)
        return [Fraction(n_factorial, factorial(k) * factorial(n - k)) for k in range(d)]


if __name__ == '__main__':
    a = RandomFunction(5, 2, 2)
    print(a)
