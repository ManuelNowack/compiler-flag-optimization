# swht

## Table of contents
- [1. Sparse Walsh-Hadamard transform](#1-sparse-walsh-hadamard-transform)
  * [1.1 Walsh-Hadamard transforms](#11-walsh-hadamard-transforms)
  * [1.2 Compressive sensing](#12-compressive-sensing)
  * [1.3 Optimizations](#13-optimizations)
- [2. Running the library](#2-running-the-library)
  * [2.1 Dependencies](#21-dependencies)
  * [2.1 Building](#21-building)
  * [2.2 Installing](#22-installing)
  * [2.4 Running in Python](#24-running-in-python)
  * [2.5 Running in C++](#25-running-in-c--)
  * [2.6 Tests](#26-tests)
  * [2.7 Setup script](#27-setup-script)
- [References](#references)

## 1. Sparse Walsh-Hadamard transform
This is an implementation of A. Amrollahi's sparse Walsh-Hadamard transform, as he defined it in [[1]](#1). Each branch corresponds to an optimization step performed (see section [[1.3]](#12-compressive-sensing) for details).

### 1.1 Walsh-Hadamard transforms
The Walsh hadamard transform is a variant of the Fourier transform in the vector space of the finite field (or Galois field) of order 2, which corresponds to binary (0-1) vectors of size *n*.

If we make the assumption that the input signal is *K*-sparse (i.e that it has only *K* non-null frequencies in the Walsh-Hadamard domain) this algorithm gives the transform with a high probability while requiring fewer samples than a normal fast WHT. We achieve this by collecting a small subset of samples into hash buckets that we transform and from which we then retrieve the original frequency using compressive sensing over the finite field (the sparsity reduces the collision risk). By iteratively performing this process and "peeling" the source signal to make progressively more sparse, we obtain the transform with probability *>90%*.

If we make the additional assumption that the non-null frequencies of the signal are of a low degree *d* (Hamming weight of *d* or lower), we can use more efficient compressive sensing to further reduce the need for sampling.

### 1.2 Compressive sensing
We implement 3 compressive sensing algorithm in this repository:
* `naive`: This is the exhaustive search that makes no assumption on the degree of the signal frequencies.
* `random binning`: This assumes a low degree. It collects the frequency indexes into randomly selected bins and performs a binary search over each bins.
* `reed-solomon`: This assumes a low degree. It uses Reed-Solomon error correction codes to retrieve the frequency it was translated into C++ from [[2]](#2).

### 1.3 Optimizations
The branches are organized in the following order and perform the listed optimizations:
1. `original`: This is A. Amrollahi's original Python proof-of-concept on which this repository is based.
2. `master`: This is the base C++ translation of the original code.
3. `prehash`: Precomputes part of the hasing to save some time during sampling.
4. `low-py`: Reduces the Python C API overhead by pregenerating some objects and using lower-level calls.
5. `compress`: Defines an alternative bit sequence representation that allows for using lower-level and faster C++ data structures.
6. `vector`: Vectorizes some array operations (not faster).

## 2. Running the library
### 2.1 Dependencies
* gcc 8 (or later) (other compilers untested)
* cmake 3.13.4 (or later)
* python3.7 (or later)

Requirements for tests and benchmarks:
* boost 1.68.0 (or later)
* gmpxx 6.1.2 (or later)

### 2.1 Building
The library is configured and built with `cmake` with the following commands run from the project root:
```sh
cmake [-D CMAKE_BUILD_TYPE=<build-type>] -B build .
cmake --build build
```
Where `<build-type>` can be one of:
* `Release` (default): Only builds the core library and prepares for installing.
* `Debug`: Builds the library and the tests.
* `Benchmark`: Builds the library and the benchmark runners.
* `Profile`: Builds the library with instrumentation and the benchmark runners in profiling mode.

Alternatively the library can be fully built from anywhere from a bash console with the `setup.sh` script:
```bash
./<path-to-repo>/setup.sh ready [-D <build-type>] 
```

### 2.2 Installing
After building the library in `Release` mode it can be installed as a C++ library with `make`:
```sh
(cd build && sudo make install)
```
or with the `setup.sh` script:
```sh
./<path-to-repo>/setup.sh install
```
This builds the `swht_cpp` C++ dynamic library that can be linked with the `-lswht_cpp` flag (the runtime path must also be set when linking).
  
It can then be uninstalled with:
```sh
(cd build && sudo make uninstall)
```
or
```sh
./<path-to-repo>/setup.sh uninstall
```

Once the C++ library is installed the Python package can be installed with `pip` by running from the repository root:
```sh
python3 -m pip install .
```
It can then be uninstalled at any point with `pip` again:
```sh
python3 -m pip uninstall swht
```

### 2.4 Running in Python
After installing the library as a Python package, it can be imported with:
```python
import swht
```
It offers, the three useful constants `NAIVE`, `RANDOM_BINNING` and `REED_SOLOMON`, an abstract class `binary_signal` which can be overloaded to help implement a transformable signal and the `swht` function with the following API:
```python
swht(signal: Callable[[Tuple[Literal[0, 1]]], float],
    cs_algorithm: str, n: int, K: int, C: float = 1.3,
    ratio: float = 1.4, robust_iterations: int = 1,
    cs_bins: int = None, cs_iterations: int = 1,
    cs_ratio: float = 2.0, degree: int = None
    ) -> Dict[Tuple[Literal[0, 1]], float]
```
**Arguments:**
* `signal`: The signal to transform (must be callable with a tuple of binary ints and respond with a float value)
* `cs_algorithm`: The compressive sensing algorithm to use (can be one of `NAIVE`, `RANDOM_BINNING` or `REED_SOLOMON`)
* `n`: Signal index size (finite field vector dimension)
* `K`: Signal sparsity

**Optional arguments**
* `C`: Bucket constant (adjusts the number of buckets used)
* `ratio`: Buckets reduction ratio
* `robust_iterations`: Number of robust iterations to perform (the default of 1 calls the basic non-robust algorithm)

**Compressive sensing arguments**
* `cs_bins`: (random binning) Number of bins to hash into before binary search (if `degree` is defined instead this value is inferred from it)
* `cs_iterations`: (random binning) Number of binary search passes to perform
* `cs_ratio`: (random binning) Reduction ratio of the binary search bins
* `degree`: (Reed-Solomon or random binning) Degree of the random signal frequencies

**Examples**  
Some typical simple calls would look like the following:
```python
from swht import *
swht(signal, NAIVE, n, K) # basic call
swht(signal, NAIVE, n, K, robust_iterations=10) # robust call
swht(signal, RANDOM_BINNING, n, K, cs_bins=nbins) # random binning call
swht(signal, RANDOM_BINNING, n, K, degree=d) # alt. random binning call
swht(signal, REED_SOLOMON, n, K, degree=d) # Reed-Solomon call
```

### 2.5 Running in C++
The library can be run directly from C++ by linking the `swht_cpp` library from [build/src](build/src) and including [swht.h](src/swht.h) in the [src](src) folder (or getting both from your installation).

It is called with the following interface:
```C++
frequency_map swht(
    PyObject *signal, std::string cs_algorithm, // target
    unsigned long n, unsigned long K,           // size
    unsigned long robust_iterations = 1ul,      // robust option
    double C = 1.3, double ratio = 1.4,         // buckets
    unsigned long cs_bins = 0ul,                // random binning (1)
    unsigned long cs_iterations = 1ul,          // random binning (2)
    double cs_ratio = 2.0,                      // random binning (3)
    unsigned long degree = 0ul                  // Reed-Solomon and random binning
);
```
It follows the same argument semantics as the Python interface, with `cs_algorithm` being one of `"naive"`, `"random binning"` or `"reed-solomon"` (case insensitive). The `frequency_map` type is a `std::map` with `std::vector<unsigned>` as keys and `double` as values.

*__Note:__ From the* `compress` *branch and on, it changes to a* `std::unordered_map` *.*

### 2.6 Tests
After building in `Debug` mode, the unit tests can be run from the [build](build) folder with:
```sh
ctest [<ctest-options>...]
```
or from anywhere with:
```sh
./<path-to-repo>/setup.sh tests [--verbose|-v] [-m <module>|<pattern>]
```
where the `-m <module>` and `<pattern>` options allows for the selection of a subset of tests.

### 2.7 Setup script
To get the manual for the `setup.sh` script, use:
```sh
./<path-to-repo>/setup.sh -h
```

## References
<a id="1">[1]</a>
A. Amrollahi, A. Zandieh, M. Kapralov, and A. Krause, “Efficiently
Learning Fourier Sparse Set Functions,” in Advances in Neural Information
Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d.
Alch ́e-Buc, E. Fox, and R. Garnett, Eds., vol. 32. Curran Associates, Inc.,
2019. [Online]. Available: https://proceedings.neurips.cc/paper/2019/file/c77331e51c5555f8f935d3344c964bd5-Paper.pdf

<a id="2">[2]</a>
T. Filiba, "reedsolomon," Nov. 2020. [Online]. Available: https://github.com/tomerfiliba/reedsolomon
