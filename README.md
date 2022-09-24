# Compiler Flags

## Project Structure

* `BOCS` contains [Bayesian Optimization of Combinatorial Structures (BOCS)](https://github.com/baptistar/BOCS)
* `SRTuner` contains [SRTuner](https://github.com/sunggg/SRTuner)
* `analysis` contains various scripts to analyze the output
* `cDatasets` contains a shell script to download and install additional datasets from [cBench](https://sourceforge.net/projects/cbenchmark/files/cDatasets/)
* `compiler_opt` contains our code
* `evaluation` contains the output of the experiments
* `gcc_flags` contains a compilation of gcc flags
* `samples` contains randomly sampled runtime measurements
* `simulation` contains the output of the simulated experiments
* `sparse-dsft` contains an implementation of Fourier-sparse approximations of set functions by Chris Wendler and Eliza Wszola
* `stability` contains the output of the stability experiments
* `swht` contains an implementation of sparse Walsh-Hadamard transform by Kaiko Bonstein and Andisheh Amrollahi

## Requirements

* Conda
* CMake
* GCC

## Installation

```
conda create -n compiler-flags -c conda-forge python anytree cvxopt cvxpy fast-histogram jinja2 matplotlib numpy pandas pyscipopt pybind11 requests scikit-learn scipy tqdm
conda activate compiler-flags
cd sparse-dsft/cpp; cmake .; make; cd ..; pip install .; cd ..
cd swht; cmake -DCMAKE_INSTALL_PREFIX=./build -B build .; cmake --build build; cd build; make install; cd ..; pip install .; cd ..
pip install ck
ck pull repo:mlcommons@ck-mlops
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
cd cDatasets; ./download.sh; ./install.sh; cd ..
ck detect soft:compiler.gcc
conda env config vars set OMP_NUM_THREADS=1
conda env config vars set LD_LIBRARY_PATH=$(conda info --base)/envs/compiler-flags/lib
mkdir results
```

Setting `OMP_NUM_THREADS` to 1 restricts CVXPY to a single thread. This ensures concurrent benchmarks do not fight over threads. Furthermore, we observed CVXPY solves these optimization problems faster on a (highly) scalable system when not parallelized.

Setting `LD_LIBRARY_PATH` ensures swht can import the libpython*.so files when using conda.
