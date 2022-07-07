# Compiler Flags

## Requirements

* Conda
* CMake
* GCC

## Installation

```
conda create -n compiler-flags -c conda-forge python anytree cvxopt cvxpy fast-histogram matplotlib numpy pandas pyscipopt pybind11 requests scikit-learn scipy tqdm
conda activate compiler-flags
cd sparse-dsft/cpp; cmake .; make; cd ..; pip install .; cd ..
pip install ck
ck pull repo:mlcommons@ck-mlops
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
cd cDatasets; ./download.sh; ./install.sh; cd ..
ck detect soft:compiler.gcc
conda env config vars set OMP_NUM_THREADS=1
mkdir results
```

Setting `OMP_NUM_THREADS` to 1 restricts CVXPY to a single thread. This ensures concurrent benchmarks do not fight over threads. Furthermore, we observed CVXPY solves these optimization problems faster on a (highly) scalable system when not parallelized.
