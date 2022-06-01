# Compiler Flags

## Requirements

* Conda
* CMake
* GCC

## Installation

```
conda create -n compiler-flags -c conda-forge python anytree cvxopt cvxpy fast-histogram matplotlib numpy pandas pyscipopt pybind11 scikit-learn scipy tqdm
conda activate compiler-flags
cd sparse-dsft/cpp
cmake .
make
cd ..
pip install .
cd ..
pip install ck
ck pull repo:mlcommons@ck-mlops
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
ck detect soft:compiler.gcc
mkdir results
```

## Larger Datasets

Larger datasets were made available in a Google Drive which no longer appears to be accessible. However, we can add new datasets ourselves, e.g., from the original cBench.
First, determine the tags of a program by looking at the key `dict.run_cmds.default.dataset_tags` in the output of
```
ck load program:cbench-network-dijkstra --min
```
where `cbench-network-dijkstra` may be substitued by any program.

Then, add the file(s) using the recovered tags.
```
ck add dataset:cdataset-dijkstra-0002 --tags=dijkstra,dataset --file=cBench/network_dijkstra_data/2.dat
ck add dataset:image-ppm-0002 --tags=image,ppm,dataset --file=cBench/consumer_jpeg_data/2.ppm
ck add dataset:adpcm-0002 --tags=audio,adpcm,dataset --file=cBench/telecom_data/2.adpcm
```
