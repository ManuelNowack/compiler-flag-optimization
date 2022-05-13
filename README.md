# Compiler Flags

## Installation
```
pip install ck
ck pull repo:mlcommons@ck-mlops
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
```
### SRTuner
```
pip install fast_histogram numpy anytree pandas
ck add dataset:cdataset-dijkstra-0002 --tags=dijkstra,dataset --file=cBench/network_dijkstra_data/2.dat
ck add dataset:image-ppm-0002 --tags=image,ppm,dataset --file=cBench/consumer_jpeg_data/2.ppm
ck add dataset:adpcm-0002 --tags=audio,adpcm,dataset --file=cBench/telecom_data/2.adpcm
```
