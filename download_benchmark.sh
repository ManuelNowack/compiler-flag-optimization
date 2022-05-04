#!/bin/bash
if [ ! -d benchmarks ]; then
    echo "Setting up benchmarks"
    mkdir benchmarks
    cd benchmarks
    echo "Downloading polybench"
    curl -L "https://sourceforge.net/projects/polybench/files/polybench-c-4.2.tar.gz" > polybench.tar.gz
    echo "Extracting polybench"
    mkdir polybench
    tar -xzf polybench.tar.gz -C polybench && rm polybench.tar.gz
    mv polybench/polybench-c-4.2/* polybench && rm -rf polybench/polybench-c-4.2
    echo "Changing permissions for timing script"
    chmod 755 polybench/utilities/time_benchmark.sh
    echo "Downloading cBench"
    curl -L "https://sourceforge.net/projects/cbenchmark/files/cBench/V1.1/cBench_V1.1.tar.gz" > cBench.tar.gz
    echo "Extracting cBench"
    mkdir cBench
    tar -xzf cBench.tar.gz -C cBench && rm cBench.tar.gz
    echo "Fixing order of -lm flag"
    sed "s/-lm \*.o/\*.o -lm/" -i benchmarks/cBench/*/src/Makefile*
    sed "s/__compile \$COMPILER/.\/__compile \$COMPILER/" -i benchmarks/cBench/all_compile
    cd ..
fi
