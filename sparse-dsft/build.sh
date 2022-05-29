#!/bin/bash
echo "Changing directory..."
cd cpp
pwd
echo "Compiling high performance code"
cmake .
make
echo "Changing directory ..."
cd ..
pwd
echo "Installing python modules"
pip install .
