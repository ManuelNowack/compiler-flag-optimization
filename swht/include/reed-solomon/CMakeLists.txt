#=====================================
# Reed-Solomon error correction codes
#=====================================

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Build object
add_library(reed_solomon OBJECT reed-solomon.cpp reed-solomon.h)
set_property(TARGET reed_solomon PROPERTY POSITION_INDEPENDENT_CODE ON)
