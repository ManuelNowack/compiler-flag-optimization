#=============================
# Python module configuration
#=============================

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Create Python module
Python3_add_library(swht MODULE swhtmodule.cpp)
target_include_directories(swht PRIVATE ${Python3_INCLUDE_DIRS} "${SRC_DIR}")
target_link_libraries(swht PRIVATE ${Python3_LIBRARIES} swht_cpp)
