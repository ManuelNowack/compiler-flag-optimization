cmake_minimum_required(VERSION 2.8.9)
project(sparsedsft)

include_directories(cpp)

set(CMAKE_CXX_FLAGS " -pthread -std=c++14 -funroll-loops -O2 -march=native -mavx2 -fPIC -lrt")

set(HEADER_FILES
    src/common.hpp
    src/dependencygraph.hpp
    src/threadpool.hpp
    src/posettp.hpp
    src/baseline.hpp
    src/lasso.hpp
    src/structset.hpp
    src/correlation.hpp)

set(SOURCE_FILES
    src/main.cpp
    src/common.cpp
    src/dependencygraph.cpp
    src/threadpool.cpp
    src/posettp.cpp
    src/baseline.cpp
    src/lasso.cpp
    src/structset.cpp
    src/correlation.cpp)

add_executable(estimator ${SOURCE_FILES})

add_library(dsft STATIC ${SOURCE_FILES} src/export.cpp ${HEADER_FILES} src/export.hpp)
install (TARGETS dsft ARCHIVE DESTINATION lib)

add_library(dsft-shared SHARED ${SOURCE_FILES} src/export.cpp ${HEADER_FILES} src/export.hpp)
install (TARGETS dsft-shared LIBRARY DESTINATION lib)
