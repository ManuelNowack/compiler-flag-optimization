#============
# Main build
#============

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()


# Project name
project(swhtmodule)

# C/C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Compiler flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-O3 -Wall -Wextra -pedantic) # clang flags (LLVM Clang C++)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-O3 -Wall -Wextra -pedantic) # clang flags (Apple Clang C++)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-O3 -Wall -Wextra -pedantic) #gcc flags (GNU C++)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_compile_options(-O3 -w3) # icc flags (Intel C++)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/O2 /W4) # msvc flags (Microsoft Visual C++)
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "Benchmark" OR "${CMAKE_BUILD_TYPE}" STREQUAL "Profile")
    # add_compile_options(-march=skylake-avx512)
    add_compile_options(-march=native)
else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native X86_ARCH_FLAG)
    if(X86_ARCH_FLAG)
        add_compile_options(-march=native)
    endif()
endif()

message(STATUS "Compiler options set")

# Build options
option(PYTHON_API "Indicate whether to build the Python API or not" ON)

# Paths
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_SRC_DIR "${CMAKE_SOURCE_DIR}/include")

# Ready and fetch general includes
add_subdirectory("${INCLUDE_SRC_DIR}")
message(STATUS "Includes ready")

# Define main targets
add_subdirectory("${SRC_DIR}")
message(STATUS "Objects ready")

# Add benchmarking utilities
if("${CMAKE_BUILD_TYPE}" STREQUAL "Benchmark" OR "${CMAKE_BUILD_TYPE}" STREQUAL "Profile")
    add_subdirectory("${CMAKE_SOURCE_DIR}/benchmarking")
    message(STATUS "Benchmarking configured")
endif()


# Add testing interface
find_package(Boost QUIET COMPONENTS unit_test_framework)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    if(Boost_FOUND)
        enable_testing()
        add_subdirectory("${CMAKE_SOURCE_DIR}/tests")
        message(STATUS "Tests configured")
    else()
        message(STATUS "Boost testing not available")
    endif()
endif()

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
    message(STATUS "Uninstall target set")
endif()

