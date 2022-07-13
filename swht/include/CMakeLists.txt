#===================
# External includes
#===================


# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Ready external includes
add_subdirectory(fastwht)
add_subdirectory(reed-solomon)

# message(STATUS "Unpacking SCIP")
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E tar xzf "${INCLUDE_SRC_DIR}/scipoptsuite-7.0.3.tgz"
#     WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
# )
# message(STATUS "Configuring SCIP")
# set(SCIPOPT_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/scipoptsuite-7.0.3")
# set(SCIPOPT_BINARY_DIR "${SCIPOPT_SOURCE_DIR}/build")
# make_directory("${SCIPOPT_BINARY_DIR}")
# set(SCIPOPT_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/scip_install")
# make_directory("${SCIPOPT_INSTALL_DIR}")
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=../scip_install
#         -DIPOPT=OFF -DREADLINE=OFF -DZIMPL=OFF -DZLIB=OFF
#         -Bbuild -H.
#     WORKING_DIRECTORY "${SCIPOPT_SOURCE_DIR}"
# )
# message(STATUS "Installing SCIP (this may take a while).")
# execute_process(COMMAND ${CMAKE_COMMAND} --build "${SCIPOPT_BINARY_DIR}" --target install)

# Build random signal class
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    find_package(Python3 COMPONENTS Development REQUIRED)
    find_library(GMP_LOCATE NAMES libgmpxx gmpxx)
    if ("${GMP_LOCATE}" STREQUAL "GMP_LOCATE-NOTFOUND")
        message(FATAL_ERROR "Cannot locate gmpxx library (required).")
    endif()
    find_path(GMP_LOCATE gmpxx.h)
    if ("${GMP_LOCATE}" STREQUAL "GMP_LOCATE-NOTFOUND")
        message(FATAL_ERROR "Cannot locate gmpxx.h file (required).")
    endif()
    add_library(RandomSignal SHARED random_signal.cpp random_signal.h)
    target_include_directories(RandomSignal PRIVATE "${SRC_DIR}/utils" ${Python3_INCLUDE_DIRS})
    target_link_libraries(RandomSignal PRIVATE gmpxx ${Python3_LIBRARIES})
endif()

# Set build type option
if("${CMAKE_BUILD_TYPE}" STREQUAL "Benchmark")
    set(BENCHMARKING_BUILD true)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Profile")
    set(BENCHMARKING_BUILD true)
    set(PROFILING_BUILD true)
endif()


# Export build info to code
configure_file(build_info.h.in build_info.h)
