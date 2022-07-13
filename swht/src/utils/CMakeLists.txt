#=====================
# Utilities gathering
#=====================

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Path definition
set(UTILS_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Gather files
set(UTILS_FILES
    ${UTILS_PATH}/global_constants.h
    ${UTILS_PATH}/Hasher.cpp
    ${UTILS_PATH}/Hasher.h
    ${UTILS_PATH}/linear_algebra.h
)

# Forward list
set(UTILS ${UTILS_FILES} PARENT_SCOPE)
