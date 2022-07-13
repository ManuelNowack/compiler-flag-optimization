#=========================================
# Finite field compressive sensing module
#=========================================

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Path definition
set(CS_PROVIDERS_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Gather files
set(CS_PROVIDERS_FILES
    ${CS_PROVIDERS_PATH}/cs_factory.h
    ${CS_PROVIDERS_PATH}/cs_factory.cpp
    ${CS_PROVIDERS_PATH}/finite_field_cs.h
    ${CS_PROVIDERS_PATH}/naive_cs.cpp
    ${CS_PROVIDERS_PATH}/naive_cs.h
    ${CS_PROVIDERS_PATH}/random_binning_cs.cpp
    ${CS_PROVIDERS_PATH}/random_binning_cs.h
    ${CS_PROVIDERS_PATH}/reed_solomon_cs.cpp
    ${CS_PROVIDERS_PATH}/reed_solomon_cs.h
)

# Forward list
set(CS_PROVIDERS ${CS_PROVIDERS_FILES} PARENT_SCOPE)
