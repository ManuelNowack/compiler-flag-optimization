#==============
# SWHT targets
#==============

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Require Python 3
find_package(Python3 COMPONENTS Development REQUIRED)

# Gather utilities
set(UTILS "")
add_subdirectory(utils)

# Gather CS providers
set(CS_PROVIDERS "")
add_subdirectory(cs_providers)

# Build basic SWHT
add_library(swht_kernel SHARED swht_basic.cpp swht_robust.cpp swht_kernel.h ${UTILS} ${CS_PROVIDERS} "${CMAKE_BINARY_DIR}/include/build_info.h")
target_include_directories(swht_kernel PUBLIC
    ${Python3_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/utils"
    "${CMAKE_CURRENT_SOURCE_DIR}/cs_providers"
    "${INCLUDE_SRC_DIR}"
    "${CMAKE_BINARY_DIR}/include/"
)
target_link_libraries(swht_kernel PUBLIC fwht_kernel m reed_solomon ${Python3_LIBRARIES})

# Build SWHT C++ interface
add_library(swht_cpp SHARED swht.cpp swht.h)
target_include_directories(swht_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/utils")
target_link_libraries(swht_cpp PRIVATE swht_kernel)

# Release install options
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    include(GNUInstallDirs)
    set_target_properties(swht_cpp PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER swht.h
    )
    set_target_properties(swht_kernel PROPERTIES
        PUBLIC_HEADER swht_kernel.h
    )
    install(TARGETS swht_cpp swht_kernel
        PUBLIC_HEADER
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/swht"
        LIBRARY
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/utils/global_constants.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/swht"
    )

    # Record output filename
    file(WRITE "${CMAKE_BINARY_DIR}/src/libname.log"
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\n"
        "${CMAKE_SHARED_LIBRARY_PREFIX}swht_kernel${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# Configure Python interface
add_subdirectory(python_module)
