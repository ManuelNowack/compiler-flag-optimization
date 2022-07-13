#==========================
# Fast Hadamard kernel lib
#==========================


# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Create library target
add_library(fwht_kernel OBJECT hadamard.cpp)
target_include_directories(fwht_kernel PRIVATE
    fxt
    fxt/aux0
    fxt/bits
    fxt/perm
    fxt/walsh
)
set_property(TARGET fwht_kernel PROPERTY POSITION_INDEPENDENT_CODE ON)
