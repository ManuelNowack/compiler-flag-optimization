#====================
# Tests registration
#====================

# cmake minimum version
cmake_minimum_required(VERSION 3.13.4)

# Load Boost.test
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

# Detailed test registering function
function(add_boost_suite TEST_FILE_NAME)

    # Generate executable
    string(REGEX REPLACE "test_(.+).cpp" "\\1" SUITE_NAME ${TEST_FILE_NAME})
    add_executable(test_${SUITE_NAME} ${TEST_FILE_NAME})

    # Find all auto-registered Boost test cases
    file(READ "${TEST_FILE_NAME}" SOURCE_FILE_CONTENTS)
    string(REGEX MATCHALL "BOOST_AUTO_TEST_CASE\\( *([A-Za-z_0-9]+) *\\)" 
           FOUND_TESTS ${SOURCE_FILE_CONTENTS})

    # Register each Boost test case individually in CTest
    foreach(HIT ${FOUND_TESTS})
        string(REGEX REPLACE ".*\\( *([A-Za-z_0-9]+) *\\).*" "\\1" TEST_NAME ${HIT})
        add_test("${SUITE_NAME}.${TEST_NAME}" test_${SUITE_NAME} --run_test=${TEST_NAME} --catch_system_error=yes)
    endforeach()

endfunction()


# Linear algebra tests
add_boost_suite(test_linear_algebra.cpp)
target_sources(test_linear_algebra PRIVATE)
target_include_directories(test_linear_algebra PRIVATE "${SRC_DIR}/utils")

# Reed-Solomon tests
add_boost_suite(test_reed_solomon.cpp)
target_include_directories(test_reed_solomon PRIVATE "${INCLUDE_SRC_DIR}/reed-solomon")
target_link_libraries(test_reed_solomon PRIVATE m reed_solomon)

# Naive SWHT tests
add_boost_suite(test_swht_naive.cpp)
target_include_directories(test_swht_naive PRIVATE "${SRC_DIR}" "${INCLUDE_SRC_DIR}" "${CMAKE_BINARY_DIR}/include")
target_link_libraries(test_swht_naive PRIVATE m swht_kernel RandomSignal)

# Random binning SWHT tests
add_boost_suite(test_swht_randbin.cpp)
target_include_directories(test_swht_randbin PRIVATE "${SRC_DIR}" "${INCLUDE_SRC_DIR}" "${CMAKE_BINARY_DIR}/include")
target_link_libraries(test_swht_randbin PRIVATE m swht_kernel RandomSignal)

# Reed-Solomon SWHT tests
add_boost_suite(test_swht_reed_solomon.cpp)
target_include_directories(test_swht_reed_solomon PRIVATE "${SRC_DIR}" "${INCLUDE_SRC_DIR}" "${CMAKE_BINARY_DIR}/include")
target_link_libraries(test_swht_reed_solomon PRIVATE m swht_kernel RandomSignal)

# Naive robust SWHT tests
add_boost_suite(test_swht_robust.cpp)
target_include_directories(test_swht_robust PRIVATE "${SRC_DIR}" "${INCLUDE_SRC_DIR}" "${CMAKE_BINARY_DIR}/include")
target_link_libraries(test_swht_robust PRIVATE m swht_kernel RandomSignal)
