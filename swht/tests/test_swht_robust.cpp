//==================================
// Integration test for robust SWHT
//==================================

// Get Boost.test libraries
#define BOOST_TEST_MODULE test_swht_robust
#include <boost/test/included/unit_test.hpp>

#include "swht_kernel.h"
#include "cs_factory.h"
#include "build_info.h"
#include "random_signal.h"

#include <iostream>
#include <cstdlib>
#include <climits>


// Naive robust test
BOOST_AUTO_TEST_CASE( naive ) {

    // Initialize the Python interpreter
    Py_Initialize();

    // Parameters and signal
    unsigned long n = 22;
    unsigned long k = 15;
    unsigned long degree = 5;
    unsigned long iterations = 10;
    RandomSignal signal = RandomSignal(n, k, degree);

    // Display generated signal
    std::cout << "Signal: " << signal << std::endl;

    // Call SWHT function
    frequency_map out;
    swht_robust<NAIVE_CS>(&signal, out, n, k, 1.3, 1.4, iterations);

    // Display estimated signal
    std::cout << "Output: " << out << std::endl;

    // Compare output to original
    bool cmp = signal == out;
    if (cmp) {
        std::cout << "success" << std::endl;
    } else {
        std::cout << "failure" << std::endl;
    }

    // Exit
    Py_FinalizeEx();
    BOOST_CHECK(cmp);
}


// Random binning robust test
BOOST_AUTO_TEST_CASE( randbin ) {
    
    // Initialize the Python interpreter
    Py_Initialize();

    // Parameters and signal
    unsigned long n = 22;
    unsigned long k = 15;
    unsigned long degree = 5;
    unsigned long iterations = 10;
    unsigned long n_bins = degree * degree;
    unsigned long n_iterations = 2;
    double ratio = 2;
    RandomSignal signal = RandomSignal(n, k, degree);

    // Display generated signal
    std::cout << "Signal: " << signal << std::endl;

    // Call SWHT function
    frequency_map out;
    swht_robust<RANDOM_BINNING_CS>(&signal, out, n, k, 1.3, 1.4, iterations, n_bins, n_iterations, ratio);

    // Display estimated signal
    std::cout << "Output: " << out << std::endl;

    // Compare output to original
    bool cmp = signal == out;
    if (cmp) {
        std::cout << "success" << std::endl;
    } else {
        std::cout << "failure" << std::endl;
    }

    // Exit
    Py_FinalizeEx();
    BOOST_CHECK(cmp);
}


// Reed-Solomon robust test
BOOST_AUTO_TEST_CASE( reed_solomon ) {
    
    // Initialize the Python interpreter
    Py_Initialize();

    // Parameters and signal
    unsigned long n = 22;
    unsigned long k = 15;
    unsigned long degree = 5;
    unsigned long iterations = 10;
    RandomSignal signal = RandomSignal(n, k, degree);

    // Display generated signal
    std::cout << "Signal: " << signal << std::endl;

    // Call SWHT function
    frequency_map out;
    swht_robust<REED_SOLOMON_CS>(&signal, out, n, k, 1.3, 1.4, iterations, degree);

    // Display estimated signal
    std::cout << "Output: " << out << std::endl;

    // Compare output to original
    bool cmp = signal == out;
    if (cmp) {
        std::cout << "success" << std::endl;
    } else {
        std::cout << "failure" << std::endl;
    }

    // Exit
    Py_FinalizeEx();
    BOOST_CHECK(cmp);
}
