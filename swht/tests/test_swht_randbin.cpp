//====================================================
// Integration test for basic SHT with random binning
//====================================================

// Get Boost.test libraries
#define BOOST_TEST_MODULE test_swht_randbin
#include <boost/test/included/unit_test.hpp>

#include "swht_kernel.h"
#include "cs_factory.h"
#include "random_binning_cs.h"
#include "build_info.h"
#include "random_signal.h"

#include <iostream>
#include <cstdlib>


// CS class unit test
BOOST_AUTO_TEST_CASE( cs ) {
    unsigned long n = 10;
    unsigned long n_bins = 5;
    unsigned long iterations = 3;
    double ratio = 2;
    random_binning_cs cs = random_binning_cs(n, n_bins, iterations, ratio);
    cs.random_engine = std::default_random_engine(0);
    cs.update();
    vbits x = {1, 0, 0, 1, 0, 1, 0, 0, 1, 0};
    vbits z(cs.number_of_measurements);
    for (unsigned long i = 0; i < cs.number_of_measurements; i++) {
        z[i] = cs.inner_product(x, i);
    }
    cs.recover_frequency(z);
    BOOST_REQUIRE_EQUAL(x.size(), z.size());
    for (unsigned long i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(x[i], z[i]);
    }
}

// General integration test
BOOST_AUTO_TEST_CASE( integration ) {
    
    // Initialize the Python interpreter
    Py_Initialize();

    // Parameters and signal
    unsigned long n = 22;
    unsigned long k = 15;
    unsigned long degree = 5;
    unsigned long n_bins = degree * degree;
    unsigned long iterations = 2;
    double ratio = 2;
    RandomSignal signal = RandomSignal(n, k, degree);

    // Display generated signal
    std::cout << "Signal: " << signal << std::endl;

    // Call SWHT function
    frequency_map out;
    swht_basic<RANDOM_BINNING_CS>(&signal, out, n, k, 1.3, 1.4, n_bins, iterations, ratio);

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
