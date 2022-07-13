//=================================
// Integration test for basic SWHT
//=================================

// Get Boost.test libraries
#define BOOST_TEST_MODULE test_swht_naive
#include <boost/test/included/unit_test.hpp>

#include "swht_kernel.h"
#include "cs_factory.h"
#include "build_info.h"
#include "random_signal.h"

#include <iostream>
#include <cstdlib>
#include <climits>


// Quick CS test
BOOST_AUTO_TEST_CASE( cs ) {
    unsigned long n = 10;
    finite_field_cs *cs = get_finite_field_cs<NAIVE_CS>(n);
    vbits x = {1, 0, 0, 1, 0, 1, 0, 0, 1, 0};
    vbits z(cs->number_of_measurements);
    for (unsigned long i = 0; i < cs->number_of_measurements; i++) {
        z[i] = cs->inner_product(x, i);
    }
    cs->recover_frequency(z);
    BOOST_REQUIRE_EQUAL(x.size(), z.size());
    for (unsigned long i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(x[i], z[i]);
    }
    delete cs;
}


// General integration test
BOOST_AUTO_TEST_CASE( integration ) {
    
    // Initialize the Python interpreter
    Py_Initialize();

    // Parameters and signal
    unsigned long n = 22;
    unsigned long k = 15;
    unsigned long degree = 5;
    RandomSignal signal = RandomSignal(n, k, degree);

    // Display generated signal
    std::cout << "Signal: " << signal << std::endl;

    // Call SWHT function
    frequency_map out;
    swht_basic<NAIVE_CS>(&signal, out, n, k, 1.3, 1.4);

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
