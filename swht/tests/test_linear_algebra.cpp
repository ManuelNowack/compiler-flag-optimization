//===========================
// Linear algebra unit tests
//===========================


// Get Boost.test libraries
#define BOOST_TEST_MODULE test_linear_algebra
#include <boost/test/included/unit_test.hpp>

#include "linear_algebra.h"
#include <vector>
#include <list>

// Bit-matrix-vector product test
BOOST_AUTO_TEST_CASE( mvm ) {
    unsigned n = 12;
    bits P[] = {26, 11, 6, 26, 12, 14, 2, 27, 24, 27, 9, 30};
    bits f = 26;
    bit *output = new bit[n];
    bit solution[] = {1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1};
    matrix_dot_vector(P, f, n, output);
    for (unsigned i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(output[i], solution[i]);
    }
    delete[] output;
}

// Bit-matrix-transpose-vector product test
BOOST_AUTO_TEST_CASE( mtvm ) {
    unsigned n = 12;
    // unsigned b = 5;
    bits P[] = {18, 15, 5, 17, 16, 26, 28, 1, 0, 10, 21, 14};
    vbits f = {1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1};
    bits solution = 1ul;
    matrixT_dot_vector(P, f, n, output)
    BOOST_CHECK_EQUAL(solution, output);
}

// Vector-vector xor test
BOOST_AUTO_TEST_CASE( vvx ) {
    unsigned n = 12;
    bit v[] = {0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0};
    bit w[] = {1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0};
    bit solution[] = {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0};
    bit *output = new bit[n];
    vector_xor_vector(v, w, n, output)
    for (unsigned i = 0; i < n; i++) {
        BOOST_CHECK_EQUAL(output[i], solution[i]);
    }
    delete[] output;
}

// Vector-vector boolean inner product test
BOOST_AUTO_TEST_CASE( bool_inner ) {
    bit x[] = {0, 1, 1, 0, 1};
    bit y[] = {1, 0, 1, 1, 1};
    int out = bool_inner_product(x, y, 5);
    BOOST_CHECK_EQUAL(out, 0);
}
