//=======================
// Hasher utility (impl)
//=======================


#include "Hasher.h"
#include "linear_algebra.h"

#include "build_info.h"

#ifdef PROFILING_BUILD
    #include "profiling.h"
    DURATION_READY
#endif

#include <cmath>
#include <random>


/** Constructor
 * Readies the parameters and the random permutation binary matrix P.
 * 
 * @param n: original index size
 * @param b: bucket index size
 * @param signal: signal to query
 */
Hasher::Hasher(unsigned long n, unsigned long b, PyObject *signal) : n(n), b(b), signal(signal) {
    
    // Ready attributes
    B = 1ul << b;
    P = new bits[n];
    hashed_indexes = new bit[n * B];

    // Set seed
#ifdef BENCHMARKING_BUILD
    std::default_random_engine engine(0);
#else
    std::default_random_engine engine({std::random_device()()});
#endif
    std::uniform_int_distribution<bits> prng(0ul, B - 1ul);

    // Build hashing matrix
    for (size_t i = 0ul; i < n; i++) {
        P[i] = prng(engine);
    }

    // Ready signal querying
    py_args = PyTuple_Pack(1l, Py_None);
    py_args_content = ((PyTupleObject *) py_args)->ob_item;
    signal_call = signal->ob_type->tp_call;
    py_bits[0ul] = _PyLong_Zero;
    py_bits[1ul] = _PyLong_One;

    // Precompute hashed indexes
    for (bits bucket = 0ul; bucket < B; bucket++) {
        bit *hashed_index = hashed_indexes + bucket * n;
        matrix_dot_vector(P, bucket, n, hashed_index);
    }
}

/** Destructor
 * Cleans up the permutation matrix.
 */
Hasher::~Hasher() {
    delete[] P;
    delete[] hashed_indexes;
    Py_DECREF(py_args);
}

/** Time hash
 * Gives for each possible hashed index the value yielded by the signal at their
 * unhashed and shifted counterpart.
 * 
 * @param shift: shift vector to apply to the index
 * @param out: mapping of the signal responses to their corresponding bin
 */
void Hasher::time_hash(const bit *shift, double *out) {

    // auto match = cache.find(shift);
    // if (match != cache.end()) {
    //     out = match->second;
    //     return;
    // }

    #ifdef PROFILING_BUILD
        COUNT_SAMPLE(B)
    #endif

    for (bits t = 0ul; t < B; t++) {

        #ifdef PROFILING_BUILD
            SECTION_START(time_index)
        #endif

        Py_DECREF(*py_args_content);
        *py_args_content = PyTuple_New(n);
        PyObject **py_index_items = ((PyTupleObject *) *py_args_content)->ob_item;

        bit *hashed_index = hashed_indexes + t * n;
        for (unsigned long i = 0ul; i < n; i++) {
            bit index_i = hashed_index[i] != shift[i];
            Py_INCREF(py_bits[index_i]);
            py_index_items[i] = py_bits[index_i];
        }

        #ifdef PROFILING_BUILD
            SECTION_END(time_index)
            SECTION_START(query)
        #endif

        // Perform signal query
        PyObject *result = signal_call(signal, py_args, NULL);
        out[t] = PyFloat_AsDouble(result);
        Py_DECREF(result);

        #ifdef PROFILING_BUILD
            SECTION_END(query)
        #endif
    }
    // cache[shift] = out;
}

/** Frequency hash
 * Computes the hash of an index.
 * 
 * @param frequency: frequency to be hashed
 * @param out: hashed frequency (out)
 */
bits Hasher::frequency_hash(const vbits &frequency) {
    matrixT_dot_vector(P, frequency, n, freq_hash)
    return freq_hash;
}
