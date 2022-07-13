//================
// Hasher utility
//================

#ifndef HASHER_H
#define HASHER_H


#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "global_constants.h"


/** Hashing provider
 * Object that generates a random binary hashing matrix and applies
 * time and frequency hashing while maintaining a result cache.
 */
struct Hasher {
    unsigned long n, b, B;
    bits *P;
    bit *hashed_indexes;
    PyObject *signal;
    ternaryfunc signal_call;
    PyObject *py_args;
    PyObject **py_args_content;
    PyObject *py_bits[2ul];
    // std::map<bits, std::map<bits, double>> cache;

    /** Constructor
     * Readies the parameters and the random permutation binary matrix P.
     * 
     * @param n: original index size
     * @param b: bucket index size
     * @param signal: signal to query
     */
    Hasher(unsigned long n, unsigned long b, PyObject *signal);
    ~Hasher();

    /** Time hash
     * Gives for each possible hashed index the value yielded by the signal at their
     * unhashed and shifted counterpart.
     * 
     * @param shift: shift vector to apply to the index
     * @param out: mapping of the signal responses to their corresponding bin
     */
    void time_hash(const bit *shift, double *out);

    /** Frequency hash
     * Computes the hash of an index.
     * 
     * @param frequency: frequency to be hashed
     * @param out: hashed frequency (out)
     */
    bits frequency_hash(const vbits &frequency);
};

#endif
