//=====================
// Basic SWHT function
//=====================

#ifndef SWHT_BASIC_H
#define SWHT_BASIC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "global_constants.h"

#include <map>

/** SWHT
 * Non-robust SWHT core function.
 * 
 * @param signal: Observed signal
 * @param out: Output as a frequency-amplitude map
 * @param n: Set dimension
 * @param K: Sparsity
 * @param C: Bucket constant
 * @param ratio: Buckets reduction ratio
 * @param n_bins: (random binning - optional) number of bins for binary search
 * @param iterations: (random binning - optional) number of iterations for binary search
 * @param bin_ratio: (random binning - optional) bins reduction ratio for binary search
 * @param degree: (Reed-Solomon - optional) degree of the frequencies of the signal
 */
template <int cs_algorithm, typename ... Args>
int swht_basic(PyObject *signal, frequency_map &out, unsigned long n, unsigned long K, double C, double ratio, Args ... cs_args);

/** Robust SWHT
 * Robust SWHT core function with noise reduction for less sparse signals.
 * 
 * @param signal: Observed signal
 * @param out: Output as a frequency-amplitude map
 * @param n: Set dimension
 * @param K: Sparsity
 * @param C: Bucket constant
 * @param ratio: Buckets reduction ratio
 * @param robust_iterations: Number of random shifts to perform for frequency retrieval
 * @param n_bins: (random binning - optional) number of bins for binary search
 * @param iterations: (random binning - optional) number of iterations for binary search
 * @param bin_ratio: (random binning - optional) bins reduction ratio for binary search
 * @param degree: (Reed-Solomon - optional) degree of the frequencies of the signal
 */
template <int cs_algorithm, typename ... Args>
int swht_robust(PyObject *signal, frequency_map &out, unsigned long n, unsigned long K, double C, double ratio,
    unsigned long robust_iterations, Args ... cs_args);


#endif
