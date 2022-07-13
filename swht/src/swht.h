//====================
// swht C++ interface
//====================

#ifndef SWHT_H
#define SWHT_H

#include "swht_kernel.h"

#include <unordered_map>
#include <string>


/** CS algorithms
 * Record of the available compressive sensing algorithms.
 */
const static std::unordered_map<std::string, int> cs_algorithms = {
    {"naive",           0},
    {"random binning",  1},
    {"reed-solomon",    2}
};


/** SWHT
 * General C++ interface for the SWHT transform.
 * 
 * @param signal: Input signal
 * @param n: Set dimension
 * @param K: Sparsity
 * @param C: (optional) Bucket constant
 * @param ratio: (optional) Buckets reduction ratio
 * @param robust_iterations: (optional) Number of random shifts to perform for frequency retrieval
 * @param n_bins: (random binning - optional) number of bins for binary search
 * @param iterations: (random binning - optional) number of iterations for binary search
 * @param bin_ratio: (random binning - optional) bins reduction ratio for binary search
 * @param degree: (Reed-Solomon - optional) degree of the frequencies of the signal
 * 
 * @return: A vector to double map of the signal frequencies
 */
frequency_map swht(
    PyObject *signal, std::string cs_algorithm,                 // target
    unsigned long n, unsigned long K,                           // size
    unsigned long robust_iterations = 1ul,                      // robust option
    double C = 1.3, double ratio = 1.4,                         // buckets
    unsigned long cs_bins = 0ul,                                // random binning
    unsigned long cs_iterations = 1ul, double cs_ratio = 2.0,   // random binning optional
    unsigned long degree = 0ul                                  // Reed-Solomon and random binning
);

#endif
