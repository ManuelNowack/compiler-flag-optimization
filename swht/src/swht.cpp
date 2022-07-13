//=====================================
// swht C++ interface (implementation)
//=====================================

#include "swht.h"
#include "build_info.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>


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
frequency_map swht(PyObject *signal, std::string cs_algorithm, unsigned long n,
    unsigned long K, unsigned long robust_iterations, double C, double ratio,
    unsigned long cs_bins, unsigned long cs_iterations, double cs_ratio,
    unsigned long degree) {

    // Check signal
    if (!PyCallable_Check(signal)) {
        throw std::invalid_argument("Signal must be callable.");
    }

    // Check cs algorithm
    std::transform(cs_algorithm.begin(), cs_algorithm.end(), cs_algorithm.begin(), ::tolower);
    auto cs_algo_match = cs_algorithms.find(cs_algorithm);
    if (cs_algo_match == cs_algorithms.end()) {
        throw std::invalid_argument("Unrecognized CS algorithm (" + cs_algorithm + ").");
    }
    int cs_algo_num = cs_algo_match->second;

    // Check n
    if (n == 0ul) {
        throw std::invalid_argument("n cannot be 0, must be strictly positive.");
    }

    // Check random binning
    if (cs_algo_num == 1) {
        
        // bins
        if (cs_bins == 0ul) {
            if (degree == 0ul) {
                throw std::invalid_argument("Random binning requires either a number of bins or a degree (none given).");
            }
            cs_bins = degree * degree;
        }

        // iterations
        if (cs_iterations == 0ul) {
            throw std::invalid_argument("Random binning requires at least 1 iteration (0 given).");
        }

        // ratio
        if (cs_ratio <= 1.0) {
            throw std::invalid_argument("Random binning requires ratio strictly greater than 1");
        }
    }

    // Check Reed-Solomon
    else if (cs_algo_num == 2) {
        if (degree == 0ul) {
            throw std::invalid_argument("Must specify degree (higher than 0) for Reed-Solomon.");
        }
#ifndef BENCHMARKING_BUILD
        if (degree > (n / 2ul)) {
            throw std::invalid_argument("For degrees higher than n/2 use naive instead.");
        }
#endif
    }

    // Check K
#ifndef BENCHMARKING_BUILD
    if (K > (1ul << n) || (degree > 0ul && K > std::pow(n, degree))) {
        throw std::invalid_argument("K is too high, consider using a standard WHT for non-sparse signals.");
    }
#endif

    // Check robust iterations
    if (robust_iterations == 0ul) {
        throw std::invalid_argument("Must perform at least one iteration for exact algorithm, more for robust (0 given).");
    }

    // Check C
    if (n < ceil(log2(K * C))) {
        throw std::invalid_argument("Bucket space dimension larger than finite field (n < b), consider reducing C.");
    }

    // Check ratio
    if (ratio <= 1.0) {
        throw std::invalid_argument("Ratio must strictly higher than 1.");
    }

    // Check degree
    if (degree > n) {
        throw std::invalid_argument("Degree cannot be higher than finite field dimension (d > n given).");
    }

    // Call swht
    frequency_map out;
    bool use_basic = robust_iterations == 1ul;
    switch (cs_algo_num) {
    case 0:
        if (use_basic) swht_basic<0>(signal, out, n, K, C, ratio);
        else swht_robust<0>(signal, out, n, K, C, ratio, robust_iterations);
        break;
    case 1:
        if (use_basic) swht_basic<1>(signal, out, n, K, C, ratio, cs_bins, cs_iterations, cs_ratio);
        else swht_robust<1>(signal, out, n, K, C, ratio, robust_iterations, cs_bins, cs_iterations, cs_ratio);
        break;
    case 2:
        if (use_basic) swht_basic<2>(signal, out, n, K, C, ratio, degree);
        else swht_robust<2>(signal, out, n, K, C, ratio, robust_iterations, degree);
        break;
    default:
        throw std::runtime_error("This should not happen (weird cs algorithm).");
    }

    return out;
}
