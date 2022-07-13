//=====================
// Basic SWHT function
//=====================


#include "swht_kernel.h"
#include "Hasher.h"
#include "cs_factory.h"
#include "linear_algebra.h"

#include "build_info.h"
#ifdef PROFILING_BUILD
    #include "profiling.h"
    DURATION_READY
#endif

#include "fastwht/hadamardKernel.h"

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>


#define TOLERANCE 0.005


struct amplitude_iterator;


/** SWHT
 * Non-robust SWHT core function.
 * 
 * @param signal: Observed signal
 * @param current_estimate: Output as a frequency-amplitude map
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

/** Amplitude iterator
 * A simple iterator to go through an array of double.
 */
struct amplitude_iterator {

    // Relevant types
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = double;
    using pointer           = value_type *;
    using reference         = value_type &;

    // Constructors/destructors
    amplitude_iterator(value_type *ptr): m_ptr(ptr) {}

    // Reference operators
    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }
    reference operator[](size_t i) { return m_ptr[i]; }

    // Forward operators
    amplitude_iterator &operator++() { m_ptr++; return *this; }
    amplitude_iterator operator++(int) { amplitude_iterator tmp = *this; ++(*this); return tmp; }
    amplitude_iterator &operator+=(difference_type n) { m_ptr += n; return *this; }
    amplitude_iterator operator+(difference_type n) { return amplitude_iterator(m_ptr + n); }
    friend amplitude_iterator operator+(difference_type n, const amplitude_iterator &a) { return amplitude_iterator(a.m_ptr + n); }

    // Backward operators
    amplitude_iterator &operator--() { m_ptr--; return *this; }
    amplitude_iterator operator--(int) { amplitude_iterator tmp = *this; --(*this); return tmp; }
    amplitude_iterator &operator-=(difference_type n) { m_ptr -= n; return *this; }
    amplitude_iterator operator-(difference_type n) { return amplitude_iterator(m_ptr - n); }

    // Relational operators
    bool operator==(const amplitude_iterator &b) { return m_ptr == b.m_ptr; };
    bool operator!=(const amplitude_iterator &b) { return m_ptr != b.m_ptr; };
    difference_type operator-(const amplitude_iterator &b) { return m_ptr - b.m_ptr; }
    bool operator<(const amplitude_iterator &b) { return m_ptr < b.m_ptr; }
    bool operator>(const amplitude_iterator &b) { return m_ptr > b.m_ptr; }
    bool operator<=(const amplitude_iterator &b) { return m_ptr <= b.m_ptr; }
    bool operator>=(const amplitude_iterator &b) { return m_ptr >= b.m_ptr; }
    
private:
    double *m_ptr;
};
template <int cs_algorithm, typename ... Args> 
int swht_robust(PyObject *signal, frequency_map &current_estimate, unsigned long n, unsigned long K, double C, double ratio,
unsigned long robust_interations, Args ... cs_args) {
    
    // Base parameters
    #ifdef PROFILING_BUILD
        SECTION_START(init)
    #endif
    unsigned long B = K * C;                                                // number of buckets we are hashing into (to be rounded up)
    unsigned long b = ceil(log2(B));                                        // index bit-size for the buckets
    unsigned long T = (unsigned long)(log(B) / log(ratio)) - 1ul;           // number of iterations to perform
    finite_field_cs *cs_provider = get_finite_field_cs<cs_algorithm>(n, cs_args...);    // finite field cs provider
    auto prng = std::bind(std::uniform_int_distribution<unsigned>(0u, 1u),
        std::default_random_engine(std::random_device()()));                // random bits generator
    
    bit *random_shifts = new bit[robust_interations * n];
    bit *xored_vector = new bit[n];
    #ifdef PROFILING_BUILD
        SECTION_END(init)
    #endif

    for (unsigned long i = 0ul; i < T; i++) {
        
        // Create hashing processor
        #ifdef PROFILING_BUILD
            SECTION_START(hasher)
        #endif
        Hasher hash(n, b, signal);
        cs_provider->update();
        unsigned long N = cs_provider->number_of_measurements;
        #ifdef PROFILING_BUILD
            SECTION_END(hasher)
        #endif


        // Current frequency estimate hashing
        //===================================
        // Map all the current frequency-amplitude estimates to the bucket
        // corresponding to their hashed frequency for later subtraction
        // from the hased and transformed buckets.

        #ifdef PROFILING_BUILD
            SECTION_START(current_hash)
        #endif
        std::vector<std::pair<vbits, double>> *hashed_estimate = new std::vector<std::pair<vbits, double>>[hash.B]();
        for (auto it = current_estimate.begin(); it != current_estimate.end(); it++) {
            bits hashed_frequency = hash.frequency_hash(it->first);
            hashed_estimate[hashed_frequency].push_back(*it);
        }
        #ifdef PROFILING_BUILD
            SECTION_END(current_hash)
        #endif


        // Robust iterations
        //==================
        // Prepare multiple sampling and frequency retrieval attempts with different random
        // shifts. The most stable results will be kept for robustness.

        #ifdef PROFILING_BUILD
            SECTION_START(robust_init)
        #endif

        // Ready data structures
        double    *reference_signal                = new double[hash.B];
        double    *hashed_WHT                      = new double[N * hash.B];
        vbits     *measurements_dictionary         = new vbits[hash.B];
        double    *amplitudes_dictionary           = new double[hash.B * robust_interations];
        unsigned  *successful_tries                = new unsigned[hash.B]();
        const bit **successful_tries_random_shifts = new const bit*[hash.B * robust_interations];
        for (bits bucket = 0ul; bucket < hash.B; bucket++) {
            measurements_dictionary[bucket] = vbits(N, 0u);
        }
        for (unsigned long j = 0ul; j < robust_interations; j++)
            for (unsigned long k = 0ul; k < n; k++)
                random_shifts[j * n + k] = prng();
        
        #ifdef PROFILING_BUILD
            SECTION_END(robust_init)
        #endif

        // for (const vbits &random_shift: random_shifts) {
        for (unsigned long shift_i = 0ul; shift_i < robust_interations; shift_i++) {
            const bit *random_shift = random_shifts + shift_i * n;


            // Source signal sampling
            //=======================
            // Sample the source signal (time-domain) then transform it (frequency-domain)
            // without shifting (reference signal) and with shifts generated by the cs
            // provider for later comparison.

            // #ifdef PROFILING_BUILD
            //     SECTION_START(sampling)
            // #endif

            // Compute reference signal transformed hash (base shift = 0)
            hash.time_hash(random_shift, reference_signal);
            #ifdef PROFILING_BUILD
                SECTION_START(transform)
            #endif
            fwhtKernelOrdinary(hash.B, reference_signal);
            for (unsigned long j = 0ul; j < hash.B; j++) reference_signal[j] /= hash.B;
            #ifdef PROFILING_BUILD
                SECTION_END(transform)
            #endif

            // Compute all shifted signal transformed hashes according to detection algorithm
            for (size_t k = 0ul; k < N; k++) {
                double *hashed_signal = hashed_WHT + k * hash.B;
                const bit *shift = cs_provider->measurement_matrix_row(k);
                vector_xor_vector(shift, random_shift, n, xored_vector)
                hash.time_hash(xored_vector, hashed_signal);
                #ifdef PROFILING_BUILD
                    SECTION_START(transform)
                #endif
                fwhtKernelOrdinary(hash.B, hashed_signal);
                for (unsigned long j = 0ul; j < hash.B; j++) hashed_signal[j] /= hash.B;
                #ifdef PROFILING_BUILD
                    SECTION_END(transform)
                #endif
            }

            // #ifdef PROFILING_BUILD
            //     SECTION_END(sampling)
            // #endif


            // Detect and extract frequencies from buckets
            //============================================
            // For each bucket of the reference signal, find a target (non-hashed) frequency
            // to map it to, while ensuring maximum sparsity through subtraction of previously
            // extracted frequencies.

            #ifdef PROFILING_BUILD
                SECTION_START(detection)
            #endif

            // Iterate through each bucket (frequency hash)
            for (bits bucket = 0ul; bucket < hash.B; bucket++) {

                // Subtract current estimate from hashed reference signal to increase sparsity
                for (std::pair<vbits, double> &X: hashed_estimate[bucket]) {
                    if (bool_inner_product(X.first.data(), random_shift, n)) {
                        reference_signal[bucket] += X.second;
                    } else {
                        reference_signal[bucket] -= X.second;
                    }
                }
                if (std::abs(reference_signal[bucket] - 0.0) < TOLERANCE)
                    continue; // ignore near-zero amplitudes!

                // If it is still a strong amplitude, count it as a hit
                successful_tries_random_shifts[bucket * robust_interations + successful_tries[bucket]] = random_shift;

                // Subtract current estimate from hashed shifted WHT to increase sparsity
                if (!hashed_estimate[bucket].empty()) {
                    for (unsigned long j = 0ul; j < N; j++) {
                        const bit *shift = cs_provider->measurement_matrix_row(j);
                        vector_xor_vector(shift, random_shift, n, xored_vector)
                        for (std::pair<vbits, double> &X: hashed_estimate[bucket]) {
                            if (bool_inner_product(X.first.data(), xored_vector, n)) {
                                hashed_WHT[j * hash.B + bucket] += X.second;
                            } else {
                                hashed_WHT[j * hash.B + bucket] -= X.second;
                            }
                        }
                    }
                }

                // Record the number of sign-flipping per index
                for (unsigned long j = 0ul; j < N; j++) {
                    if (hashed_WHT[j * hash.B + bucket] * reference_signal[bucket] < 0.0)
                        measurements_dictionary[bucket][j]++;
                }
                amplitudes_dictionary[bucket * robust_interations + successful_tries[bucket]++] =
                    reference_signal[bucket];
            }

            #ifdef PROFILING_BUILD
                SECTION_END(detection)
            #endif
        }

        // Detection arrays cleanup
        delete[] hashed_estimate;
        delete[] reference_signal;
        delete[] hashed_WHT;


        // Robust frequency retrieval
        //===========================
        // Rebuild the measurements with the most stable index values then retrieve the
        // corresponding frequency (using CS) and amplitude (using median).

        #ifdef PROFILING_BUILD
            SECTION_START(robust_retrieval)
        #endif

        // Rebuild the frequencies that got per-index at least half the hits in their buckets
        frequency_map detected_frequency;
        for (bits bucket = 0ul; bucket < hash.B; bucket++) {
            if (!successful_tries[bucket]) continue;

            // Compute bit-wise measurement based on which indexes have gotten at least half of the hits
            vbits measurement(N, 0u);
            for (unsigned long j = 0ul; j < N; j++) {
                if (measurements_dictionary[bucket][j] > successful_tries[bucket] / 2.0) {
                    measurement[j] = 1u;
                // } else {
                //     measurement[j] = 0;
                }
            }

            // Recover the actual frequency from the measurement using the cs recovery algorithm and check its validity
            cs_provider->recover_frequency(measurement);
            if (hash.frequency_hash(measurement) != bucket)
                continue; // ignore invalid frequencies

            // Flip sign of amplitudes
            for (unsigned long j = 0ul; j < successful_tries[bucket]; j++) {
                if (bool_inner_product(
                    measurement.data(), successful_tries_random_shifts[bucket * robust_interations + j], n)
                ) {
                    amplitudes_dictionary[bucket * robust_interations + j] =
                        -amplitudes_dictionary[bucket * robust_interations + j];
                }
            }

            // Assign the median of the amplitudes to the recovered frequency
            unsigned long amplitude_bucket_size = successful_tries[bucket];
            size_t start_point = bucket * robust_interations;
            size_t end_point = start_point + amplitude_bucket_size;
            size_t mid_point = start_point + amplitude_bucket_size / 2ul;
            std::nth_element(
                amplitude_iterator(amplitudes_dictionary + start_point),
                amplitude_iterator(amplitudes_dictionary + mid_point),
                amplitude_iterator(amplitudes_dictionary + end_point)
            );
            if (amplitude_bucket_size & 1ul) {
                detected_frequency[measurement] = amplitudes_dictionary[mid_point];
            } else {
                double low_median = amplitudes_dictionary[mid_point];
                size_t high_mid_point = mid_point + 1ul;
                std::nth_element(
                    amplitude_iterator(amplitudes_dictionary + start_point),
                    amplitude_iterator(amplitudes_dictionary + high_mid_point),
                    amplitude_iterator(amplitudes_dictionary + end_point)
                );
                detected_frequency[measurement] = (low_median + amplitudes_dictionary[high_mid_point]) / 2.0;
            }
        }

        // Post-retrieval arrays cleanup
        delete[] measurements_dictionary;
        delete[] amplitudes_dictionary;
        delete[] successful_tries;
        delete[] successful_tries_random_shifts;

        #ifdef PROFILING_BUILD
            SECTION_END(robust_retrieval)
        #endif


        // Iterative updates
        //==================
        // Add detected frequencies to the current estimate (or sum them if they already exist)
        // and purge estimated frequencies that have become too close to zero.

        #ifdef PROFILING_BUILD
            SECTION_START(updates)
        #endif

        for (auto &&frequency: detected_frequency) {
            auto match = current_estimate.find(frequency.first);
            if (match != current_estimate.end()) {
                match->second += frequency.second;
                if (std::abs(match->second - 0.0) < TOLERANCE)
                    current_estimate.erase(match);
            } else {
                current_estimate[frequency.first] = frequency.second;
            }
        }

        // Buckets sizes reduction
        B = std::ceil(B / ratio);
        b = ceil(log2(B));

        #ifdef PROFILING_BUILD
            SECTION_END(updates)
        #endif
    }

    // Final cleanup
    delete[] random_shifts;
    delete[] xored_vector;
    
    // Return estimate
    delete cs_provider;
    return 0;
}

// Per-algorithm instantiations
template int swht_robust<NAIVE_CS>(PyObject *signal, frequency_map &out, unsigned long n, unsigned long K, double C, double ratio,
    unsigned long robust_iterations);
template int swht_robust<RANDOM_BINNING_CS, unsigned long, unsigned long, double>(
    PyObject *signal, frequency_map &out, unsigned long n, unsigned long K, double C, double ratio, unsigned long robust_iterations,
    unsigned long cs_bins, unsigned long cs_iterations, double cs_ratio);
template int swht_robust<REED_SOLOMON_CS, unsigned long>(PyObject *signal, frequency_map &out, unsigned long n, unsigned long K,
    double C, double ratio, unsigned long robust_iterations, unsigned long degree);


