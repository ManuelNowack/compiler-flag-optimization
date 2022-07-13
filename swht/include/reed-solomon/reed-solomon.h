//===============================================
// Reed-Solomon correction codes reduced encoder
//===============================================

// Disclaimer: The methods on this page are translations (with some small modifications)
// of the 'reedsolomon' Python module created by Tomer Filiba et al.
// The original code can be found here: https://github.com/tomerfiliba/reedsolomon


#ifndef REED_SOLOMON_H
#define REED_SOLOMON_H


#include <vector>

typedef unsigned long rs_int;


/** Encoder
 * Creates, holds and uses the log and anti-log lookup tables
 * to perform error correction with the Reed-Solomon algorithm.
 */
struct reed_solomon_encoder {

    std::vector<rs_int> gf_exp;         // anti-log LUT
    std::vector<rs_int> gf_log;         // log LUT
    rs_int field_characteristic;        // boundary value of the GF
    rs_int characteristic_exponent;     // exponent base 2 of the field characteristic
    unsigned long n_correction_symbols; // amount of numbers from the GF required to represent a syndrome

    /** Constructor/destructor
     * Initializes the log and anti-log lookup tables.
     */
    reed_solomon_encoder(rs_int characteristic_exponent, rs_int degree);
    ~reed_solomon_encoder();

    /** Primes number generator
     * Finds a prime number large enough for the given exponent.
     */
    rs_int find_prime_poly(rs_int characteristic_exponent);

    /** Calculate identity syndromes
     * Computes a measurement matrix whose columns are the syndromes polynomials
     * of an identity matrix's rows.
     */
    unsigned long calculate_all_syndromes(unsigned long n, unsigned *&all_syndromes);
    // unsigned long calculate_all_syndromes(unsigned long n, std::vector<std::vector<unsigned>> &all_syndromes);
    
    /** Find errors
     * Given a measurement (syndrome polynomial), retrieves the indexes of
     * the positive bits in the original time index.
     */
    unsigned long find_errors(const std::vector<unsigned> &syndrome, unsigned long n, std::vector<unsigned long> &errors_locations);
};

#endif
