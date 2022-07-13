//======================================================
// Reed-Solomon correction codes reduced encoder (impl)
//======================================================

// Disclaimer: The methods on this page are translations (with some small modifications)
// of the 'reedsolomon' Python module created by Tomer Filiba et al.
// The original code can be found here: https://github.com/tomerfiliba/reedsolomon


#include "reed-solomon.h"
#include <vector>
#include <algorithm>
#include <cmath>


/** Fast carry-less multiplication
 * Performs a fast carry-less multiplication within GF(2^characteristic_exponent).
 * 
 * Effectively does:
 * if (x == 0 || y == 0) return 0;
 * return gf_exp[(gf_log[x] + gf_log[y]) % field_characteristic]
 */
#define gf_mult(x, y) (x && y ? gf_exp[gf_log[x] + gf_log[y]] : 0ul)
// #define gf_mult(x, y) (x * y ? gf_exp[(gf_log[x] + gf_log[y]) % field_characteristic] : 0)

/** Fast carry-less powers of 2
 * Raises 2 to the given power within GF(2^characteristic_exponent).
 */
#define gf_pow(a) gf_exp[a]
// #define gf_pow(a) (gf_exp[(gf_log[2ul] * a) % field_characteristic])


// /** Slow carry-less multiplication
//  * Performs a carry-less mutliplication within GF(2^characteristic_exponent) without
//  * a lookup table (can be used to compute it).
//  */
// inline rs_int gf_mult_noLUT(rs_int x, rs_int y, rs_int prime, rs_int field_charcteristic_mask) {
//     rs_int r = 0;
//     while (y) {
//         if (y & 1) r ^= x;
//         y >>= 1;
//         x <<= 1;
//         if (x & field_charcteristic_mask) x ^= prime;
//     }
//     return r;
// }


/** Primes number generator
 * Finds a prime number large enough for the given exponent.
 */
rs_int reed_solomon_encoder::find_prime_poly(rs_int characteristic_exponent) {
    rs_int field_charcteristic_next = (1ul << (characteristic_exponent + 1ul)) - 1ul;
    std::vector<bool> sieve(field_charcteristic_next / 2ul, true);
    for (unsigned long i = 3ul; i < std::floor(std::sqrt(field_charcteristic_next)) + 1; i += 2ul) {
        if (sieve[i / 2ul]) {
            for (unsigned long j = i * i / 2ul; j < sieve.size(); j += i) {
                sieve[j] = false;
            }
        }
    }
    std::vector<rs_int> prime_candidates;
    if (field_characteristic < 2ul) prime_candidates.push_back(2ul);
    for (unsigned long i = 1ul; i < field_charcteristic_next / 2ul; i++) {
        rs_int candidate = 2ul * i + 1ul;
        if (sieve[i] && candidate > field_characteristic)
            prime_candidates.push_back(candidate);
    }

    // Select correct candidate
    for (const rs_int &prime: prime_candidates) {
        std::vector<int> seen(field_characteristic + 1ul, 0);
        bool conflict = false;
        rs_int x = 1ul;
        for (unsigned long i = 0ul; i < field_characteristic; i++) {
            // x = gf_mult_noLUT(x, 2ul, prime, field_characteristic + 1);
            x <<= 1ul;
            if (x & (field_characteristic + 1ul)) x ^= prime;
            if (x > field_characteristic || seen[x] == 1) {
                conflict = true;
                break;
            } else {
                seen[x] = 1;
            }
        }
        if (!conflict) return prime;
    }
    return 0ul;
}


/** Constructor/destructor
 * Initializes the log and anti-log lookup tables.
 */
reed_solomon_encoder::reed_solomon_encoder(rs_int characteristic_exponent, rs_int degree):
    characteristic_exponent(characteristic_exponent) {

    // Ready attributes
    field_characteristic = (1ul << characteristic_exponent) - 1ul;
    gf_exp = std::vector<rs_int>(field_characteristic * 2ul);
    gf_log = std::vector<rs_int>(field_characteristic + 1ul, 0ul);
    n_correction_symbols = 2ul * degree;

    // Get prime number for multiplications
    rs_int prime = find_prime_poly(characteristic_exponent);
    
    // Fill log and anti-log tables
    rs_int x = 1ul;
    for (unsigned long i = 0ul; i < field_characteristic; i++) {
        gf_exp[i] = x;
        gf_log[x] = i;
        // x = gf_mult_noLUT(x, 2ul, prime, field_characteristic + 1);
        x <<= 1ul;
        if (x & (field_characteristic + 1ul)) x ^= prime;
    }

    // Double the anti-log table size by duplicating its values (avoids later modulos)
    for (unsigned long i = 0ul; i < field_characteristic; i++) {
        gf_exp[i + field_characteristic] = gf_exp[i];
    }
}

reed_solomon_encoder::~reed_solomon_encoder() {}


// TODO: There is a lot of potential speedup to be had here (remove modulos, eliminate codeword, etc.)
/** Calculate identity syndromes
 * Computes a measurement matrix whose columns are the syndromes polynomials
 * of an identity matrix's rows.
 */
unsigned long reed_solomon_encoder::calculate_all_syndromes(unsigned long n, unsigned *&all_syndromes) {
    unsigned long n_correction_symbols_bits = characteristic_exponent * n_correction_symbols;
    all_syndromes = new unsigned[n_correction_symbols_bits * n];
    rs_int *codeword = new rs_int[n]();
    for (unsigned long i = 0ul; i < n; i++) {
        codeword[i] = 1ul;
        for (unsigned long j = 0ul, j_p = 0ul; j < n_correction_symbols; j++, j_p += characteristic_exponent) {
            rs_int power = gf_pow(j);
            rs_int syndrome = codeword[0ul];
            for (unsigned long k = 1ul; k < n; k++) {
                syndrome = gf_mult(syndrome, power) ^ codeword[k];
            }
            for (unsigned long k = 0ul; k < characteristic_exponent; k++) {
                all_syndromes[(j_p + k) * n + i] = (syndrome >> k) & 1ul;
            }
        }
        codeword[i] = 0ul;
    }
    delete[] codeword;
    return n_correction_symbols_bits;
}


/** Find errors
 * Given a measurement (syndrome polynomial), retrieves the indexes of
 * the positive bits in the original time index.
 */
unsigned long reed_solomon_encoder::find_errors(const std::vector<unsigned> &syndrome, unsigned long n, std::vector<size_t> &errors_locations) {
    
    // Turn syndrome from bit representation to GF ints representation
    std::vector<rs_int> rs_syndrome(n_correction_symbols, 0ul);
    for (unsigned long i = 0ul, i_p = 0ul; i < n_correction_symbols; i++, i_p += characteristic_exponent) {
        for (unsigned long j = 0ul; j < characteristic_exponent; j++) {
            rs_syndrome[i] += syndrome[i_p + j] << j;
        }
    }

    // Find error locator and evaluator polynomials with Berlekamp-Massey algorithm
    std::vector<rs_int> error_locator = {1ul};
    std::vector<rs_int> old_locator = {1ul};
    for (unsigned long i = 0ul; i < n_correction_symbols; i++) {
        rs_int delta = rs_syndrome[i];
        size_t error_size = error_locator.size();
        for (unsigned long j = 1ul; j < error_size; j++) {
            rs_int err_loc = error_locator[error_size - (j + 1ul)];
            rs_int synd = rs_syndrome[i - j];
            delta ^= gf_mult(err_loc, synd);
        }
        old_locator.push_back(0ul);
        if (delta) {
            if (old_locator.size() > error_size) {
                rs_int inv_delta = gf_exp[field_characteristic - gf_log[delta]];
                std::vector<rs_int> new_locator(old_locator);
                for (rs_int &x: new_locator) x = gf_mult(x, delta);
                old_locator = std::move(error_locator);
                for (rs_int &x: old_locator) x = gf_mult(x, inv_delta);
                error_locator = std::move(new_locator);
            }
            size_t old_size = old_locator.size();
            error_size = error_locator.size();
            size_t max_size = std::max(error_size, old_size);
            std::vector<rs_int> error_locator_tmp(max_size, 0ul);
            std::move(error_locator.begin(), error_locator.end(), error_locator_tmp.begin() + (max_size - error_size));
            for (size_t j = 0ul; j < old_size; j++) {
                error_locator_tmp[j + (max_size - old_size)] ^= gf_mult(old_locator[j], delta);
            }
            error_locator = std::move(error_locator_tmp);
        }
    }
    
    // Drop leading 0's then reverse the vector
    std::vector<rs_int> cropped_error_locator;
    if (!error_locator.empty()) {
        auto first_non_zero = error_locator.rend() - 1;
        while (!*first_non_zero) first_non_zero--;
        cropped_error_locator = std::vector<rs_int>(error_locator.rbegin(), first_non_zero + 1);
    }

    // Find the roots by bruteforce search
    for (size_t i = 0ul; i < n; i++) {
        rs_int power = gf_pow(i);
        rs_int evaluation = cropped_error_locator[0ul];
        for (size_t j = 1ul; j < cropped_error_locator.size(); j++)
            evaluation = gf_mult(evaluation, power) ^ cropped_error_locator[j];
        if (!evaluation) errors_locations.push_back(n - 1ul - i);
    }
    return 0ul;
}
