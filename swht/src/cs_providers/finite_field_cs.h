//===================================
// Abstract finite field CS provider
//===================================

#ifndef FINITE_FIELD_CS_H
#define FINITE_FIELD_CS_H


#include "global_constants.h"


/** Finite field CS provider
 * Abstract type describing an arbitrary finite field compressive
 * sensing provider.
 */
struct finite_field_cs {

    unsigned long n, number_of_measurements;

    /** Constructor/destructor
     * Must be overrided
     * 
     * @param n: time-domain index size
     */
    finite_field_cs(unsigned long n, unsigned long m): n(n), number_of_measurements(m) {}
    virtual ~finite_field_cs() {}

    /** Update
     * Internally update the provider for a new iteration.
     */
    virtual void update() = 0;
    
    /** Measurement matrix
     * Returns the required vector from the measurement matrix.
     * 
     * @param i: index of the measurement matrix row to return
     * @return: a row of the measurement matrix (as a vector reference)
     */
    virtual const bit *measurement_matrix_row(unsigned long i) = 0;

    /** Low degree vector recovery
     * Performs the compressive sensing search on a given sequence of bits
     * to extract an original (supposedly low degree) frequency from it.
     * 
     * @param measurement: bit sequence to be turned into a frequency (inout)
     */
    virtual void recover_frequency(vbits &measurement) = 0;

    /** Inner product
     * Performs the inner product between the given frequency and the
     * given row of the measurement matrix.
     * 
     * @param frequency: lhs of the inner product
     * @param row: row of the measurement matrix to perform the inner product with (rhs)
     * @return: inner product as a binary (0-1) int
     */
    virtual bit inner_product(const vbits &frequency, unsigned long row) {
        const bit *matrix_row = measurement_matrix_row(row);
        bit out = 0u;
        for (unsigned long i = 0ul; i < n; i++) {
            out ^= frequency[i] && matrix_row[i];
        }
        return out;
    }
};

#endif
