//==========
// Naive CS
//==========


#ifndef NAIVE_CS_H
#define NAIVE_CS_H


#include "finite_field_cs.h"


/** Naive CS provider
 * Detects frequency through exhaustive search.
 */
struct naive_cs: public finite_field_cs {

    // Identity matrix
    bit *I;
    
    /** Constructor/destructor
     * Generates an identity matrix.
     * 
     * @param n: time-domain index size
     */
    naive_cs(unsigned long n);
    ~naive_cs();

    /** Update
     * This is not needed for this provider (does nothing).
     */
    void update();

    /** Measurement matrix
     * Returns the required vector from the measurement matrix.
     * 
     * @param i: index of the measurement matrix row to return
     * @return: a row of the measurement matrix (as a vector reference)
     */
    const bit *measurement_matrix_row(unsigned long i);

    /** Low degree vector recovery
     * This is not necesary for this algorithm (returns identity).
     * 
     * @param measurement: bit sequence returned as-is
     */
    void recover_frequency(vbits &measurement);

    /** Inner product
     * Performs the inner product between the given frequency and the
     * given row of the measurement matrix.
     * Note: this overridden version accelerates inner product with identity vectors.
     * 
     * @param frequency: lhs of the inner product
     * @param row: row of the measurement matrix to perform the inner product with (rhs)
     * @return: inner product as a binary (0-1) int
     */
    bit inner_product(const vbits &frequency, unsigned long row);
};

#endif
