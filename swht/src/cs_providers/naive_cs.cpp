//===========================
// Naive CS (implementation)
//===========================


#include "naive_cs.h"


/** Constructor/destructor
 * Generates an identity matrix.
 * 
 * @param n: time-domain index size
 */
naive_cs::naive_cs(unsigned long n): finite_field_cs(n, n) {
    I = new bit[n * n]();
    for (unsigned long i = 0ul; i < n; i++) {
        I[i * n + i] = 1u;
    }
}
naive_cs::~naive_cs() {
    delete[] I;
}

/** Update
 * This is not needed for this provider (does nothing).
 */
void naive_cs::update() {}

/** Measurement matrix
 * Returns the required vector from the measurement matrix.
 * 
 * @param i: index of the measurement matrix row to return
 * @return: a row of the measurement matrix (as a vector reference)
 */
const bit *naive_cs::measurement_matrix_row(unsigned long i) {
    return I + i * n;
}

/** Low degree vector recovery
 * This is not necesary for this algorithm (returns identity).
 * 
 * @param measurement: bit sequence returned as-is
 */
void naive_cs::recover_frequency(vbits &measurement) {
    (void) measurement; // ignored
}

/** Inner product
 * Performs the inner product between the given frequency and the
 * given row of the measurement matrix.
 * Note: this overridden version simply returns the row^th index.
 * 
 * @param frequency: lhs of the inner product
 * @param row: row of the measurement matrix to perform the inner product with (rhs)
 * @return: inner product as a binary (0-1) int
 */
bit naive_cs::inner_product(const vbits &frequency, unsigned long row) {
    return frequency[row];
}
