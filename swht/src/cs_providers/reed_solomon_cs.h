//=================
// Reed-Solomon CS
//=================


#ifndef REED_SOLOMON_CS_H
#define REED_SOLOMON_CS_H


#include "finite_field_cs.h"
#include "reed-solomon/reed-solomon.h"


/** Reed-Solomon CS provider
 * 
 */
struct reed_solomon_cs: public finite_field_cs {

    reed_solomon_encoder encoder;
    bit *measurement_matrix;

    /** Constructor/destructor
     * Computes the Reed-solomon log and anti-log tables and builds the
     * measurement matrix.
     * 
     * @param n: time-domain index size
     * @param degree: degree of the input signal
     */
    reed_solomon_cs(unsigned long n, unsigned long degree);
    ~reed_solomon_cs();

    /** Update
     * Does nothing.
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
     * Performs Reed-Solomon error finding to extract the positive frequency indexes.
     * 
     * @param measurement: bit sequence to be turned into a frequency (inout)
     */
    void recover_frequency(vbits &measurement);
};

#endif
