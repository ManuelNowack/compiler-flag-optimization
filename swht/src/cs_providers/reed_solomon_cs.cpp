//========================
// Reed-Solomon CS (impl)
//========================


#include "reed_solomon_cs.h"

#include <cmath>


/** Constructor/destructor
 * Computes the Reed-solomon log and anti-log tables and builds the
 * measurement matrix.
 * 
 * @param n: time-domain index size
 * @param degree: degree of the input signal
 */
reed_solomon_cs::reed_solomon_cs(unsigned long n, unsigned long degree):
    finite_field_cs(n, 0), encoder(std::ceil(std::log2(n + 1ul)), degree) {

    number_of_measurements = encoder.calculate_all_syndromes(n, measurement_matrix);
}
reed_solomon_cs::~reed_solomon_cs() {
    delete[] measurement_matrix;
}


/** Update
 * Does nothing.
 */
void reed_solomon_cs::update() {}


/** Measurement matrix
 * Returns the required vector from the measurement matrix.
 * 
 * @param i: index of the measurement matrix row to return
 * @return: a row of the measurement matrix (as a vector reference)
 */
const bit *reed_solomon_cs::measurement_matrix_row(unsigned long i) {
    return measurement_matrix + i * n;
}


/** Low degree vector recovery
 * Performs Reed-Solomon error finding to extract the positive frequency indexes.
 * 
 * @param measurement: bit sequence to be turned into a frequency (inout)
 */
void reed_solomon_cs::recover_frequency(vbits &measurement) {
    std::vector<unsigned long> errors_locations;
    encoder.find_errors(measurement, n, errors_locations);
    measurement = vbits(n, 0u);
    for (const unsigned long &i: errors_locations)
        measurement[i] = 1u;
}
