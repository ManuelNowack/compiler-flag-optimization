//===================
// Random binning CS
//===================


#ifndef RANDOM_BINNING_CS_H
#define RANDOM_BINNING_CS_H


#include "finite_field_cs.h"

#include <unordered_map>
#include <limits>
#include <random>

typedef std::unordered_map<unsigned long, std::vector<unsigned long>> coord;
typedef std::unordered_map<unsigned long, std::vector<std::pair<unsigned long, unsigned long>>> recovery_mapping;


/** Random binning CS provider
 * Accelerated compressive sensing based on sending frequencies to randomly
 * selected bins and retrieving them iteratively by binary searching each
 * bin with the assumption that only one non-zero frequency at most got
 * sent to each one (likely if n_bins > degree).
 */
struct random_binning_cs: public finite_field_cs {

    unsigned long iterations, n_bins;
    double ratio;
    std::default_random_engine random_engine;
    std::vector<bit *> measurement_matrix;
    std::vector<coord> coordinates;
    std::vector<recovery_mapping> recovery;
    static const unsigned long all_ones = std::numeric_limits<unsigned long>::max();

    /** Constructor/destructor
     * Sets the parameters.
     * 
     * @param n: time-domain index size
     * @param n_bins: number of bins to randomly fill
     * @param iterations: number of passes to attempt bin searching
     * @param ratio: bins reduction ratio between passes
     */
    random_binning_cs(unsigned long n, unsigned long n_bins, unsigned long iterations, double ratio);
    ~random_binning_cs();

    /** Update
     * Generates iterative random frequency coordinates binnings and
     * binary searches through the created bins.
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
     * Performs rounds of bin-wise binary searches and combines the results
     * into the extracted frequency.
     * 
     * @param measurement: bit sequence to be turned into a frequency (inout)
     */
    void recover_frequency(vbits &measurement);
};

#endif
