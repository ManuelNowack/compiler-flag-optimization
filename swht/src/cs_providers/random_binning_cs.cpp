//====================================
// Random binning CS (implementation)
//====================================


#include "random_binning_cs.h"
#include "linear_algebra.h"
#include "build_info.h"


/** Constructor/destructor
 * Sets the parameters.
 * 
 * @param n: time-domain index size
 * @param n_bins: number of bins to randomly fill
 * @param iterations: number of passes to attempt bin searching
 * @param ratio: bins reduction ratio between passes
 */
random_binning_cs::random_binning_cs(unsigned long n, unsigned long n_bins, unsigned long iterations, double ratio):
    finite_field_cs(n, 0ul), iterations(iterations), n_bins(n_bins), ratio(ratio),
#ifdef BENCHMARKING_BUILD
    random_engine(0),
#else
    random_engine(std::random_device()()),
#endif
    coordinates(iterations, coord()), recovery(iterations, recovery_mapping()) {}

random_binning_cs::~random_binning_cs() {
    for (unsigned long i = 0ul; i < number_of_measurements; i++)
        delete[] measurement_matrix[i];
}


/** Update
 * Generates iterative random frequency coordinates binnings and
 * binary searches through the created bins.
 */
void random_binning_cs::update() {

    // Initialize per-iteration mappings
    unsigned long tmp_n_bins = n_bins;
    for (unsigned long i = 0ul; i < iterations; i++) {
        coordinates[i].clear();
        recovery[i].clear();
    }
    for (unsigned long i = 0ul; i < number_of_measurements; i++)
        delete[] measurement_matrix[i];
    measurement_matrix.clear();

    // Ready one set of mappings per iteration
    unsigned long index = 0ul;
    for (unsigned long i = 0ul; i < iterations; i++) {

        // Ready pseudo-random generation
        std::uniform_int_distribution<unsigned long> prng(0ul, tmp_n_bins);

        // Distribute coordinates in random bins
        for (unsigned long j = 0ul; j < n; j++) {
            unsigned long bin = prng(random_engine);
            if (coordinates[i].find(bin) == coordinates[i].end())
                coordinates[i][bin] = std::vector<unsigned long>();
            coordinates[i][bin].push_back(j);
        }

        // Generate measurement matrix with per-bin binary search bits
        for (auto &&bin_and_coords: coordinates[i]) {
            unsigned long coord_length = bin_and_coords.second.size();
            unsigned long bit_length = (unsigned long) std::ceil(std::log2(coord_length));
            recovery[i][bin_and_coords.first] = std::vector<std::pair<unsigned long, unsigned long>>();
            for (unsigned long bit_i = 0ul; bit_i < bit_length; bit_i++) {
                bit *shift = new bit[n]();
                auto coord_index = bin_and_coords.second.begin();
                for (unsigned long j = 0ul; j < coord_length; j++) {
                    shift[*coord_index] = (j / (1ul << bit_i)) % 2u;
                    coord_index++;
                }
                measurement_matrix.push_back(shift);
                recovery[i][bin_and_coords.first].push_back(std::pair<unsigned long, unsigned long>(index++, bit_i));
            }
            bit *shift = new bit[n]();
            for (auto coord_index = bin_and_coords.second.begin(); coord_index != bin_and_coords.second.end(); coord_index++) {
                shift[*coord_index] = 1u;
            }
            measurement_matrix.push_back(shift);
            recovery[i][bin_and_coords.first].push_back(std::pair<unsigned long, unsigned long>(index++, all_ones));
        }

        // Update number of bins for next iteration
        tmp_n_bins = std::floor(tmp_n_bins / ratio);
    }

    // Set parent values
    number_of_measurements = measurement_matrix.size();
}


/** Measurement matrix
 * Returns the required vector from the measurement matrix.
 * 
 * @param i: index of the measurement matrix row to return
 * @return: a row of the measurement matrix (as a vector reference)
 */
const bit *random_binning_cs::measurement_matrix_row(unsigned long i) {
    return measurement_matrix[i];
}


/** Low degree vector recovery
 * Performs rounds of bin-wise binary searches and combines the results
 * into the extracted frequency.
 * 
 * @param measurement: bit sequence to be turned into a frequency (inout)
 */
void random_binning_cs::recover_frequency(vbits &measurement) {

    // Update frequency in rounds
    vbits current_estimate(n, 0u);
    for (unsigned long i = 0; i < iterations; i++) {

        // For each bin attempt to find the coordinate of the non-zero frequency bit (supposing there is only one)
        vbits residual_estimate(n, 0u);
        for (auto &&bin_and_indexbit: recovery[i]) {
            const std::vector<unsigned long> bin_coordinates = coordinates[i][bin_and_indexbit.first];
            unsigned long recovered_bit_index = 0ul; // non-zero frequency index

            // Perform binary search
            unsigned all_ones_bit = 0u;
            for (auto &&index_and_bit: bin_and_indexbit.second) {

                // Check if the coordinate bit at <index> is positive after subtracting the current estimate
                unsigned residual_measurement = measurement[index_and_bit.first] !=
                    bool_inner_product(measurement_matrix[index_and_bit.first], current_estimate.data(), n); // Can be turned into matrix op

                // Keep all ones shift result for edge case checking
                if (index_and_bit.second == all_ones) {
                    all_ones_bit = residual_measurement;
                    continue;
                }
                
                // Combine coordinate bits to get the integer coordinate
                recovered_bit_index += residual_measurement * (1ul << index_and_bit.second);
            }

            // Case 1: positive bit found at index >0
            if (recovered_bit_index) { // recovered_bit_index != 0
                if (recovered_bit_index < bin_coordinates.size())
                    residual_estimate[bin_coordinates[recovered_bit_index]] = 1u;
            }
            
            // Case 2: (edge case) positive bit at index 0
            else if (all_ones_bit) { // recovered_bit_index == 0 && all_ones_bit == 1
                residual_estimate[bin_coordinates[0]] = 1u;
            }

            // Case 3: (default) no positive bit (all zeros)
            // Nothing to do
        }

        // Adjust current estimate for the next round
        bit *current_estimate_ptr = current_estimate.data();
        bit *residual_estimate_ptr = residual_estimate.data();
        vector_xor_vector(current_estimate_ptr, residual_estimate_ptr, n, current_estimate_ptr)
    }

    // Output estimate after all rounds
    measurement = current_estimate;
}
