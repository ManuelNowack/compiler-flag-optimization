//========================================================
// Factory for finite field CS providers (implementation)
//========================================================

#include "cs_factory.h"
#include "naive_cs.h"
#include "random_binning_cs.h"
#include "reed_solomon_cs.h"


/** Algorithm selector
 * Factory to generate an instance of a finite field CS provider.
 */
template <> finite_field_cs *get_finite_field_cs<NAIVE_CS>(unsigned long n) {
    return new naive_cs(n);
}
template <> finite_field_cs *get_finite_field_cs<RANDOM_BINNING_CS, unsigned long, unsigned long, double>(
    unsigned long n, unsigned long n_bins, unsigned long iterations, double ratio) {
    return new random_binning_cs(n, n_bins, iterations, ratio);
}
template <> finite_field_cs *get_finite_field_cs<REED_SOLOMON_CS, unsigned long>(unsigned long n, unsigned long degree) {
    return new reed_solomon_cs(n, degree);
}
