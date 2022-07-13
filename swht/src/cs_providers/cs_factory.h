//=======================================
// Factory for finite field CS providers
//=======================================

#ifndef CS_FACTORY_H
#define CS_FACTORY_H


#include "finite_field_cs.h"


// Possible algorithms
#define NAIVE_CS 0
#define RANDOM_BINNING_CS 1
#define REED_SOLOMON_CS 2


/** Algorithm selector
 * Factory to generate an instance of a finite field CS provider.
 */
template <int cs_algorithm, typename ... Args>
finite_field_cs *get_finite_field_cs(unsigned long n, Args ... cs_args) {
    return nullptr; // error case
}
template <> finite_field_cs *get_finite_field_cs<NAIVE_CS>(unsigned long n);
template <> finite_field_cs *get_finite_field_cs<RANDOM_BINNING_CS, unsigned long, unsigned long, double>(
    unsigned long n, unsigned long n_bins, unsigned long iterations, double ratio);
template <> finite_field_cs *get_finite_field_cs<REED_SOLOMON_CS, unsigned long>(unsigned long n, unsigned long degree);

#endif
