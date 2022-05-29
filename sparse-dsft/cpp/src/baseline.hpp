#ifndef BASELINE_H
#define BASELINE_H

#include <vector>
#include <queue>
#include <iostream>
#include <cmath>
#include "common.hpp"

extern uint64_t bas_it_ctr;

c_real compute_max_correlation_baseline(bool* x, c_real* y, bool* b, bool* fb, int32_t samples, int32_t features);
c_real compute_max_correlation_recursive_baseline(bool* x, c_real* y, bool* b, bool* fb, int32_t samples, int32_t features);
void traverse_rec(std::vector<int32_t>& freq, c_real& best_corr, std::vector<int32_t>& best_freq,
    std::vector<int32_t>& active, bool* x, c_real* y, int32_t samples, int32_t features);

#endif
