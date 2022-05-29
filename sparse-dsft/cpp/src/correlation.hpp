#ifndef CORRELATION_HPP
#define CORRELATION_HPP

#include <iostream>
#include <cmath>
#include <queue>
#include <cstring>
#include "common.hpp"

extern uint64_t opt_it_ctr;

c_real compute_max_correlation_bitwise(uint64_t* x, c_real* y, uint64_t* b, bool* fb, int32_t samples, int32_t features);
c_real compute_max_correlation_bitwise(int32_t* x, c_real* y, int32_t* b, bool* fb, int32_t samples, int32_t features);
c_real compute_max_correlation_bitwise(uint64_t* x, c_real* y, uint64_t* b, bool* fb, int32_t samples, int32_t features,
                                       uint64_t*& q, int32_t*& q_val, c_real*& mus, int32_t& q_size);

c_real compute_max_correlation_bitwise_rec(uint64_t* x, c_real* y, uint64_t* b, bool* fb, uint64_t* v, int32_t samples, int32_t features);

#endif
