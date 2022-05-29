#ifndef EXPORT_HPP
#define EXPORT_HPP

#include <iostream>

#include "common.hpp"
#include "baseline.hpp"
#include "threadpool.hpp"
#include "dependencygraph.hpp"
#include "lasso.hpp"
#include "correlation.hpp"

void test_me();

void c_fit(bool* x, double* yd, int32_t samples, int32_t features,
           std::vector<std::vector<std::vector<bool>>>& freqs,
           std::vector<std::vector<double>>& coefs,
           double C, int32_t Nlams, double Steps, bool Recursive);

#endif
