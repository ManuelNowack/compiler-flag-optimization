#ifndef LASSO_HPP
#define LASSO_HPP

#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "common.hpp"
#include "correlation.hpp"
#include "baseline.hpp"
#include "dependencygraph.hpp"
#include "threadpool.hpp"
#include "posettp.hpp"

struct Estimator {
  std::vector<std::vector<bool>> freqs;
  std::vector<c_real> coefs;
};

struct BitwiseEstimator64 {
  std::vector<std::vector<uint64_t>> freqs;
  std::vector<c_real> coefs;
};

struct BitwiseEstimator32 {
  std::vector<std::vector<int32_t>> freqs;
  std::vector<c_real> coefs;
};

Estimator* coordinate_descent(bool* x, c_real* y, int32_t m, int32_t n, bool* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, Estimator* est, bool center, char function,
    DependencyGraph* dg, int64_t& corrtime, bool recursive);
    
void coordinate_descent_regularization_path(bool* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center, char function,
    Estimator**& models, c_real*& lams, DependencyGraph* dg, int64_t& corrtime, bool recursive);
    
BitwiseEstimator64* coordinate_descent(uint64_t* x, c_real* y, uint64_t* v, int32_t m, int32_t n, uint64_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator64* est, bool center, int64_t& corrtime, TP_TYPE* tp, bool recursive);
    
void coordinate_descent_regularization_path(uint64_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator64**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp, bool recursive);
    
BitwiseEstimator32* coordinate_descent(int32_t* x, c_real* y, int32_t m, int32_t n, int32_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator32* est, bool center, int64_t& corrtime, TP_TYPE* tp);
    
void coordinate_descent_regularization_path(int32_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator32**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp);
    
BitwiseEstimator64* coordinate_descent(uint64_t* x, c_real* y, uint64_t* v, int32_t m, int32_t n, uint64_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator64* est, bool center, int64_t& corrtime, TP_TYPE* tp,
    uint64_t*& q, int32_t*& q_val, int32_t& q_size, bool recursive);
    
void coordinate_descent_regularization_path(uint64_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator64**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp,
    uint64_t*& q, int32_t*& q_val, int32_t& q_size, bool recursive);

void coordinate_descent_regularization_path(uint64_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator64**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp,
    uint64_t*& q, int32_t*& q_val, c_real*& mus, int32_t& q_size, bool recursive);

double squared_norm(double* vec, int32_t size);
double dot(double* vec1, double* vec2, int32_t size);
double dot_neg(double* vec1, double* vec2, int32_t size);
double _mm256_reduce_add_pd(__m256d v);
__m256d _mm256_abs_pd(__m256d v);

float squared_norm(float* vec, int32_t size);
float dot(float* vec1, float* vec2, int32_t size);
float dot_neg(float* vec1, float* vec2, int32_t size);
float _mm256_reduce_add_ps(__m256 v);
__m256 _mm256_abs_ps(__m256 v);

#endif
