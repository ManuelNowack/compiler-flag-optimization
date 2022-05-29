#ifndef COMMON_DSFT_HPP
#define COMMON_DSFT_HPP

#include <iostream>
#include <ctime>
#include <limits>
#include <cassert>
#include <thread>
#include "immintrin.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//  These are variants used in the experiments, modify here                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BITWISE true
#define _64_BIT true
#define CYCLIC_BUFFER true // won't work with !_64_BIT, when !ITERATIVE, this must be set to false
#define FEWER_LOOPS true
#define PARALLEL true      // won't work with !_64_BIT
#define ITERATIVE true     // iterative execution, otherwise recursive. TODO: make this picked dynamically
#define GET_STATS false    // return some stats, like max queue size. currently works only for threaded bindings
#define BIT_AVX false       // use AVX for bitwise operations in computations on correlation, currently works only in parallel, CHRIS: setting this to true does not work!
#define JOB_STEALING false  // job stealing with a mutex-based lock mechanism, CHRIS: setting this to true does not work!

#define GOOD_OLD_IMPL true // if this is false, some experiments with false sharing reduction will be made

#define AVX_EXPERIMENT false

#define STRUCT_EXPERIMENT false

//~ #define TP_TYPE PosetTP
#define TP_TYPE ThreadPool

////////////////////////////////////////////////////////////////////////////////////////////////////
// End of experiments variants                                                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////

#define INIT_CYCLIC_SIZE 800000  // initial size of the cyclic buffer (#elements)

// currently supported types: double, float (due to AVX)
#define c_real double   // leave it as it is, unless you have trouble with memory bandwidth
#define PRINT_STATS false    // print some stats for iterative - this is used for debugging

#define AVX true        // don't set to false, unless running an ancient computer

// don't change this one!
#if !AVX
#undef BIT_AVX
#define BIT_AVX false
#undef AVX_EXPERIMENT
#define AVX_EXPERIMENT false
#endif

// Print error message and abort
void err(const std::string& msg);

// Get some timings
#define GIGA 1000000000
typedef struct timespec clocktime;
int64_t get_time_difference(clocktime *start, clocktime *end);
int get_time(clocktime *time);

double squared_norm(double* vec, int32_t size);
double dot(double* vec1, double* vec2, int32_t size);
// DOT (vec1, -vec2)
// TODO: multiplying by -1 and FMA could be pipelined?
double dot_neg(double* vec1, double* vec2, int32_t size);
#if AVX
double _mm256_reduce_add_pd(__m256d v);
__m256d _mm256_abs_pd(__m256d v);
#endif

float squared_norm(float* vec, int32_t size);
float dot(float* vec1, float* vec2, int32_t size);
// DOT (vec1, -vec2)
// TODO: multiplying by -1 and FMA could be pipelined?
float dot_neg(float* vec1, float* vec2, int32_t size);
#if AVX
float _mm256_reduce_add_ps(__m256 v);
__m256 _mm256_abs_ps(__m256 v);
#endif

#endif
