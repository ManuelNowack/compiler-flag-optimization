#include "common.hpp"

void err(const std::string& msg) {
    std::cerr << "ERROR: " << msg << "!\n";
    exit(-1);
}

int64_t get_time_difference(clocktime *start, clocktime *end) {
  int64_t seconds_diff = (int64_t)(end->tv_sec)
      - (int64_t)(start->tv_sec);
  int64_t end_nano = end->tv_nsec;
  int64_t start_nano = start->tv_nsec;
  if (start_nano <= end_nano)
    return seconds_diff * GIGA + end_nano - start_nano;
  return (seconds_diff - 1) * GIGA + GIGA - (start_nano - end_nano);
}

int get_time(clocktime *time) {
    return clock_gettime(CLOCK_REALTIME, time);
}

double squared_norm(double* vec, int32_t size) {
  #if AVX
    __m256d acc0, acc1, acc2, acc3;
    acc0 = _mm256_setzero_pd();
    acc1 = _mm256_setzero_pd();
    acc2 = _mm256_setzero_pd();
    acc3 = _mm256_setzero_pd();
    int32_t sv = ((size >> 2) << 2);
    int32_t sa = ((size >> 4) << 4);
    int32_t s = 0;
    double res = 0.0;
    for (; s < sa; s += 16) {
      __m256d v0 = _mm256_loadu_pd(vec + s + 0);
      __m256d v1 = _mm256_loadu_pd(vec + s + 4);
      __m256d v2 = _mm256_loadu_pd(vec + s + 8);
      __m256d v3 = _mm256_loadu_pd(vec + s + 12);
      acc0 = _mm256_fmadd_pd(v0, v0, acc0);
      acc1 = _mm256_fmadd_pd(v1, v1, acc1);
      acc2 = _mm256_fmadd_pd(v2, v2, acc2);
      acc3 = _mm256_fmadd_pd(v3, v3, acc3);
    }
    for (; s < sv; s += 4) {
      __m256d v0 = _mm256_loadu_pd(vec + s);
      acc0 = _mm256_fmadd_pd(v0, v0, acc0);
    }
    for (; s < size; ++s) {
      res += vec[s] * vec[s];
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);
    return res + _mm256_reduce_add_pd(acc0);
  #else
    double acc0 = 0;
    double acc1 = 0;
    double acc2 = 0;
    double acc3 = 0;
    int32_t sa = (size >> 2) << 2;
    int32_t s = 0;
    for (; s < sa; s += 4) {
      acc0 += vec[s + 0] * vec[s + 0];
      acc1 += vec[s + 1] * vec[s + 1];
      acc2 += vec[s + 2] * vec[s + 2];
      acc3 += vec[s + 3] * vec[s + 3];
    }
    for (; s < size; ++s) {
      acc0 += vec[s] * vec[s];
    }
    return (acc0 + acc1) + (acc2 + acc3);
  #endif
}

double dot(double* vec1, double* vec2, int32_t size) {
  #if AVX
    __m256d acc0, acc1, acc2, acc3;
    acc0 = _mm256_setzero_pd();
    acc1 = _mm256_setzero_pd();
    acc2 = _mm256_setzero_pd();
    acc3 = _mm256_setzero_pd();
    int32_t sv = (size >> 2) << 2;
    int32_t sa = (size >> 4) << 4;
    int32_t s = 0;
    double res = 0.0;
    for (; s < sa; s += 16) {
      __m256d v0 = _mm256_loadu_pd(vec1 + s + 0);
      __m256d v1 = _mm256_loadu_pd(vec1 + s + 4);
      __m256d v2 = _mm256_loadu_pd(vec1 + s + 8);
      __m256d v3 = _mm256_loadu_pd(vec1 + s + 12);
      __m256d u0 = _mm256_loadu_pd(vec2 + s + 0);
      __m256d u1 = _mm256_loadu_pd(vec2 + s + 4);
      __m256d u2 = _mm256_loadu_pd(vec2 + s + 8);
      __m256d u3 = _mm256_loadu_pd(vec2 + s + 12);
      acc0 = _mm256_fmadd_pd(v0, u0, acc0);
      acc1 = _mm256_fmadd_pd(v1, u1, acc1);
      acc2 = _mm256_fmadd_pd(v2, u2, acc2);
      acc3 = _mm256_fmadd_pd(v3, u3, acc3);
    }
    for (; s < sv; s += 4) {
      __m256d v0 = _mm256_loadu_pd(vec1 + s);
      __m256d u0 = _mm256_loadu_pd(vec2 + s);
      acc0 = _mm256_fmadd_pd(v0, u0, acc0);
    }
    for (; s < size; ++s) {
      res += vec1[s] * vec2[s];
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);
    return res + _mm256_reduce_add_pd(acc0);
  #else
    double acc0 = 0;
    double acc1 = 0;
    double acc2 = 0;
    double acc3 = 0;
    int32_t sa = (size >> 2) << 2;
    int32_t s = 0;
    for (; s < sa; s += 4) {
      acc0 += vec1[s + 0] * vec2[s + 0];
      acc1 += vec1[s + 1] * vec2[s + 1];
      acc2 += vec1[s + 2] * vec2[s + 2];
      acc3 += vec1[s + 3] * vec2[s + 3];
    }
    for (; s < size; ++s) {
      acc0 += vec1[s] * vec2[s];
    }
    return (acc0 + acc1) + (acc2 + acc3);
  #endif
}

// DOT (vec1, -vec2)
// TODO: multiplying by -1 and FMA could be pipelined?
double dot_neg(double* vec1, double* vec2, int32_t size) {
  #if AVX
    __m256d acc0, acc1, acc2, acc3;
    acc0 = _mm256_setzero_pd();
    acc1 = _mm256_setzero_pd();
    acc2 = _mm256_setzero_pd();
    acc3 = _mm256_setzero_pd();
    __m256d mone = _mm256_set1_pd(-1.0);
    int32_t sv = ((size >> 2) << 2);
    int32_t sa = ((size >> 4) << 4);
    int32_t s = 0;
    double res = 0.0;
    for (; s < sa; s += 16) {
      __m256d v0 = _mm256_loadu_pd(vec1 + s + 0);
      __m256d v1 = _mm256_loadu_pd(vec1 + s + 4);
      __m256d v2 = _mm256_loadu_pd(vec1 + s + 8);
      __m256d v3 = _mm256_loadu_pd(vec1 + s + 12);
      __m256d u0 = _mm256_mul_pd(_mm256_loadu_pd(vec2 + s + 0), mone);
      __m256d u1 = _mm256_mul_pd(_mm256_loadu_pd(vec2 + s + 4), mone);
      __m256d u2 = _mm256_mul_pd(_mm256_loadu_pd(vec2 + s + 8), mone);
      __m256d u3 = _mm256_mul_pd(_mm256_loadu_pd(vec2 + s + 12), mone);
      acc0 = _mm256_fmadd_pd(v0, u0, acc0);
      acc1 = _mm256_fmadd_pd(v1, u1, acc1);
      acc2 = _mm256_fmadd_pd(v2, u2, acc2);
      acc3 = _mm256_fmadd_pd(v3, u3, acc3);
    }
    for (; s < sv; s += 4) {
      __m256d v0 = _mm256_loadu_pd(vec1 + s);
      __m256d u0 = _mm256_mul_pd(_mm256_loadu_pd(vec2 + s), mone);
      acc0 = _mm256_fmadd_pd(v0, u0, acc0);
    }
    for (; s < size; ++s) {
      res += vec1[s] * -vec2[s];
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);
    return res + _mm256_reduce_add_pd(acc0);
  #else
    double acc0 = 0;
    double acc1 = 0;
    double acc2 = 0;
    double acc3 = 0;
    int32_t sa = (size >> 2) << 2;
    int32_t s = 0;
    for (; s < sa; s += 4) {
      acc0 += vec1[s + 0] * -vec2[s + 0];
      acc1 += vec1[s + 1] * -vec2[s + 1];
      acc2 += vec1[s + 2] * -vec2[s + 2];
      acc3 += vec1[s + 3] * -vec2[s + 3];
    }
    for (; s < size; ++s) {
      acc0 += vec1[s] * -vec2[s];
    }
    return (acc0 + acc1) + (acc2 + acc3);
  #endif
}

#if AVX
double _mm256_reduce_add_pd(__m256d v) {
  double out[2];
  __m256d v1 = _mm256_permute_pd(v, 0x5);
  __m256d s1 = _mm256_add_pd(v, v1);
  __m128d sl = _mm256_extractf128_pd(s1, 0);
  __m128d sh = _mm256_extractf128_pd(s1, 1);
  __m128d v2 = _mm_add_pd(sl, sh);
  _mm_storeu_pd(out, v2);
  return out[0];
}

__m256d _mm256_abs_pd(__m256d v) {
  __m256d mone = _mm256_set1_pd(-1);
  __m256d neg = _mm256_mul_pd(v, mone);
  return _mm256_max_pd(v, neg);
}
#endif

float squared_norm(float* vec, int32_t size) {
  #if AVX
    __m256 acc0, acc1, acc2, acc3;
    acc0 = _mm256_setzero_ps();
    acc1 = _mm256_setzero_ps();
    acc2 = _mm256_setzero_ps();
    acc3 = _mm256_setzero_ps();
    int32_t sv = ((size >> 3) << 3);
    int32_t sa = ((size >> 5) << 5);
    int32_t s = 0;
    float res = 0.0;
    for (; s < sa; s += 32) {
      __m256 v0 = _mm256_loadu_ps(vec + s + 0);
      __m256 v1 = _mm256_loadu_ps(vec + s + 8);
      __m256 v2 = _mm256_loadu_ps(vec + s + 16);
      __m256 v3 = _mm256_loadu_ps(vec + s + 24);
      acc0 = _mm256_fmadd_ps(v0, v0, acc0);
      acc1 = _mm256_fmadd_ps(v1, v1, acc1);
      acc2 = _mm256_fmadd_ps(v2, v2, acc2);
      acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }
    for (; s < sv; s += 8) {
      __m256 v0 = _mm256_loadu_ps(vec + s);
      acc0 = _mm256_fmadd_ps(v0, v0, acc0);
    }
    for (; s < size; ++s) {
      res += vec[s] * vec[s];
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    return res + _mm256_reduce_add_ps(acc0);
  #else
    float acc0 = 0;
    float acc1 = 0;
    float acc2 = 0;
    float acc3 = 0;
    int32_t sa = (size >> 2) << 2;
    int32_t s = 0;
    for (; s < sa; s += 4) {
      acc0 += vec[s + 0] * vec[s + 0];
      acc1 += vec[s + 1] * vec[s + 1];
      acc2 += vec[s + 2] * vec[s + 2];
      acc3 += vec[s + 3] * vec[s + 3];
    }
    for (; s < size; ++s) {
      acc0 += vec[s] * vec[s];
    }
    return (acc0 + acc1) + (acc2 + acc3);
  #endif
}

float dot(float* vec1, float* vec2, int32_t size) {
  #if AVX
    __m256 acc0, acc1, acc2, acc3;
    acc0 = _mm256_setzero_ps();
    acc1 = _mm256_setzero_ps();
    acc2 = _mm256_setzero_ps();
    acc3 = _mm256_setzero_ps();
    int32_t sv = (size >> 3) << 3;
    int32_t sa = (size >> 5) << 5;
    int32_t s = 0;
    float res = 0.0;
    for (; s < sa; s += 32) {
      __m256 v0 = _mm256_loadu_ps(vec1 + s + 0);
      __m256 v1 = _mm256_loadu_ps(vec1 + s + 8);
      __m256 v2 = _mm256_loadu_ps(vec1 + s + 16);
      __m256 v3 = _mm256_loadu_ps(vec1 + s + 24);
      __m256 u0 = _mm256_loadu_ps(vec2 + s + 0);
      __m256 u1 = _mm256_loadu_ps(vec2 + s + 8);
      __m256 u2 = _mm256_loadu_ps(vec2 + s + 16);
      __m256 u3 = _mm256_loadu_ps(vec2 + s + 24);
      acc0 = _mm256_fmadd_ps(v0, u0, acc0);
      acc1 = _mm256_fmadd_ps(v1, u1, acc1);
      acc2 = _mm256_fmadd_ps(v2, u2, acc2);
      acc3 = _mm256_fmadd_ps(v3, u3, acc3);
    }
    for (; s < sv; s += 8) {
      __m256 v0 = _mm256_loadu_ps(vec1 + s);
      __m256 u0 = _mm256_loadu_ps(vec2 + s);
      acc0 = _mm256_fmadd_ps(v0, u0, acc0);
    }
    for (; s < size; ++s) {
      res += vec1[s] * vec2[s];
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    return res + _mm256_reduce_add_ps(acc0);
  #else
    float acc0 = 0;
    float acc1 = 0;
    float acc2 = 0;
    float acc3 = 0;
    int32_t sa = (size >> 2) << 2;
    int32_t s = 0;
    for (; s < sa; s += 4) {
      acc0 += vec1[s + 0] * vec2[s + 0];
      acc1 += vec1[s + 1] * vec2[s + 1];
      acc2 += vec1[s + 2] * vec2[s + 2];
      acc3 += vec1[s + 3] * vec2[s + 3];
    }
    for (; s < size; ++s) {
      acc0 += vec1[s] * vec2[s];
    }
    return (acc0 + acc1) + (acc2 + acc3);
  #endif
}

// DOT (vec1, -vec2)
// TODO: multiplying by -1 and FMA could be pipelined?
float dot_neg(float* vec1, float* vec2, int32_t size) {
  #if AVX
    __m256 acc0, acc1, acc2, acc3;
    acc0 = _mm256_setzero_ps();
    acc1 = _mm256_setzero_ps();
    acc2 = _mm256_setzero_ps();
    acc3 = _mm256_setzero_ps();
    __m256 mone = _mm256_set1_ps(-1.0);
    int32_t sv = ((size >> 3) << 3);
    int32_t sa = ((size >> 5) << 5);
    int32_t s = 0;
    float res = 0.0;
    for (; s < sa; s += 32) {
      __m256 v0 = _mm256_loadu_ps(vec1 + s + 0);
      __m256 v1 = _mm256_loadu_ps(vec1 + s + 8);
      __m256 v2 = _mm256_loadu_ps(vec1 + s + 16);
      __m256 v3 = _mm256_loadu_ps(vec1 + s + 24);
      __m256 u0 = _mm256_mul_ps(_mm256_loadu_ps(vec2 + s + 0), mone);
      __m256 u1 = _mm256_mul_ps(_mm256_loadu_ps(vec2 + s + 8), mone);
      __m256 u2 = _mm256_mul_ps(_mm256_loadu_ps(vec2 + s + 16), mone);
      __m256 u3 = _mm256_mul_ps(_mm256_loadu_ps(vec2 + s + 24), mone);
      acc0 = _mm256_fmadd_ps(v0, u0, acc0);
      acc1 = _mm256_fmadd_ps(v1, u1, acc1);
      acc2 = _mm256_fmadd_ps(v2, u2, acc2);
      acc3 = _mm256_fmadd_ps(v3, u3, acc3);
    }
    for (; s < sv; s += 8) {
      __m256 v0 = _mm256_loadu_ps(vec1 + s);
      __m256 u0 = _mm256_mul_ps(_mm256_loadu_ps(vec2 + s), mone);
      acc0 = _mm256_fmadd_ps(v0, u0, acc0);
    }
    for (; s < size; ++s) {
      res += vec1[s] * -vec2[s];
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    return res + _mm256_reduce_add_ps(acc0);
  #else
    float acc0 = 0;
    float acc1 = 0;
    float acc2 = 0;
    float acc3 = 0;
    int32_t sa = (size >> 2) << 2;
    int32_t s = 0;
    for (; s < sa; s += 4) {
      acc0 += vec1[s + 0] * -vec2[s + 0];
      acc1 += vec1[s + 1] * -vec2[s + 1];
      acc2 += vec1[s + 2] * -vec2[s + 2];
      acc3 += vec1[s + 3] * -vec2[s + 3];
    }
    for (; s < size; ++s) {
      acc0 += vec1[s] * -vec2[s];
    }
    return (acc0 + acc1) + (acc2 + acc3);
  #endif
}

#if AVX
float _mm256_reduce_add_ps(__m256 v) {
  float out[8];
  __m256 v1 = _mm256_permute_ps(v, 0xb1);
  __m256 s1 = _mm256_add_ps(v, v1);
  __m256 v2 = _mm256_permute_ps(s1, 0x4e);
  __m256 s2 = _mm256_add_ps(s1, v2);
  __m256 v3 = _mm256_permute2f128_ps(s2, s2 , 0x01);
  __m256 s3 = _mm256_add_ps(s2, v3);
  _mm256_storeu_ps(out, s3);
  return out[0];
}

__m256 _mm256_abs_ps(__m256 v) {
  __m256 mone = _mm256_set1_ps(-1);
  __m256 neg = _mm256_mul_ps(v, mone);
  return _mm256_max_ps(v, neg);
}
#endif
