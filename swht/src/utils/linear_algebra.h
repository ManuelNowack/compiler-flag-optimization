//============================================
// Basic matrix and vector operations (naive)
//============================================

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include "global_constants.h"

#ifndef __GNUC__
#include <immintrin.h>
#define parity(n) (_mm_popcnt_u64(n) & 1ul)
#else
#define parity(n) __builtin_parityl(n)
#endif


/** Naive matrix-vector multiplication
 * Basically working MVM.
 */
#define matrix_dot_vector(M, v, n_rows, out) {\
    for (size_t i = 0ul; i < n_rows; i++) {\
        out[i] = parity(M[i] & v);\
    }\
}

/** Naive matrix^T-vector multiplication
 * Basically working MVM with transposed matrix.
 */
#define matrixT_dot_vector(M, v, n, out)\
    bits out = 0ul;\
    for (size_t i = 0ul; i < n; i++) {\
        out ^= M[i] * v[i];\
    }\


/** Naive vector-vector XOR
 * Basically working. !!! Can break if not given 0-1 !!!
 */
#define vector_xor_vector(x, y, n, out) \
    for (unsigned long i = 0ul; i < n; i++) {\
        out[i] = x[i] != y[i];\
    }\


/** Naive boolean inner-product
 * Can break if not given 0-1.
 */
inline unsigned bool_inner_product(const bit *v, const bit *w, unsigned long n) {
    unsigned out = 0ul;
    for (unsigned long i = 0ul; i < n; i++) {
        out ^= v[i] && w[i];
    }
    return out;
}

#endif
