//====================================
// Constants and types for global use
//====================================

#ifndef GLOBAL_CONSTANTS_H
#define GLOBAL_CONSTANTS_H


#include <vector>
#include <cstdint>
#include <unordered_map>

// Single bit
typedef u_int32_t bit;

// Bit sequence (big-endian)
typedef std::vector<bit> vbits;

// Compact bit sequence (little-endian)
typedef uint64_t bits;

// Frequency mapping
struct bit_hash {
    size_t operator()(const vbits &op) const {
        size_t out = 0ul;
        for (size_t i = 0; i < op.size(); i++) {
            out ^= op[i] << (i % 64ul);
        }
        return out;
    }
};
typedef std::unordered_map<vbits, double, bit_hash> frequency_map;

#endif
