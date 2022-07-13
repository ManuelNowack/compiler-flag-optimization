//===========================================
// Low-level Intel cycle counting (GNU only)
//===========================================

#ifndef TIMING_H
#define TIMING_H

#include <cstdint>

union timestamp {
    uint64_t int64;
    struct {
        uint32_t low, high;
    } int32;
};

#define start_stamp(stamp) asm volatile ("CPUID\n\t"\
    "RDTSC\n\t"\
    "mov %%edx, %0\n\t"\
    "mov %%eax, %1\n\t": "=r" (stamp.int32.high), "=r" (stamp.int32.low):: "%rax", "%rbx", "%rcx", "%rdx");

#define end_stamp(out, start) {timestamp stamp;\
    asm volatile ("RDTSCP\n\t"\
    "mov %%edx, %0\n\t"\
    "mov %%eax, %1\n\t"\
    "CPUID\n\t": "=r" (stamp.int32.high), "=r" (stamp.int32.low):: "%rax", "%rbx", "%rcx", "%rdx");\
    out = stamp.int64 - start.int64;}

#endif
