//===========================
// Profiling functionalities
//===========================

#ifndef PROFILING_H
#define PROFILING_H


#include "timing.h"

#include <unordered_map>
#include <functional>

// Global storage for the counters
extern std::unordered_map<std::string, uint64_t> timestamps;
extern uint64_t sample_counter;

// Iterate counters
inline void iterate_sections(std::function<void(std::string, uint64_t, uint64_t)> f) {
    for (const auto &[section, cycles]: timestamps) {
        if (cycles > 0ul)
            f(section, cycles, sample_counter);
    }
}

// Initialize the global counters
#define PROFILING_INIT std::unordered_map<std::string, uint64_t> timestamps;\
    uint64_t sample_counter = 0ul;

// Reset the global counters (for after warmup)
#define PROFILING_RESET timestamps.clear(); sample_counter = 0ul;

// Ready section duration holder
#define DURATION_READY static uint64_t local_duration;

// Count sampling occurences
#define COUNT_SAMPLE(n) sample_counter += n;

// Declare section initial timestamp
#define START_STAMP(name) section_ ## name ## _start
#define SECTION_STAMP_DECLARE(name) timestamp START_STAMP(name);

// Start section timer
#define SECTION_TIMER_START(name) start_stamp(START_STAMP(name));

// Initialize and immediately start section timer
#define SECTION_START(name) SECTION_STAMP_DECLARE(name)\
    SECTION_TIMER_START(name)

// Stop section timer and increment time by measured step
#define SECTION_END(name) end_stamp(local_duration, START_STAMP(name))\
    {\
        auto match = timestamps.find(#name);\
        if (match == timestamps.end()) {\
            timestamps[#name] = local_duration;\
        } else {\
            match->second += local_duration;\
        }\
    }

#endif
