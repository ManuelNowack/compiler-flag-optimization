#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <vector>
#include <queue>
#include <chrono>
#include <cstring>
#include <cmath>
#include <future>
#include "common.hpp"

#if JOB_STEALING
#include <mutex>
#include <atomic>
#endif

void update_freq_range(uint64_t* x, uint64_t* b, bool* fb,
  size_t begin_idx, size_t end_idx, int32_t featbits);
    
void cmc_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed);
    
void cmc_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed);
    
void cmc_part_rec_subcall(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta, uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed);
    
void cmc_part_rec(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part_rec_subcall(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta, uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed);
    
void cmcpos_part_rec(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed);
    
    
    void cmc_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real& best_corr_local, uint64_t* local_b, double delta,
    uint64_t* q, int32_t* q_val, c_real* mus, int32_t& q_size,
    int64_t& max_queue, int64_t& total_processed);
    
#if BIT_AVX

void cmc_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed);
    
void cmc_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed);

void cmc_part_rec_subcall_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta, uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed);
    
void cmc_part_rec_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part_rec_subcall_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta, uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed);
    
void cmcpos_part_rec_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed);
    
#endif

#if STRUCT_EXPERIMENT

#include "structset.hpp"

void update_freq_range(uint64_t* x, uint64_t* b, bool* fb,
  size_t begin_idx, size_t end_idx, int32_t featbits);
  
void cmc_part(SetX x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, SetFreq* local_b, double delta,
    SetQueue* queue,
    int64_t* max_queue, int64_t* total_processed);
    
void cmcpos_part(SetX x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, SetFreq* local_b, double delta,
    SetQueue* queue,
    int64_t* max_queue, int64_t* total_processed);

#define FMJ FindMaxJob
#define FMPJ FindMaxPosJob

// ThreadPool will be initialized with parameterized number of threads,
// but currently not all might be active during the execution.
// To get max_corr, min(n_threads, features) are used
class ThreadPool {

private:

  struct FindMaxJob {
    SetX x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    SetFreq* local_b;
    double delta;
    SetQueue* queue;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      cmc_part(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, queue, max_queue, total_processed);
    }
    FindMaxJob(SetX x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, SetFreq* local_b,
        double delta, SetQueue* queue,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b),
        delta(delta), queue(queue),
        max_queue(max_queue), total_processed(total_processed) {
      /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
    }
  };

  struct FindMaxJobRec {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t* v;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      //~ #if BIT_AVX
      //~ cmc_part_rec_avx(x, y, b, samples, features, first_q_val, best_corr,
          //~ best_corr_thread, local_b, delta, v, max_queue, total_processed);
      //~ #else
      //~ cmc_part_rec(x, y, b, samples, features, first_q_val, best_corr,
          //~ best_corr_thread, local_b, delta, v, max_queue, total_processed);
      //~ #endif
    }
    FindMaxJobRec(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t* v, int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta), v(v),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJob {
    SetX x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    SetFreq* local_b;
    double delta;
    SetQueue* queue;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      cmcpos_part(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, queue, max_queue, total_processed);
    }
    FindMaxPosJob(SetX x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, SetFreq* local_b, double delta,
        SetQueue* queue,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        queue(queue),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };

  struct FindMaxPosJobRec {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t* v;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      //~ #if BIT_AVX
      //~ cmcpos_part_rec_avx(x, y, b, samples, features, first_q_val, best_corr,
          //~ best_corr_thread, local_b, delta, v, max_queue, total_processed);
      //~ #else
      //~ cmcpos_part_rec(x, y, b, samples, features, first_q_val, best_corr,
          //~ best_corr_thread, local_b, delta, v, max_queue, total_processed);
      //~ #endif
    }
    FindMaxPosJobRec(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t* v, int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta), v(v),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };

  int32_t n_threads;
  int32_t jobs_done;
  bool* job_done;
  std::vector<std::thread> threads;
  uint64_t* local_bs;
  SetFreq* local_bs_struct;
  bool running;
  c_real* best_corr_threads;
  uint64_t* vs;

  std::queue<FMJ*>* jobs;
  std::queue<FMPJ*>* jobs_pos;
  std::queue<FindMaxJobRec*>* jobs_rec;
  std::queue<FindMaxPosJobRec*>* jobs_pos_rec;
  
  SetQueue* queues;
  bool stopped;
  
  int64_t* max_queues;
  int64_t* total_processeds;
  
  int32_t nt_max_corr;
  int32_t featbits;
  
public:

  bool is_running() {
    return running;
  }

  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive, double delta,
      int64_t& max_queue, int64_t& total_processed) {
        
        for (int64_t i = 0; i < samples; ++i) {
      std::cout << y[i] << " ";
    }
    std::cout << "\n";
    
    SetX sx(x, samples, features);

    size_t n = features;
    featbits = ((features + 63) >> 6);
    c_real best_corr = 0;
    int32_t rowbits = featbits;
    
    nt_max_corr = std::min(n_threads, features);
    //~ int32_t nt_update = std::min(n_threads, samples);

    jobs_done = 0;
    
    if (!local_bs) {
      local_bs = new uint64_t[nt_max_corr * featbits];
      local_bs_struct = new SetFreq[nt_max_corr];
      std::memset(local_bs, 0, sizeof(uint64_t) * nt_max_corr * featbits);
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        local_bs_struct[i].init(local_bs + i * featbits, 0, featbits);
      }
      if (!recursive) {
        for (int32_t i = 0; i < nt_max_corr; ++i) {
          queues[i].init(INIT_CYCLIC_SIZE, features);
        }
      }
    }
    
    if (recursive && !vs) {
      vs = new uint64_t[nt_max_corr * rowbits];
    }
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      job_done[i] = false;
      best_corr_threads[i] = 0.0;
      max_queues[i] = 0;
      total_processeds[i] = 0;
    }
    std::vector<std::future<void>> results;
    std::vector<FMJ*> local_jobs;
    std::vector<FindMaxJobRec*> local_jobs_rec;
    if (recursive) {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxJobRec* j = new FindMaxJobRec(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        local_jobs_rec.push_back(j);
        jobs_rec[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_rec(i); }));
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxJob* j = new FindMaxJob(sx, y, b, samples, features,
            i, best_corr, best_corr_threads + thread_no,
            local_bs_struct + thread_no, delta, queues + thread_no,
            max_queues + thread_no, total_processeds + thread_no);
        local_jobs.push_back(j);
        jobs[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max(i); }));
      }
    }
    
    do {
      c_real new_best_corr = 0.0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        c_real bct = best_corr_threads[i];
        if (std::fabs(bct) > std::fabs(new_best_corr)) {
          new_best_corr = bct;
        }
      }
      if (std::fabs(best_corr) < std::fabs(new_best_corr)) {
        best_corr = new_best_corr;
      }
      jobs_done = 0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        jobs_done += job_done[i];
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } while (jobs_done != nt_max_corr);
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      results[i].get();
    }
    
    c_real new_best_corr = 0.0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      c_real bct = best_corr_threads[i];
      if (std::fabs(bct) > std::fabs(new_best_corr)) {
        new_best_corr = bct;
      }
    }
    if (std::fabs(best_corr) < std::fabs(new_best_corr)) {
      best_corr = new_best_corr;
    }
    
    std::cout << "BEST: " << best_corr << "\n";

    SetFreq b_struct;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      if (best_corr_threads[i] == best_corr) {
        b_struct.copy(local_bs_struct[i]);
      }
    }
    
    for (int64_t i = 0; i < samples; i++) {
      fb[i] = leq(b_struct, sx, i);
      std::cout << fb[i];
    }
    std::cout << std::endl;
    
    for (int64_t i = 0; i < samples; ++i) {
      std::cout << y[i] << " ";
    }
    std::cout << "\n";
    
    std::memcpy(b, b_struct.data, featbits * sizeof(uint64_t));
    
    #if GET_STATS
    int64_t new_max_queue = 0;
    int64_t new_total_processed = 0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      new_max_queue = std::max(new_max_queue, max_queues[i]);
      new_total_processed += total_processeds[i];
    }
    max_queue = new_max_queue;
    total_processed = new_total_processed;
    #endif
    
    for (FMJ* job : local_jobs) {
      delete job;
    }
    for (FindMaxJobRec* job : local_jobs_rec) {
      delete job;
    }
    
    return best_corr;
  }
  
  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmc_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max(int32_t id) {
    FMJ* function = nullptr;
    while (!jobs[id].empty()) {
      function = jobs[id].front();
      jobs[id].pop();
      if (function) {
        function->call();
        function = nullptr;
      }
    }
    job_done[id] = true;
  }

  void run_find_max_rec(int32_t id) {
    FindMaxJobRec* function = nullptr;
    while (!jobs_rec[id].empty()) {
      function = jobs_rec[id].front();
      jobs_rec[id].pop();
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
  }
  
  c_real cmcpos_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive, double delta,
      int64_t& max_queue, int64_t& total_processed) {
        
    SetX sx(x, samples, features);

    size_t n = features;
    int32_t featbits = ((features + 63) >> 6);
    c_real best_corr = std::numeric_limits<c_real>::min();
    int32_t rowbits = featbits;
    
    int32_t nt_max_corr = std::min(n_threads, features);
    //~ int32_t nt_update = std::min(n_threads, samples);

    jobs_done = 0;
    
    if (!local_bs) {
      local_bs = new uint64_t[nt_max_corr * featbits];
      local_bs_struct = new SetFreq[nt_max_corr];
      std::memset(local_bs, 0, sizeof(uint64_t) * nt_max_corr * featbits);
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        local_bs_struct[i].init(local_bs + i * featbits, 0, featbits);
      }
      if (!recursive) {
        for (int32_t i = 0; i < nt_max_corr; ++i) {
          queues[i].init(INIT_CYCLIC_SIZE, features);
        }
      }
    }
    if (recursive && !vs) {
      vs = new uint64_t[nt_max_corr * rowbits];
    }
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      job_done[i] = false;
      best_corr_threads[i] = std::numeric_limits<c_real>::min();
      max_queues[i] = 0;
      total_processeds[i] = 0;
    }
    std::vector<std::future<void>> results;
    std::vector<FMPJ*> local_jobs;
    std::vector<FindMaxPosJobRec*> local_jobs_rec;
    if (recursive) {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxPosJobRec* j = new FindMaxPosJobRec(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        local_jobs_rec.push_back(j);
        jobs_pos_rec[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_pos_rec(i); }));
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxPosJob* j = new FindMaxPosJob(sx, y, b, samples, features,
            i, best_corr, best_corr_threads + thread_no,
            local_bs_struct + thread_no, delta, queues + thread_no,
            max_queues + thread_no, total_processeds + thread_no);
        local_jobs.push_back(j);
        jobs_pos[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_pos(i); }));
      }
    }
    
    do {
      c_real new_best_corr = 0.0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        c_real bct = best_corr_threads[i];
        if (bct > new_best_corr) {
          new_best_corr = bct;
        }
      }
      if (best_corr < new_best_corr) {
        best_corr = new_best_corr;
      }
      jobs_done = 0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        jobs_done += job_done[i];
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } while (jobs_done != nt_max_corr);
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      results[i].get();
    }
    
    c_real new_best_corr = 0.0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      c_real bct = best_corr_threads[i];
      if (bct > new_best_corr) {
        new_best_corr = bct;
      }
    }
    if (best_corr < new_best_corr) {
      best_corr = new_best_corr;
    }
    SetFreq b_struct;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      if (best_corr_threads[i] == best_corr) {
        b_struct.copy(local_bs_struct[i]);
      }
    }
    
    for (int64_t i = 0; i < samples; i++) {
      fb[i] = leq(b_struct, sx, i);
    }
    std::memcpy(b, b_struct.data, featbits * sizeof(uint64_t));
    
    #if GET_STATS
    int64_t new_max_queue = 0;
    int64_t new_total_processed = 0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      new_max_queue = std::max(new_max_queue, max_queues[i]);
      new_total_processed += total_processeds[i];
    }
    max_queue = new_max_queue;
    total_processed = new_total_processed;
    #endif
    
    for (FMPJ* job : local_jobs) {
      delete job;
    }
    for (FindMaxPosJobRec* job : local_jobs_rec) {
      delete job;
    }
    
    return best_corr;
  }
  
  c_real cmcpos_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmcpos_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max_pos(int32_t id) {
    FMPJ* function = nullptr;
    while (!jobs_pos[id].empty()) {
      function = jobs_pos[id].front();
      jobs_pos[id].pop();
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
  }
  
  void run_find_max_pos_rec(int32_t id) {
    FindMaxPosJobRec* function = nullptr;
    while (!jobs_pos_rec[id].empty()) {
      function = jobs_pos_rec[id].front();
      jobs_pos_rec[id].pop();
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
  }
  
  void stop() {
    if (!stopped) {
      delete[] best_corr_threads;
      delete[] jobs;
      delete[] jobs_rec;
      delete[] jobs_pos;
      delete[] jobs_pos_rec;
      delete[] job_done;
      delete[] max_queues;
      delete[] total_processeds;
      if (local_bs) {
        delete[] local_bs_struct;
        delete[] local_bs;
        local_bs_struct = nullptr;
        local_bs = nullptr;
      }
      delete[] queues;
      if (vs) {
        delete[] vs;
      }
    }
    stopped = true;
  }
  
  ~ThreadPool() {
    stop();
  }

  ThreadPool(int32_t n) : n_threads(n) {
    best_corr_threads = new c_real[n_threads];
    jobs = new std::queue<FMJ*>[n_threads];
    jobs_pos = new std::queue<FMPJ*>[n_threads];
    jobs_rec = new std::queue<FindMaxJobRec*>[n_threads];
    jobs_pos_rec = new std::queue<FindMaxPosJobRec*>[n_threads];
    job_done = new bool[n_threads];
    local_bs = nullptr;
    local_bs_struct = nullptr;
    running = true;
    stopped = false;
    queues = new SetQueue[n_threads];
    vs = nullptr;
    max_queues = new int64_t[n_threads];
    total_processeds = new int64_t[n_threads];
  };
  

};
























#else



#if CYCLIC_BUFFER
#define FMJ FindMaxJobCyclic
#define FMPJ FindMaxPosJobCyclic
#else
#define FMJ FindMaxJob
#define FMPJ FindMaxPosJob
#endif


#if GOOD_OLD_IMPL

// ThreadPool will be initialized with parameterized number of threads,
// but currently not all might be active during the execution.
// To get max_corr, min(n_threads, features) are used
class ThreadPool {

private:

  struct FindMaxJob {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmc_part_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #else
      cmc_part(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #endif
    }
    #if JOB_STEALING
    void launder(uint64_t* local_b, c_real* best_corr_thread,
                int64_t* max_queue, int64_t* total_processed) {
      this->local_b = local_b;
      this->best_corr_thread = best_corr_thread;
      this->max_queue = max_queue;
      this->total_processed = total_processed;
    }
    #endif
    FindMaxJob(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxJobCyclic {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t** q;
    int32_t** q_val;
    c_real** mus;
    int32_t* q_size;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmc_part_avx(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #else
      cmc_part(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #endif
    }
    #if JOB_STEALING
    void launder(uint64_t* local_b, c_real* best_corr_thread, uint64_t** q,
                int32_t** q_val, c_real** mus, int32_t* q_size,
                int64_t* max_queue, int64_t* total_processed) {
      this->local_b = local_b;
      this->best_corr_thread = best_corr_thread;
      this->q = q;
      this->q_val = q_val;
      this->mus = mus;
      this->q_size = q_size;
      this->max_queue = max_queue;
      this->total_processed = total_processed;
    }
    #endif
    FindMaxJobCyclic(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        q(q), q_val(q_val), mus(mus), q_size(q_size),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxJobRec {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t* v;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmc_part_rec_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #else
      cmc_part_rec(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #endif
    }
    #if JOB_STEALING
    void launder(uint64_t* local_b, c_real* best_corr_thread,
                int64_t* max_queue, int64_t* total_processed) {
      this->local_b = local_b;
      this->best_corr_thread = best_corr_thread;
      this->max_queue = max_queue;
      this->total_processed = total_processed;
    }
    #endif
    FindMaxJobRec(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t* v, int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta), v(v),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJob {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmcpos_part_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #else
      cmcpos_part(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #endif
    }
    #if JOB_STEALING
    void launder(uint64_t* local_b, c_real* best_corr_thread,
                int64_t* max_queue, int64_t* total_processed) {
      this->local_b = local_b;
      this->best_corr_thread = best_corr_thread;
      this->max_queue = max_queue;
      this->total_processed = total_processed;
    }
    #endif
    FindMaxPosJob(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJobCyclic {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t** q;
    int32_t** q_val;
    c_real** mus;
    int32_t* q_size;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmcpos_part_avx(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #else
      cmcpos_part(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #endif
    }
    #if JOB_STEALING
    void launder(uint64_t* local_b, c_real* best_corr_thread, uint64_t** q,
                int32_t** q_val, c_real** mus, int32_t* q_size,
                int64_t* max_queue, int64_t* total_processed) {
      this->local_b = local_b;
      this->best_corr_thread = best_corr_thread;
      this->q = q;
      this->q_val = q_val;
      this->mus = mus;
      this->q_size = q_size;
      this->max_queue = max_queue;
      this->total_processed = total_processed;
    }
    #endif
    FindMaxPosJobCyclic(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        q(q), q_val(q_val), mus(mus), q_size(q_size),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJobRec {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t* v;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmcpos_part_rec_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #else
      cmcpos_part_rec(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #endif
    }
    #if JOB_STEALING
    void launder(uint64_t* local_b, c_real* best_corr_thread,
                int64_t* max_queue, int64_t* total_processed) {
      this->local_b = local_b;
      this->best_corr_thread = best_corr_thread;
      this->max_queue = max_queue;
      this->total_processed = total_processed;
    }
    #endif
    FindMaxPosJobRec(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t* v, int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta), v(v),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };

  int32_t n_threads;
  int32_t jobs_done;
  bool* job_done;
  std::vector<std::thread> threads;
  uint64_t* local_bs;
  bool running;
  c_real* best_corr_threads;
  uint64_t* vs;

  std::queue<FMJ*>* jobs;
  std::queue<FMPJ*>* jobs_pos;
  std::queue<FindMaxJobRec*>* jobs_rec;
  std::queue<FindMaxPosJobRec*>* jobs_pos_rec;
  
  uint64_t** qs;
  int32_t** q_vals;
  c_real** mus;
  int32_t* q_sizes;
  bool stopped;
  
  int64_t* max_queues;
  int64_t* total_processeds;
  
  int32_t nt_max_corr;
  int32_t featbits;
  
  #if JOB_STEALING
  struct SimpleLock {
    std::atomic<bool> locked;
    SimpleLock() {
      locked = false;
    }
    void lock() {
      bool expect;
      do {
        expect = false;
      } while(!locked.compare_exchange_strong(expect, true, std::memory_order_acq_rel));
    }
    void unlock() {
      bool expect;
      do {
        expect = true;
      } while(!locked.compare_exchange_strong(expect, false, std::memory_order_acq_rel));
    }
  };
  
  //~ std::vector<std::mutex> qmutex;
  std::vector<SimpleLock> qmutex;
  #endif
  
public:

  bool is_running() {
    return running;
  }

  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive, double delta,
      int64_t& max_queue, int64_t& total_processed) {

    size_t n = features;
    featbits = ((features + 63) >> 6);
    c_real best_corr = 0;
    
    #if BIT_AVX
    int32_t rowbits = (((features + 255) >> 8) << 2);
    #else
    int32_t rowbits = featbits;
    #endif
    
    nt_max_corr = std::min(n_threads, features);
    //~ int32_t nt_update = std::min(n_threads, samples);

    jobs_done = 0;
    
    if (!local_bs) {
      local_bs = new uint64_t[nt_max_corr * featbits];
      std::memset(local_bs, 0, sizeof(uint64_t) * nt_max_corr * featbits);
      #if CYCLIC_BUFFER
      if (!recursive) {
        for (int32_t i = 0; i < nt_max_corr; ++i) {
          qs[i] = new uint64_t[INIT_CYCLIC_SIZE * rowbits];
          q_vals[i] = new int32_t[INIT_CYCLIC_SIZE];
          mus[i] = new c_real[INIT_CYCLIC_SIZE];
          q_sizes[i] = INIT_CYCLIC_SIZE;
        }
      }
      #endif
    }
    
    #if JOB_STEALING
    if (qmutex.size() == 0) {
      //~ std::vector<std::mutex> tmutex(nt_max_corr);
      std::vector<SimpleLock> tmutex(nt_max_corr);
      qmutex.swap(tmutex);
    }
    #endif
    
    if (recursive && !vs) {
      vs = new uint64_t[nt_max_corr * rowbits];
    }
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      job_done[i] = false;
      best_corr_threads[i] = 0.0;
      max_queues[i] = 0;
      total_processeds[i] = 0;
    }
    std::vector<std::future<void>> results;
    std::vector<FMJ*> local_jobs;
    std::vector<FindMaxJobRec*> local_jobs_rec;
    if (recursive) {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxJobRec* j = new FindMaxJobRec(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        local_jobs_rec.push_back(j);
        jobs_rec[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_rec(i); }));
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        #if CYCLIC_BUFFER
        FindMaxJobCyclic* j = new FindMaxJobCyclic(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            qs + thread_no, q_vals + thread_no, mus + thread_no, q_sizes + thread_no,
            max_queues + thread_no, total_processeds + thread_no);
        #else
        FindMaxJob* j = new FindMaxJob(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            max_queues + thread_no, total_processeds + thread_no);
        #endif
        local_jobs.push_back(j);
        jobs[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max(i); }));
      }
    }
    
    do {
      c_real new_best_corr = 0.0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        c_real bct = best_corr_threads[i];
        if (std::fabs(bct) > std::fabs(new_best_corr)) {
          new_best_corr = bct;
        }
      }
      if (std::fabs(best_corr) < std::fabs(new_best_corr)) {
        best_corr = new_best_corr;
      }
      jobs_done = 0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        jobs_done += job_done[i];
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } while (jobs_done != nt_max_corr);
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      results[i].get();
    }
    
    c_real new_best_corr = 0.0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      c_real bct = best_corr_threads[i];
      if (std::fabs(bct) > std::fabs(new_best_corr)) {
        new_best_corr = bct;
      }
    }
    if (std::fabs(best_corr) < std::fabs(new_best_corr)) {
      best_corr = new_best_corr;
    }

    for (int32_t i = 0; i < nt_max_corr; ++i) {
      if (best_corr_threads[i] == best_corr) {
        std::memcpy(b, local_bs + i * featbits, featbits * sizeof(uint64_t));
      }
    }
    
        //~ std::cout << "BEST: " << best_corr << "\n";
    
    for (int64_t i = 0; i < samples; i++) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * rowbits + j] & b[j]) != b[j]) {
          break;
        }
      }
        fb[i] = (j == featbits);
              //~ std::cout << fb[i];
    }
    //~ std::cout << std::endl;
    
    //~ for (int64_t i = 0; i < samples; ++i) {
      //~ std::cout << y[i] << " ";
    //~ }
    //~ std::cout << "\n";
    
    #if GET_STATS
    int64_t new_max_queue = 0;
    int64_t new_total_processed = 0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      new_max_queue = std::max(new_max_queue, max_queues[i]);
      new_total_processed += total_processeds[i];
    }
    max_queue = new_max_queue;
    total_processed = new_total_processed;
    #endif
    
    // will this suffer much from false sharing? ---> could be, this is actually pretty slow
    //~ int32_t spt = (samples + nt_update - 1) / nt_update;
    //~ std::vector<std::future<void>> results_update;
    //~ for (int32_t i = 0; i < nt_update; ++i) {
      //~ results_update.push_back(std::async([=]{
          //~ return update_freq_range(x, b, fb, i * spt, std::min(i * spt + spt, samples), featbits);
      //~ }));
    //~ }
    //~ for (int32_t i = 0; i < nt_update; ++i) {
      //~ results_update[i].get();
    //~ }
    
    for (FMJ* job : local_jobs) {
      delete job;
    }
    for (FindMaxJobRec* job : local_jobs_rec) {
      delete job;
    }
    
    return best_corr;
  }
  
  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmc_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max(int32_t id) {
    FMJ* function = nullptr;
    while (!jobs[id].empty()) {
      #if JOB_STEALING
      qmutex[id].lock();
      if (!jobs[id].empty()) {
      #endif
        function = jobs[id].front();
        jobs[id].pop();
      #if JOB_STEALING
      }
      qmutex[id].unlock();
      #endif
      if (function) {
        function->call();
        function = nullptr;
      }
    }
    job_done[id] = true;
    #if JOB_STEALING
    bool ndone = true;
    while (ndone) {
      ndone = false;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        if (i != id) {
          qmutex[i].lock();
          if (!jobs[i].empty()) {
            function = jobs[i].front();
            jobs[i].pop();
          }
          qmutex[i].unlock();
          if (function) {
            ndone = true;
            #if CYCLIC_BUFFER
            function->launder(local_bs + id * featbits, best_corr_threads + id,
                             qs + id, q_vals + id, mus + id, q_sizes + id,
                             max_queues + id, total_processeds + id);
            #else
            function->launder(local_bs + id * featbits, best_corr_threads + id,
                 max_queues + id, total_processeds + id);
            #endif
            function->call();
            function = nullptr;
          }
        }
        if (jobs_done == nt_max_corr) {
          break;
        }
      }
    }
    #endif
  }

  void run_find_max_rec(int32_t id) {
    FindMaxJobRec* function = nullptr;
    while (!jobs_rec[id].empty()) {
      #if JOB_STEALING
      qmutex[id].lock();
      if (!jobs_rec[id].empty()) {
      #endif
        function = jobs_rec[id].front();
        jobs_rec[id].pop();
      #if JOB_STEALING
      }
      qmutex[id].unlock();
      #endif
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
    #if JOB_STEALING
    bool ndone = true;
    while (ndone) {
      ndone = false;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        if (i != id) {
          qmutex[i].lock();
          if (!jobs_rec[i].empty()) {
            function = jobs_rec[i].front();
            jobs_rec[i].pop();
          }
          qmutex[i].unlock();
          if (function) {
            ndone = true;
            function->launder(local_bs + id * featbits, best_corr_threads + id,
                 max_queues + id, total_processeds + id);
            function->call();
            function = nullptr;
          }
        }
      }
    }
    #endif
  }
  
  c_real cmcpos_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive, double delta,
      int64_t& max_queue, int64_t& total_processed) {

    size_t n = features;
    int32_t featbits = ((features + 63) >> 6);
    c_real best_corr = std::numeric_limits<c_real>::min();
    
    #if BIT_AVX
    int32_t rowbits = (((features + 255) >> 8) << 2);
    #else
    int32_t rowbits = featbits;
    #endif
    
    int32_t nt_max_corr = std::min(n_threads, features);
    //~ int32_t nt_update = std::min(n_threads, samples);

    jobs_done = 0;
    
    if (!local_bs) {
      local_bs = new uint64_t[nt_max_corr * featbits];
      std::memset(local_bs, 0, sizeof(uint64_t) * nt_max_corr * featbits);
      #if CYCLIC_BUFFER
      if (!recursive) {
        for (int32_t i = 0; i < nt_max_corr; ++i) {
          qs[i] = new uint64_t[INIT_CYCLIC_SIZE * rowbits];
          q_vals[i] = new int32_t[INIT_CYCLIC_SIZE];
          mus[i] = new c_real[INIT_CYCLIC_SIZE];
          q_sizes[i] = INIT_CYCLIC_SIZE;
        }
      }
      #endif
    }
    if (recursive && !vs) {
      vs = new uint64_t[nt_max_corr * rowbits];
    }
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      job_done[i] = false;
      best_corr_threads[i] = std::numeric_limits<c_real>::min();
      max_queues[i] = 0;
      total_processeds[i] = 0;
    }
    std::vector<std::future<void>> results;
    std::vector<FMPJ*> local_jobs;
    std::vector<FindMaxPosJobRec*> local_jobs_rec;
    if (recursive) {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxPosJobRec* j = new FindMaxPosJobRec(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        local_jobs_rec.push_back(j);
        jobs_pos_rec[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_pos_rec(i); }));
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        #if CYCLIC_BUFFER
        FindMaxPosJobCyclic* j = new FindMaxPosJobCyclic(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            qs + thread_no, q_vals + thread_no, mus + thread_no, q_sizes + thread_no,
            max_queues + thread_no, total_processeds + thread_no);
        #else
        FindMaxPosJob* j = new FindMaxPosJob(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            max_queues + thread_no, total_processeds + thread_no);
        #endif
        local_jobs.push_back(j);
        jobs_pos[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_pos(i); }));
      }
    }
    
    do {
      c_real new_best_corr = 0.0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        c_real bct = best_corr_threads[i];
        if (bct > new_best_corr) {
          new_best_corr = bct;
        }
      }
      if (best_corr < new_best_corr) {
        best_corr = new_best_corr;
      }
      jobs_done = 0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        jobs_done += job_done[i];
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } while (jobs_done != nt_max_corr);
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      results[i].get();
    }
    
    c_real new_best_corr = 0.0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      c_real bct = best_corr_threads[i];
      if (bct > new_best_corr) {
        new_best_corr = bct;
      }
    }
    if (best_corr < new_best_corr) {
      best_corr = new_best_corr;
    }
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      if (best_corr_threads[i] == best_corr) {
        std::memcpy(b, local_bs + i * featbits, featbits * sizeof(uint64_t));
      }
    }
    
        //~ std::cout << "BEST: " << best_corr << "\n";
    
    for (int64_t i = 0; i < samples; i++) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * rowbits + j] & b[j]) != b[j]) {
          break;
        }
      }
      fb[i] = (j == featbits);
    }
    
    #if GET_STATS
    int64_t new_max_queue = 0;
    int64_t new_total_processed = 0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      new_max_queue = std::max(new_max_queue, max_queues[i]);
      new_total_processed += total_processeds[i];
    }
    max_queue = new_max_queue;
    total_processed = new_total_processed;
    #endif
    
    for (FMPJ* job : local_jobs) {
      delete job;
    }
    for (FindMaxPosJobRec* job : local_jobs_rec) {
      delete job;
    }
    
    return best_corr;
  }
  
  c_real cmcpos_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmcpos_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max_pos(int32_t id) {
    FMPJ* function = nullptr;
    while (!jobs_pos[id].empty()) {
      #if JOB_STEALING
      qmutex[id].lock();
      if (!jobs_pos[id].empty()) {
      #endif
        function = jobs_pos[id].front();
        jobs_pos[id].pop();
      #if JOB_STEALING
      }
      qmutex[id].unlock();
      #endif
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
    #if JOB_STEALING
    bool ndone = true;
    while (ndone) {
      ndone = false;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        if (i != id) {
          qmutex[i].lock();
          if (!jobs_pos[i].empty()) {
            function = jobs_pos[i].front();
            jobs_pos[i].pop();
          }
          qmutex[i].unlock();
          if (function) {
            ndone = true;
            #if CYCLIC_BUFFER
            function->launder(local_bs + id * featbits, best_corr_threads + id,
                             qs + id, q_vals + id, mus + id, q_sizes + id,
                             max_queues + id, total_processeds + id);
            #else
            function->launder(local_bs + id * featbits, best_corr_threads + id,
                 max_queues + id, total_processeds + id);
            #endif
            function->call();
            function = nullptr;
          }
        }
      }
    }
    #endif
  }
  
  void run_find_max_pos_rec(int32_t id) {
    FindMaxPosJobRec* function = nullptr;
    while (!jobs_pos_rec[id].empty()) {
      #if JOB_STEALING
      qmutex[id].lock();
      if (!jobs_pos_rec[id].empty()) {
      #endif
        function = jobs_pos_rec[id].front();
        jobs_pos_rec[id].pop();
      #if JOB_STEALING
      }
      qmutex[id].unlock();
      #endif
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
    #if JOB_STEALING
    bool ndone = true;
    while (ndone) {
      ndone = false;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        if (i != id) {
          qmutex[i].lock();
          if (!jobs_pos_rec[i].empty()) {
            function = jobs_pos_rec[i].front();
            jobs_pos_rec[i].pop();
          }
          qmutex[i].unlock();
          if (function) {
            ndone = true;
            function->launder(local_bs + id * featbits, best_corr_threads + id,
                 max_queues + id, total_processeds + id);
            function->call();
            function = nullptr;
          }
        }
      }
    }
    #endif
  }
  
  void stop() {
    if (!stopped) {
      delete[] best_corr_threads;
      delete[] jobs;
      delete[] jobs_rec;
      delete[] jobs_pos;
      delete[] jobs_pos_rec;
      delete[] job_done;
      delete[] max_queues;
      delete[] total_processeds;
      if (local_bs) {
        delete[] local_bs;
        local_bs = nullptr;
      }
      #if CYCLIC_BUFFER
      for (int32_t i = 0; i < n_threads; ++i) {
        if (qs[i]) {
          delete[] qs[i];
        }
        if (q_vals[i]) {
          delete[] q_vals[i];
        }
        if (mus[i]) {
          delete[] mus[i];
        }
      }
      delete[] qs;
      delete[] q_vals;
      delete[] mus;
      delete[] q_sizes;
      #endif
      if (vs) {
        delete[] vs;
      }
    }
    stopped = true;
  }
  
  ~ThreadPool() {
    stop();
  }

  ThreadPool(int32_t n) : n_threads(n) {
    //~ std::cout << "THREADS " << n << "\n";
    best_corr_threads = new c_real[n_threads];
    jobs = new std::queue<FMJ*>[n_threads];
    jobs_pos = new std::queue<FMPJ*>[n_threads];
    jobs_rec = new std::queue<FindMaxJobRec*>[n_threads];
    jobs_pos_rec = new std::queue<FindMaxPosJobRec*>[n_threads];
    job_done = new bool[n_threads];
    local_bs = nullptr;
    running = true;
    stopped = false;
    #if CYCLIC_BUFFER
    qs = new uint64_t*[n_threads];
    q_vals = new int32_t*[n_threads];
    mus = new c_real*[n_threads];
    q_sizes = new int32_t[n_threads];
    for (int32_t i = 0; i < n_threads; ++i) {
      qs[i] = nullptr;
      q_vals[i] = nullptr;
      mus[i] = nullptr;
      q_sizes[i] = 0;
    }
    #endif
    vs = nullptr;
    max_queues = new int64_t[n_threads];
    total_processeds = new int64_t[n_threads];
  };
  

};





















#else

// TODO recursive should be a boolean parameter, not a precompiler variable!

// ThreadPool will be initialized with parameterized number of threads,
// but currently not all might be active during the execution.
// To get max_corr, min(n_threads, features) are used
class ThreadPool {

// todo padding of threadpayload and job structures

private:


  struct ThreadPayload {
    uint64_t* local_b;
    uint64_t* v;
    uint64_t* q;
    int32_t* q_val;
    c_real* mus;
    int32_t q_size;
    c_real best_corr_thread;
    bool job_done;
    uint64_t max_queue;
    uint64_t total_processed;
    char padding[48];

    //todo adjust padding to the sizeof c_real
  };

  struct FindMaxJob {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmc_part_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #else
      cmc_part(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #endif
    }
    FindMaxJob(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxJobCyclic {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    double delta;
    ThreadPayload* payload;
    void call() {
      #if BIT_AVX
      //~ cmc_part_avx(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          //~ local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #else
      cmc_part(x, y, b, samples, features, first_q_val, best_corr, payload->best_corr_thread,
          payload->local_b, delta, payload->q, payload->q_val, payload->mus, payload->q_size, payload->max_queue, payload->total_processed);
      #endif
    }
    FindMaxJobCyclic(uint64_t* x, c_real* y, uint64_t* b,
        c_real& best_corr, int32_t samples, int32_t features, int32_t first_q_val,
        double delta, ThreadPayload* payload): x(x), y(y), b(b),
        best_corr(best_corr), samples(samples), features(features),
        first_q_val(first_q_val), delta(delta), payload(payload) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxJobRec {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t* v;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmc_part_rec_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #else
      cmc_part_rec(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #endif
    }
    FindMaxJobRec(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t* v, int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta), v(v),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJob {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmcpos_part_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #else
      cmcpos_part(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
      #endif
    }
    FindMaxPosJob(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJobCyclic {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t** q;
    int32_t** q_val;
    c_real** mus;
    int32_t* q_size;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmcpos_part_avx(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #else
      cmcpos_part(x, y, b, samples, features, first_q_val, best_corr, best_corr_thread,
          local_b, delta, q, q_val, mus, q_size, max_queue, total_processed);
      #endif
    }
    FindMaxPosJobCyclic(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta),
        q(q), q_val(q_val), mus(mus), q_size(q_size),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };
  
  struct FindMaxPosJobRec {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    c_real& best_corr;
    c_real* best_corr_thread;
    int32_t samples;
    int32_t features;
    int32_t first_q_val;
    uint64_t* local_b;
    double delta;
    uint64_t* v;
    int64_t* max_queue;
    int64_t* total_processed;
    void call() {
      #if BIT_AVX
      cmcpos_part_rec_avx(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #else
      cmcpos_part_rec(x, y, b, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, v, max_queue, total_processed);
      #endif
    }
    FindMaxPosJobRec(uint64_t* x, c_real* y, uint64_t* b,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        uint64_t* v, int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b), delta(delta), v(v),
        max_queue(max_queue), total_processed(total_processed) {
          /*std::cout << "Created " + std::to_string(first_q_val) + "\n";*/
        }
  };

  int32_t n_threads;
  int32_t jobs_done;
  bool* job_done;
  std::vector<std::thread> threads;
  uint64_t* local_bs;
  bool running;
  c_real* best_corr_threads;
  uint64_t* vs;

  std::queue<FMJ*>* jobs;
  std::queue<FMPJ*>* jobs_pos;
  std::queue<FindMaxJobRec*>* jobs_rec;
  std::queue<FindMaxPosJobRec*>* jobs_pos_rec;
  
  uint64_t** qs;
  int32_t** q_vals;
  c_real** mus;
  int32_t* q_sizes;
  bool stopped;
  
  int64_t* max_queues;
  int64_t* total_processeds;
  
  ThreadPayload* payloads;
  
public:

  bool is_running() {
    return running;
  }

  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive, double delta,
      int64_t& max_queue, int64_t& total_processed) {

    size_t n = features;
    int32_t featbits = ((features + 63) >> 6);
    c_real best_corr = 0;
    
    #if BIT_AVX
    int32_t rowbits = (((features + 255) >> 8) << 2);
    #else
    int32_t rowbits = featbits;
    #endif
    
    int32_t nt_max_corr = std::min(n_threads, features);
    //~ int32_t nt_update = std::min(n_threads, samples);

    jobs_done = 0;
    
    if (!payloads[0].local_b) {
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        payloads[i].local_b = new uint64_t[featbits];
        std::memset(payloads[i].local_b, 0, sizeof(uint64_t) * featbits);
        #if CYCLIC_BUFFER
        if (!recursive) {
            payloads[i].q = new uint64_t[INIT_CYCLIC_SIZE * rowbits];
            payloads[i].q_val = new int32_t[INIT_CYCLIC_SIZE];
            payloads[i].mus = new c_real[INIT_CYCLIC_SIZE];
            payloads[i].q_size = INIT_CYCLIC_SIZE;
        }
        #endif
      }
    }
    
    if (recursive && !payloads[0].v) {
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        payloads[i].v = new uint64_t[rowbits];
      }
    }
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      payloads[i].job_done = false;
      payloads[i].best_corr_thread = 0.0;
      payloads[i].max_queue = 0;
      payloads[i].total_processed = 0;
    }
    std::vector<std::future<void>> results;
    std::vector<FMJ*> local_jobs;
    std::vector<FindMaxJobRec*> local_jobs_rec;
    if (recursive) {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxJobRec* j = new FindMaxJobRec(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        local_jobs_rec.push_back(j);
        jobs_rec[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_rec(i); }));
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        #if CYCLIC_BUFFER
        FindMaxJobCyclic* j = new FindMaxJobCyclic(x, y, b, best_corr, samples, features, i,
            delta, payloads + thread_no);
        #else
        FindMaxJob* j = new FindMaxJob(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            max_queues + thread_no, total_processeds + thread_no);
        #endif
        local_jobs.push_back(j);
        jobs[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max(i); }));
      }
    }

    do {
      c_real new_best_corr = 0.0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        c_real bct = payloads[i].best_corr_thread;
        if (std::fabs(bct) > std::fabs(new_best_corr)) {
          new_best_corr = bct;
        }
      }
      if (std::fabs(best_corr) < std::fabs(new_best_corr)) {
        best_corr = new_best_corr;
      }
      jobs_done = 0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        jobs_done += payloads[i].job_done;
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } while (jobs_done != nt_max_corr);
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      results[i].get();
    }
    
    c_real new_best_corr = 0.0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      c_real bct = payloads[i].best_corr_thread;
      if (std::fabs(bct) > std::fabs(new_best_corr)) {
        new_best_corr = bct;
      }
    }
    if (std::fabs(best_corr) < std::fabs(new_best_corr)) {
      best_corr = new_best_corr;
    }

    for (int32_t i = 0; i < nt_max_corr; ++i) {
      if (payloads[i].best_corr_thread == best_corr) {
        std::memcpy(b, payloads[i].local_b, featbits * sizeof(uint64_t));
      }
    }
    
    for (int64_t i = 0; i < samples; i++) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * rowbits + j] & b[j]) != b[j]) {
          break;
        }
      }
        fb[i] = (j == featbits);
    }
    
    #if GET_STATS
    int64_t new_max_queue = 0;
    int64_t new_total_processed = 0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      new_max_queue = std::max(new_max_queue, payloads[i].max_queue);
      new_total_processed += payloads[i].total_processed;
    }
    max_queue = new_max_queue;
    total_processed = new_total_processed;
    #endif
    
    // will this suffer much from false sharing? ---> could be, this is actually pretty slow
    //~ int32_t spt = (samples + nt_update - 1) / nt_update;
    //~ std::vector<std::future<void>> results_update;
    //~ for (int32_t i = 0; i < nt_update; ++i) {
      //~ results_update.push_back(std::async([=]{
          //~ return update_freq_range(x, b, fb, i * spt, std::min(i * spt + spt, samples), featbits);
      //~ }));
    //~ }
    //~ for (int32_t i = 0; i < nt_update; ++i) {
      //~ results_update[i].get();
    //~ }
    
    for (FMJ* job : local_jobs) {
      delete job;
    }
    for (FindMaxJobRec* job : local_jobs_rec) {
      delete job;
    }
    
    return best_corr;
  }
  
  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmc_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max(int32_t id) {
    while (!jobs[id].empty()) {
      FMJ* function = jobs[id].front();
      jobs[id].pop();
      if (function) {
        function->call();
      }
    }
    payloads[id].job_done = true;
  }
  
  void run_find_max_rec(int32_t id) {
    while (!jobs_rec[id].empty()) {
      FindMaxJobRec* function = jobs_rec[id].front();
      jobs_rec[id].pop();
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
  }
  
  c_real cmcpos_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive, double delta,
      int64_t& max_queue, int64_t& total_processed) {

    size_t n = features;
    int32_t featbits = ((features + 63) >> 6);
    c_real best_corr = std::numeric_limits<c_real>::min();
    
    #if BIT_AVX
    int32_t rowbits = (((features + 255) >> 8) << 2);
    #else
    int32_t rowbits = featbits;
    #endif
    
    int32_t nt_max_corr = std::min(n_threads, features);
    //~ int32_t nt_update = std::min(n_threads, samples);

    jobs_done = 0;
    
    if (!local_bs) {
      local_bs = new uint64_t[nt_max_corr * featbits];
      std::memset(local_bs, 0, sizeof(uint64_t) * nt_max_corr * featbits);
      #if CYCLIC_BUFFER
      if (!recursive) {
        for (int32_t i = 0; i < nt_max_corr; ++i) {
          qs[i] = new uint64_t[INIT_CYCLIC_SIZE * rowbits];
          q_vals[i] = new int32_t[INIT_CYCLIC_SIZE];
          mus[i] = new c_real[INIT_CYCLIC_SIZE];
          q_sizes[i] = INIT_CYCLIC_SIZE;
        }
      }
      #endif
    }
    if (recursive && !vs) {
      vs = new uint64_t[nt_max_corr * rowbits];
    }
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      job_done[i] = false;
      best_corr_threads[i] = std::numeric_limits<c_real>::min();
      max_queues[i] = 0;
      total_processeds[i] = 0;
    }
    std::vector<std::future<void>> results;
    std::vector<FMPJ*> local_jobs;
    std::vector<FindMaxPosJobRec*> local_jobs_rec;
    if (recursive) {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        FindMaxPosJobRec* j = new FindMaxPosJobRec(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        local_jobs_rec.push_back(j);
        jobs_pos_rec[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_pos_rec(i); }));
      }
    } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        #if CYCLIC_BUFFER
        FindMaxPosJobCyclic* j = new FindMaxPosJobCyclic(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            qs + thread_no, q_vals + thread_no, mus + thread_no, q_sizes + thread_no,
            max_queues + thread_no, total_processeds + thread_no);
        #else
        FindMaxPosJob* j = new FindMaxPosJob(x, y, b, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            max_queues + thread_no, total_processeds + thread_no);
        #endif
        local_jobs.push_back(j);
        jobs_pos[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max_pos(i); }));
      }
    }
    
    do {
      c_real new_best_corr = 0.0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        c_real bct = best_corr_threads[i];
        if (bct > new_best_corr) {
          new_best_corr = bct;
        }
      }
      if (best_corr < new_best_corr) {
        best_corr = new_best_corr;
      }
      jobs_done = 0;
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        jobs_done += job_done[i];
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    } while (jobs_done != nt_max_corr);
    
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      results[i].get();
    }
    
    c_real new_best_corr = 0.0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      c_real bct = best_corr_threads[i];
      if (bct > new_best_corr) {
        new_best_corr = bct;
      }
    }
    if (best_corr < new_best_corr) {
      best_corr = new_best_corr;
    }
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      if (best_corr_threads[i] == best_corr) {
        std::memcpy(b, local_bs + i * featbits, featbits * sizeof(uint64_t));
      }
    }
    
    for (int64_t i = 0; i < samples; i++) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * rowbits + j] & b[j]) != b[j]) {
          break;
        }
      }
      fb[i] = (j == featbits);
    }
    
    #if GET_STATS
    int64_t new_max_queue = 0;
    int64_t new_total_processed = 0;
    for (int32_t i = 0; i < nt_max_corr; ++i) {
      new_max_queue = std::max(new_max_queue, max_queues[i]);
      new_total_processed += total_processeds[i];
    }
    max_queue = new_max_queue;
    total_processed = new_total_processed;
    #endif
    
    for (FMPJ* job : local_jobs) {
      delete job;
    }
    for (FindMaxPosJobRec* job : local_jobs_rec) {
      delete job;
    }
    
    return best_corr;
  }
  
  c_real cmcpos_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmcpos_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max_pos(int32_t id) {
    while (!jobs_pos[id].empty()) {
      FMPJ* function = jobs_pos[id].front();
      jobs_pos[id].pop();
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
  }
  
  void run_find_max_pos_rec(int32_t id) {
    while (!jobs_pos_rec[id].empty()) {
      FindMaxPosJobRec* function = jobs_pos_rec[id].front();
      jobs_pos_rec[id].pop();
      if (function) {
        function->call();
      }
    }
    job_done[id] = true;
  }
  
  void stop() {
    if (!stopped) {

      for (int32_t i = 0; i < n_threads; ++i) {
        if (payloads[i].q) {
          delete[] payloads[i].q;
        }
        if (payloads[i].q_val) {
          delete[] payloads[i].q_val;
        }
        if (payloads[i].mus) {
          delete[] payloads[i].mus;
        }
        if (payloads[i].local_b) {
          delete[] payloads[i].local_b;
        }
        if (payloads[i].v) {
          delete[] payloads[i].v;
        }
      }
      delete[] payloads;
      
      delete[] best_corr_threads;
      delete[] jobs;
      delete[] jobs_rec;
      delete[] jobs_pos;
      delete[] jobs_pos_rec;
      delete[] job_done;
      delete[] max_queues;
      delete[] total_processeds;
      if (local_bs) {
        delete[] local_bs;
        local_bs = nullptr;
      }
      #if CYCLIC_BUFFER
      for (int32_t i = 0; i < n_threads; ++i) {
        if (qs[i]) {
          delete[] qs[i];
        }
        if (q_vals[i]) {
          delete[] q_vals[i];
        }
        if (mus[i]) {
          delete[] mus[i];
        }
      }
      delete[] qs;
      delete[] q_vals;
      delete[] mus;
      delete[] q_sizes;
      #endif
      if (vs) {
        delete[] vs;
      }
    }
    stopped = true;
  }
  
  ~ThreadPool() {
    stop();
  }

  ThreadPool(int32_t n) : n_threads(n) {
    //~ std::cout << sizeof(ThreadPayload) << "\n";
    payloads = new ThreadPayload[n_threads];
    for (int32_t i = 0; i < n_threads; ++i) {
      payloads[i].q = nullptr;
      payloads[i].q_val = nullptr;
      payloads[i].mus = nullptr;
      payloads[i].local_b = nullptr;
      payloads[i].v = nullptr;
      payloads[i].q_size = 0;
    }
    best_corr_threads = new c_real[n_threads];
    jobs = new std::queue<FMJ*>[n_threads];
    jobs_pos = new std::queue<FMPJ*>[n_threads];
    jobs_rec = new std::queue<FindMaxJobRec*>[n_threads];
    jobs_pos_rec = new std::queue<FindMaxPosJobRec*>[n_threads];
    job_done = new bool[n_threads];
    local_bs = nullptr;
    running = true;
    stopped = false;
    #if CYCLIC_BUFFER
    qs = new uint64_t*[n_threads];
    q_vals = new int32_t*[n_threads];
    mus = new c_real*[n_threads];
    q_sizes = new int32_t[n_threads];
    for (int32_t i = 0; i < n_threads; ++i) {
      qs[i] = nullptr;
      q_vals[i] = nullptr;
      mus[i] = nullptr;
      q_sizes[i] = 0;
    }
    #endif
    vs = nullptr;
    max_queues = new int64_t[n_threads];
    total_processeds = new int64_t[n_threads];
  }
  

};

#endif

#endif

#endif
