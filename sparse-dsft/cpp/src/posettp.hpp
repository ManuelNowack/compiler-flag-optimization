#ifndef POSETTP_HPP
#define POSETTP_HPP

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

#if STRUCT_EXPERIMENT

#include "structset.hpp"

#endif

void load_poset(uint64_t* x, uint64_t*& order, int32_t samples, int32_t features);

void cmc_poset_part(uint64_t* x, c_real* y, uint64_t* b, uint64_t* marked, uint64_t* order,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed);

#define FMJPoset FindMaxJobPoset


// ThreadPool will be initialized with parameterized number of threads,
// but currently not all might be active during the execution.
// To get max_corr, min(n_threads, features) are used
class PosetTP {

private:

  struct FindMaxJobPoset {
    uint64_t* x;
    c_real* y;
    uint64_t* b;
    uint64_t* marked;
    uint64_t* order;
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
      cmc_poset_part(x, y, b, marked, order, samples, features, first_q_val, best_corr,
          best_corr_thread, local_b, delta, max_queue, total_processed);
    }
    FindMaxJobPoset(uint64_t* x, c_real* y, uint64_t* b, uint64_t* marked, uint64_t* order,
        int32_t samples, int32_t features, int32_t first_q_val,
        c_real& best_corr, c_real* best_corr_thread, uint64_t* local_b, double delta,
        int64_t* max_queue, int64_t* total_processed): x(x), y(y), b(b), marked(marked), order(order),
        best_corr(best_corr), best_corr_thread(best_corr_thread), samples(samples),
        features(features), first_q_val(first_q_val), local_b(local_b),
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
  uint64_t* marked;
  uint64_t* order;

  std::queue<FMJPoset*>* jobs;
  
  uint64_t** qs;
  int32_t** q_vals;
  c_real** mus;
  int32_t* q_sizes;
  bool stopped;
  
  int64_t* max_queues;
  int64_t* total_processeds;
  
  int32_t nt_max_corr;
  int32_t featbits;
  int32_t sambits;
  
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
    sambits = ((samples + 63) >> 6);
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
    
    if (!order) {
      //~ std::cout << "init marked\n";
      marked = new uint64_t[nt_max_corr * sambits];
      load_poset(x, order, samples, features);
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
    std::vector<FMJPoset*> local_jobs;
    
    //~ std::vector<FindMaxJobRec*> local_jobs_rec;
    //~ if (recursive) {
      //~ for (uint32_t i = 0; i < n; ++i) {
        //~ int32_t thread_no = i % nt_max_corr;
        //~ FindMaxJobRec* j = new FindMaxJobRec(x, y, b, samples, features, i, best_corr,
            //~ best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            //~ vs + thread_no * rowbits, max_queues + thread_no, total_processeds + thread_no);
        //~ local_jobs_rec.push_back(j);
        //~ jobs_rec[thread_no].push(j);
      //~ }
      //~ for (int32_t i = 0; i < nt_max_corr; ++i) {
        //~ results.push_back(std::async([=]{ return run_find_max_rec(i); }));
      //~ }
    //~ } else {
      for (uint32_t i = 0; i < n; ++i) {
        int32_t thread_no = i % nt_max_corr;
        //~ #if CYCLIC_BUFFER
        //~ FindMaxJobCyclic* j = new FindMaxJobCyclic(x, y, b, samples, features, i, best_corr,
            //~ best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            //~ qs + thread_no, q_vals + thread_no, mus + thread_no, q_sizes + thread_no,
            //~ max_queues + thread_no, total_processeds + thread_no);
        //~ #else
        FindMaxJobPoset* j = new FindMaxJobPoset(x, y, b, marked + thread_no * sambits,
            order, samples, features, i, best_corr,
            best_corr_threads + thread_no, local_bs + thread_no * featbits, delta,
            max_queues + thread_no, total_processeds + thread_no);
        //~ #endif
        local_jobs.push_back(j);
        jobs[thread_no].push(j);
      }
      for (int32_t i = 0; i < nt_max_corr; ++i) {
        results.push_back(std::async([=]{ return run_find_max(i); }));
      }
    //~ }
    
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
    
    for (FMJPoset* job : local_jobs) {
      delete job;
    }
    //~ for (FindMaxJobRec* job : local_jobs_rec) {
      //~ delete job;
    //~ }
    
    return best_corr;
  }
  
  c_real cmc_full(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
      int32_t samples, int32_t features, bool recursive) {
    int64_t dummy_max;
    int64_t dummy_total;
    return cmc_full(x, y, b, fb, samples, features, recursive, 0.0, dummy_max, dummy_total);
  }

  void run_find_max(int32_t id) {
    FMJPoset* function = nullptr;
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
  
  void stop() {
    if (!stopped) {
      delete[] best_corr_threads;
      delete[] jobs;
      //~ delete[] jobs_rec;
      //~ delete[] jobs_pos;
      //~ delete[] jobs_pos_rec;
      delete[] job_done;
      delete[] max_queues;
      delete[] total_processeds;
      if (local_bs) {
        delete[] local_bs;
        local_bs = nullptr;
      }
      if (marked) {
        delete[] marked;
      }
      if (order) {
        delete[] order;
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
  
  ~PosetTP() {
    stop();
  }

  PosetTP(int32_t n) : n_threads(n) {
    best_corr_threads = new c_real[n_threads];
    jobs = new std::queue<FMJPoset*>[n_threads];
    //~ jobs_pos = new std::queue<FMPJ*>[n_threads];
    //~ jobs_rec = new std::queue<FindMaxJobRec*>[n_threads];
    //~ jobs_pos_rec = new std::queue<FindMaxPosJobRec*>[n_threads];
    job_done = new bool[n_threads];
    local_bs = nullptr;
    marked = nullptr;
    order = nullptr;
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

#endif
