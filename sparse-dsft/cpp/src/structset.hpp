#ifndef STRUCTSET_HPP
#define STRUCTSET_HPP

#include <cstring>
#include <cmath>
#include "common.hpp"

#if STRUCT_EXPERIMENT
  
struct SetX {
  uint64_t* data;
  int32_t samples;
  int32_t features;
  int32_t featbits;
  SetX(uint64_t* data, int32_t samples, int32_t features)
      : data(data), samples(samples), features(features) {
    featbits = ((features + 63) >> 6);
  }
};

struct SetFreq {
  uint64_t* data;
  uint64_t data_l;
  uint64_t data_h;
  int32_t value;
  int32_t featbits;
  SetFreq() {}
  SetFreq(uint64_t* data, int32_t value, int32_t featbits)
      : value(value), featbits(featbits) {
    if (featbits == 1) {
      data_l = data[0];
    } else if (featbits == 2) {
      data_l = data[0];
      data_h = data[1];
    } else {
      this->data = data;
    }
  }
  void init(uint64_t* data, int32_t value, int32_t featbits) {
    this->value = value;
    this->featbits = featbits;
    if (featbits == 1) {
      data_l = data[0];
    } else if (featbits == 2) {
      data_l = data[0];
      data_h = data[1];
    } else {
      this->data = data;
    }
  }
  void copy(SetFreq other) {
    value = other.value;
    featbits = other.featbits;
    if (featbits == 1) {
      data_l = other.data_l;
    } else if (featbits == 2) {
      data_l = other.data_l;
      data_h = other.data_h;
    } else {
      data = other.data;
    }
  }
  
  void memcpy(SetFreq other) {
    value = other.value;
    featbits = other.featbits;
    if (featbits == 1) {
      data_l = other.data_l;
    } else if (featbits == 2) {
      data_l = other.data_l;
      data_h = other.data_h;
    } else {
      std::memcpy(data, other.data, featbits * sizeof(uint64_t));
    }
  }
};

struct SetRecursion {
    uint64_t* data;
    // todo: make a function which calls recursive function in a loop
};

struct SetQueue {
  uint64_t* q;
  int32_t* q_val;
  c_real* mus;
  int32_t q_size;
  int32_t features;
  int32_t featbits;
  int64_t push_idx;
  int64_t pop_idx;
  int64_t push_idx_row;
  int64_t pop_idx_row;
  
  SetQueue() {
    q = nullptr;
    q_val = nullptr;
    mus = nullptr;
    q_size = 0;
  }
  
  ~SetQueue() {
    if (q) {
      delete[] q;
    }
    if (q_val) {
      delete[] q_val;
    }
    if (mus) {
      delete[] mus;
    }
  }
  
  void init(int64_t buffer_size, int32_t row_size) {
    features = row_size;
    featbits = ((features + 63) >> 6);
    q = new uint64_t[buffer_size * featbits];
    q_val = new int32_t[buffer_size];
    mus = new c_real[buffer_size];
    q_size = buffer_size;
  }
  
  void reset() {
    push_idx = 0;
    pop_idx = 0;
    push_idx_row = 0;
    pop_idx_row = 0;
  }
  
  c_real get_curr_mu() {
    return mus[pop_idx];
  }
  
  SetFreq front() {
    SetFreq freq(q + pop_idx_row, q_val[pop_idx], featbits);
    return freq;
  }
  
  void pop() {
    if (++pop_idx == q_size) {
      pop_idx = 0;
      pop_idx_row = 0;
    } else {
      pop_idx_row += featbits;
    }
  }
  
  bool not_empty() {
    return push_idx != pop_idx;
  }
  
  void push_first(int32_t first_q_val) {
    if (featbits == 1) {
      q[push_idx_row] = ((uint64_t)1 << ((first_q_val & 63)));
      q_val[push_idx] = first_q_val;
      mus[push_idx] = std::numeric_limits<c_real>::max();
      ++push_idx;
      bool endb = (push_idx != q_size);
      push_idx = endb * push_idx;
      push_idx_row = endb * push_idx_row + endb;
    } else if (featbits == 2) {
      bool islb = (first_q_val < 64);
      uint64_t set1 = ((uint64_t)1 << ((first_q_val & 63)));
      q[push_idx_row + 0] = islb * set1;
      q[push_idx_row + 1] = !islb * set1;
      q_val[push_idx] = first_q_val;
      mus[push_idx] = std::numeric_limits<c_real>::max();
      ++push_idx;
      bool endb = (push_idx != q_size);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + 2);
    } else {
      int64_t ibits = (first_q_val >> 6);
      for (int64_t j = 0; j < ibits; ++j) {
        q[push_idx_row + j] = 0;
      }
      q[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
      for (int64_t j = ibits + 1; j < featbits; ++j) {
        q[push_idx_row + j] = 0;
      }
      q_val[push_idx] = first_q_val;
      mus[push_idx] = std::numeric_limits<c_real>::max();
      ++push_idx;
      bool endb = (push_idx != q_size);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + featbits);
    }
    if (push_idx == pop_idx) {
      resize();
    }
  }
  
  void push_next(SetFreq freq, c_real max_mu) {
    //~ std::cout << "push next " + std::to_string(freq.value) + " " + std::to_string(features) + " (" + std::to_string(featbits) + ")\n";
    int32_t value = freq.value;
    if (featbits == 1) {
      uint64_t data_l = freq.data_l;
      for (size_t i = value + 1; i < features; i++) {
        //~ std::cout << "push1\n";
        q[push_idx_row] = (data_l | ((uint64_t)1 << ((i & 63))));
        q_val[push_idx] = i;
        mus[push_idx] = max_mu;
        ++push_idx;
        bool endb = (push_idx != q_size);
        push_idx = endb * push_idx;
        push_idx_row = endb * push_idx_row + endb;
        if (push_idx == pop_idx) {
          resize();
        }
      }
    } else if (featbits == 2) {
      uint64_t data_l = freq.data_l;
      uint64_t data_h = freq.data_h;
      for (size_t i = value + 1; i < features; i++) {
        bool islb = (i < 64);
        uint64_t set1 = ((uint64_t)1 << ((i & 63)));
        q[push_idx_row + 0] = (data_l | (islb * set1));
        q[push_idx_row + 1] = (data_h | (!islb * set1));
        q_val[push_idx] = i;
        mus[push_idx] = max_mu;
        ++push_idx;
        bool endb = (push_idx != q_size);
        push_idx = endb * push_idx;
        push_idx_row = endb * (push_idx_row + 2);
        if (push_idx == pop_idx) {
          resize();
        }
      }
    } else {
      for (size_t i = value + 1; i < features; i++) {
        int64_t ibits = (i >> 6);
        for (int64_t j = 0; j < ibits; ++j) {
          q[push_idx_row + j] = freq.data[j];
        }
        q[push_idx_row + ibits] = (freq.data[ibits] | ((uint64_t)1 << ((i & 63))));
        for (int64_t j = ibits + 1; j < featbits; ++j) {
          q[push_idx_row + j] = freq.data[j];
        }
        q_val[push_idx] = i;
        mus[push_idx] = max_mu;
        ++push_idx;
        bool endb = (push_idx != q_size);
        push_idx = endb * push_idx;
        push_idx_row = endb * (push_idx_row + featbits);
        if (push_idx == pop_idx) {
          if (featbits > 2) {
            resize(freq);
          } else {
            resize();
          }
        }
      }
    }
  }
  
  int64_t get_cur_qsize() {
    return push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
  }
  
  void resize() {
    uint64_t* new_q = new uint64_t[q_size * featbits * 2];
    std::memcpy(new_q, q + pop_idx_row, (q_size - pop_idx) * featbits * sizeof(uint64_t));
    std::memcpy(new_q + q_size * featbits - pop_idx_row, q, pop_idx * featbits * sizeof(uint64_t));
    uint64_t* trash = q;
    q = new_q;
    delete[] trash;
    int32_t* new_q_val = new int32_t[q_size * 2];
    std::memcpy(new_q_val, q_val + pop_idx, (q_size - pop_idx) * sizeof(int32_t));
    std::memcpy(new_q_val + q_size - pop_idx, q_val, pop_idx * sizeof(int32_t));
    int32_t* trash_val = q_val;
    q_val = new_q_val;
    delete[] trash_val;
    c_real* new_mus = new c_real[q_size * 2];
    std::memcpy(new_mus, mus + pop_idx, (q_size - pop_idx) * sizeof(c_real));
    std::memcpy(new_mus + q_size - pop_idx, mus, pop_idx * sizeof(c_real));
    c_real* trash_mus = mus;
    mus = new_mus;
    delete[] trash_mus;
    push_idx = q_size;
    pop_idx = 0;
    push_idx_row = q_size * featbits;
    pop_idx_row = 0;
    q_size *= 2;
  }
  
  void resize(SetFreq& freq) {
    int64_t freq_idx = freq.data - q;
    uint64_t* new_q = new uint64_t[q_size * featbits * 2];
    std::memcpy(new_q, q + pop_idx_row, (q_size * featbits - pop_idx_row) * sizeof(uint64_t));
    std::memcpy(new_q + q_size * featbits - pop_idx_row, q, pop_idx_row * sizeof(uint64_t));
    uint64_t* trash = q;
    q = new_q;
    delete[] trash;
    int32_t* new_q_val = new int32_t[q_size * 2];
    std::memcpy(new_q_val, q_val + pop_idx, (q_size - pop_idx) * sizeof(int32_t));
    std::memcpy(new_q_val + q_size - pop_idx, q_val, pop_idx * sizeof(int32_t));
    int32_t* trash_val = q_val;
    q_val = new_q_val;
    delete[] trash_val;
    c_real* new_mus = new c_real[q_size * 2];
    std::memcpy(new_mus, mus + pop_idx, (q_size - pop_idx) * sizeof(c_real));
    std::memcpy(new_mus + q_size - pop_idx, mus, pop_idx * sizeof(c_real));
    c_real* trash_mus = mus;
    mus = new_mus;
    delete[] trash_mus;
    if (freq_idx >= pop_idx_row) {
      freq.data = (q + freq_idx) - pop_idx_row;
    } else {
      freq.data = (q + freq_idx) + q_size * featbits - pop_idx_row;
    }
    push_idx = q_size;
    pop_idx = 0;
    push_idx_row = q_size * featbits;
    pop_idx_row = 0;
    q_size *= 2;
  }
  
};

bool leq(SetFreq freq, SetX x, int32_t idx);
bool leq(SetX x, int32_t idx1, int32_t idx2);
void load_poset(SetX& x, uint64_t*& order, int32_t samples);

#endif

#endif
