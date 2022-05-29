#include "posettp.hpp"

#if STRUCT_EXPERIMENT

// This is the poset leq comparator: define as needed
// row1 <= row2
inline bool leq(uint64_t* row1, uint64_t* row2, int32_t featbits) {
  if (featbits == 1) {
    return (row2[0] & row1[0]) == row1[0];
  } else if (featbits == 2) {
    return (row2[0] & row1[0]) == row1[0] && (row2[1] & row1[1]) == row1[1];
  } else {
    for (int32_t j = 0; j < featbits; ++j) {
      if ((row2[j] & row1[j]) != row1[j]) {
        return false;
      }
    }
    return true;
  }
}

void load_poset(uint64_t* x, uint64_t*& order, int32_t samples, int32_t features) {
  int32_t featbits = ((features + 63) >> 6);
  int32_t sambits = ((samples + 63) >> 6);
  order = new uint64_t[samples * sambits];
  for (int32_t i = 0; i < samples; ++i) {
    for (int32_t j = 0; j < sambits; ++j) {
      int32_t oidx = i * sambits + j;
      uint64_t obit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) {
          break;
        }
        bool comp = leq(x + i * featbits, x + jjdx * featbits, featbits);
        obit |= ((uint64_t)comp << jj);
      }
      order[oidx] = obit;
    }
  }
}

void mark_recursive(uint64_t* marked, uint64_t* order, int32_t idx,
    int32_t ibit, int32_t samples, int32_t sambits) {
  marked[ibit] |= ((uint64_t)1 << (idx & 63));
  for (int32_t j = ibit; j < sambits; ++j) {
    for (int32_t jj = 0; jj < 64; ++jj) {
      int32_t jjdx = j * 64 + jj;
      if (jjdx <= idx) {
        continue;
      }
      if (jjdx >= samples) {
        break;
      }
      if (!(marked[j] & ((uint64_t)1 << jj)) && order[idx * sambits + j] & ((uint64_t)1 << jj)) {
        mark_recursive(marked, order, jjdx, j, samples, sambits);
      }
    }
  }
}

inline void mark_iterative(uint64_t* marked, uint64_t* order, int32_t idx,
    int32_t ibit, int32_t samples, int32_t sambits) {
  //~ marked[ibit] |= ((int64_t)1 << (idx & 63));
  for (int32_t j = ibit + 1; j < sambits; ++j) {
    uint64_t order_j = order[idx * sambits + j];
    if (order_j) {
      marked[j] |= order_j;
    }
  }
}

// marked defines whether freq is lt rows of x
// when a new freq is tested, it should be set to 0 and gradually filled
//~ template <T_X, T_FREQ>
inline bool cmp_and_fill(uint64_t* x, uint64_t* freq, uint64_t& cmarked,
    uint64_t* marked, uint64_t* order, int32_t idx, int32_t samples) {
  int32_t ibit = (idx >> 6);
  int32_t sambits = ((samples + 63) >> 6);
  if (cmarked & ((uint64_t)1 << (idx & 63))) {
    return true;
  }
  bool res = leq(freq, x, idx);
  if (res) {
    cmarked |= order[idx * sambits + ibit];
    mark_iterative(marked, order, idx, ibit, samples, sambits);
  }
  return res;
}

//~ template <T_X, T_FREQ>
void cmc_poset_part(uint64_t* x, c_real* y, uint64_t* b, uint64_t* marked, uint64_t* order,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed) {

  //~ int64_t samples = x.samples;
  //~ int64_t features = x.features;
	size_t n = features;
  //~ int32_t featbits = x.featbits;
  int32_t featbits = ((features + 63) >> 6);
  int32_t sambits = ((samples + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
		
	std::queue<std::vector<uint64_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  std::queue<c_real> mus;
  std::vector<uint64_t> first_freq(featbits, 0);
  first_freq[ibits] = ((uint64_t)1 << ((first_q_val & 63)));
	q.push(first_freq);
  q_val.push(first_q_val);
  mus.push(std::numeric_limits<c_real>::max());

	while (!q.empty()) {
		auto freq = q.front();
    int32_t value = q_val.front();
		q.pop();
    q_val.pop();
    c_real curr_mu = mus.front();
    mus.pop();
    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    if (curr_mu <= (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
      continue;
    }
		c_real mu1 = 0;
		c_real mu2 = 0;
    
    std::memset(marked, 0, sambits * sizeof(uint64_t));
		for (int64_t i = 0; i < sambits; ++i) {
      uint64_t cmarked = marked[i];
      for (int64_t ii = 0; ii < 64; ++ii) {
        int64_t iiidx = i * 64 + ii;
        if (iiidx >= samples) {
          break;
        }
        bool less_than = cmp_and_fill(x, freq.data(), cmarked, marked, order, iiidx, samples);
        if (less_than) {
          c_real y1 = y[iiidx];
          if (y1 > 0) {
            mu1 += y1;
          } else {
            mu2 -= y1;
          }
        }
      }
    }
		c_real corr = mu1 - mu2;

    bcg = best_corr_global;
    c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
		if (std::fabs(corr) > bc) {
			*best_corr_local = corr;
      bc = std::fabs(corr);
      std::memcpy(local_b, freq.data(), featbits * sizeof(uint64_t));
		}
    c_real max_mu = std::max(mu1, mu2);
		if (max_mu > bc + delta) {
			for (size_t i = value + 1; i < n; i++) {
        int64_t ibits = (i >> 6);
				std::vector<uint64_t> freq_new(freq);
        freq_new[ibits] = (freq_new[ibits] | ((uint64_t)1 << ((i & 63))));
				q.push(freq_new);
        mus.push(max_mu);
				q_val.push(i);
			}
      #if GET_STATS
      local_max_queue = std::max(local_max_queue, (int64_t)(q.size()));
      local_total_processed += (n - value - 1);
      #endif
		}
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

#else

// This is the poset leq comparator: define as needed
// row1 <= row2
inline bool leq(uint64_t* row1, uint64_t* row2, int32_t featbits) {
  if (featbits == 1) {
    return (row2[0] & row1[0]) == row1[0];
  } else if (featbits == 2) {
    return (row2[0] & row1[0]) == row1[0] && (row2[1] & row1[1]) == row1[1];
  } else {
    for (int32_t j = 0; j < featbits; ++j) {
      if ((row2[j] & row1[j]) != row1[j]) {
        return false;
      }
    }
    return true;
  }
}

void load_poset(uint64_t* x, uint64_t*& order, int32_t samples, int32_t features) {
  int32_t featbits = ((features + 63) >> 6);
  int32_t sambits = ((samples + 63) >> 6);
  order = new uint64_t[samples * sambits];
  for (int32_t i = 0; i < samples; ++i) {
    for (int32_t j = 0; j < sambits; ++j) {
      int32_t oidx = i * sambits + j;
      uint64_t obit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) {
          break;
        }
        bool comp = leq(x + i * featbits, x + jjdx * featbits, featbits);
        obit |= ((uint64_t)comp << jj);
      }
      order[oidx] = obit;
    }
  }
}

void mark_recursive(uint64_t* marked, uint64_t* order, int32_t idx,
    int32_t ibit, int32_t samples, int32_t sambits) {
  marked[ibit] |= ((uint64_t)1 << (idx & 63));
  for (int32_t j = ibit; j < sambits; ++j) {
    for (int32_t jj = 0; jj < 64; ++jj) {
      int32_t jjdx = j * 64 + jj;
      if (jjdx <= idx) {
        continue;
      }
      if (jjdx >= samples) {
        break;
      }
      if (!(marked[j] & ((uint64_t)1 << jj)) && order[idx * sambits + j] & ((uint64_t)1 << jj)) {
        mark_recursive(marked, order, jjdx, j, samples, sambits);
      }
    }
  }
}

inline void mark_iterative(uint64_t* marked, uint64_t* order, int32_t idx,
    int32_t ibit, int32_t samples, int32_t sambits) {
  //~ marked[ibit] |= ((int64_t)1 << (idx & 63));
  for (int32_t j = ibit + 1; j < sambits; ++j) {
    uint64_t order_j = order[idx * sambits + j];
    if (order_j) {
      marked[j] |= order_j;
    }
  }
}

// marked defines whether freq is lt rows of x
// when a new freq is tested, it should be set to 0 and gradually filled
inline bool cmp_and_fill(uint64_t* x, uint64_t* freq, uint64_t& cmarked,
    uint64_t* marked, uint64_t* order, int32_t idx, int32_t samples, int32_t features) {
  int32_t ibit = (idx >> 6);
  int32_t featbits = ((features + 63) >> 6);
  int32_t sambits = ((samples + 63) >> 6);
  if (cmarked & ((uint64_t)1 << (idx & 63))) {
    return true;
  }
  bool res = leq(freq, x + idx * featbits, featbits);
  if (res) {
    cmarked |= order[idx * sambits + ibit];
    mark_iterative(marked, order, idx, ibit, samples, sambits);
  }
  return res;
}

void cmc_poset_part(uint64_t* x, c_real* y, uint64_t* b, uint64_t* marked, uint64_t* order,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  int32_t sambits = ((samples + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
		
	std::queue<std::vector<uint64_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  std::queue<c_real> mus;
  std::vector<uint64_t> first_freq(featbits, 0);
  first_freq[ibits] = ((uint64_t)1 << ((first_q_val & 63)));
	q.push(first_freq);
  q_val.push(first_q_val);
  mus.push(std::numeric_limits<c_real>::max());

	while (!q.empty()) {
		auto freq = q.front();
    int32_t value = q_val.front();
		q.pop();
    q_val.pop();
    c_real curr_mu = mus.front();
    mus.pop();
    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    if (curr_mu <= (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
      continue;
    }
		c_real mu1 = 0;
		c_real mu2 = 0;
    
    std::memset(marked, 0, sambits * sizeof(uint64_t));
		for (int64_t i = 0; i < sambits; ++i) {
      uint64_t cmarked = marked[i];
      for (int64_t ii = 0; ii < 64; ++ii) {
        int64_t iiidx = i * 64 + ii;
        if (iiidx >= samples) {
          break;
        }
        bool less_than = cmp_and_fill(x, freq.data(), cmarked, marked, order, iiidx, samples, features);
        if (less_than) {
          c_real y1 = y[iiidx];
          if (y1 > 0) {
            mu1 += y1;
          } else {
            mu2 -= y1;
          }
        }
      }
    }
		c_real corr = mu1 - mu2;

    bcg = best_corr_global;
    c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
		if (std::fabs(corr) > bc) {
			*best_corr_local = corr;
      bc = std::fabs(corr);
      std::memcpy(local_b, freq.data(), featbits * sizeof(uint64_t));
		}
    c_real max_mu = std::max(mu1, mu2);
		if (max_mu > bc + delta) {
			for (size_t i = value + 1; i < n; i++) {
        uint64_t ibits = (i >> 6);
				std::vector<uint64_t> freq_new(freq);
        freq_new[ibits] = (freq_new[ibits] | ((uint64_t)1 << ((i & 63))));
				q.push(freq_new);
        mus.push(max_mu);
				q_val.push(i);
			}
      #if GET_STATS
      local_max_queue = std::max(local_max_queue, (int64_t)(q.size()));
      local_total_processed += (n - value - 1);
      #endif
		}
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

#endif
