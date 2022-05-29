#include "threadpool.hpp"

void cmc_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
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
    
		for (int64_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * featbits + j] & freq[j]) != freq[j]) {
          break;
        }
      }
      if (j == featbits) {
        c_real y1 = y[i];
        if (y1 > 0) {
          mu1 += y1;
        } else {
          mu2 -= y1;
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

void cmc_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	c_real best_corr = 0;
  
  uint64_t* qp = *q;
  int32_t* q_valp = *q_val;
  c_real* musp = *mus;
  int32_t q_sizep = *q_size;

  int64_t push_idx = 0;
  int64_t pop_idx = 0;
  int64_t push_idx_row = 0;
  int64_t pop_idx_row = 0;
  
  #if FEWER_LOOPS
  if (featbits == 1) {
    
    qp[push_idx_row] = ((uint64_t)1 << ((first_q_val & 63)));
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * push_idx_row + endb;
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * 2];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep - pop_idx_row, qp, pop_idx * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      (*q) = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      (*q_val) = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      (*mus) = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep;
      pop_idx_row = 0;
      q_sizep *= 2;
      (*q_size) = q_sizep;
    }
    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t freq = qp[pop_idx_row];
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        int32_t i = 0;
        #if AVX_EXPERIMENT
        __m256d mu1v = _mm256_setzero_pd();
        __m256d mu2v = _mm256_setzero_pd();
        __m256d zeros = _mm256_setzero_pd();
        __m256i vv = _mm256_set1_epi64x(freq);
        int32_t samples4 = (samples / 4) * 4;
        for (; i < samples4; i += 4) {
          __m256i xv = _mm256_loadu_si256((__m256i*)(x + i));
          __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
          if (!_mm256_testz_si256(av, av)) {
            __m256d yv = _mm256_blendv_pd(zeros, _mm256_loadu_pd(y + i), _mm256_castsi256_pd(av));
            __m256d gv = _mm256_cmp_pd(zeros, yv, _CMP_LT_OQ);
            mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gv));
            mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(yv, zeros, gv));
          }
        }
        mu1 = _mm256_reduce_add_pd(mu1v);
        mu2 = _mm256_reduce_add_pd(mu2v);
        #endif
        for (; i < samples; ++i) {
          if ((x[i] & freq) == freq) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }
        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          *best_corr_local = corr;
          bc = std::fabs(corr);
          local_b[0] = freq;
        }
        
        c_real max_mu = std::max(mu1, mu2);
        //~ std::cout << "MAX_MU: " + std::to_string(max_mu) + "\n";
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            qp[push_idx_row] = (freq | ((uint64_t)1 << ((i & 63))));
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * push_idx_row + endb;
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_sizep * 2];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              (*q) = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              (*q_val) = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              (*mus) = new_mus;
              delete[] trash_mus;
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep;
              pop_idx_row = 0;
              q_sizep *= 2;
              (*q_size) = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_sizep);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * pop_idx_row + endb2;
      //~ std::cout << std::to_string(push_idx) + " " + std::to_string(pop_idx) + "\n";
    }
    
  } else if (featbits == 2) {

    bool islb = (first_q_val < 64);
    uint64_t set1 = ((uint64_t)1 << ((first_q_val & 63)));
    qp[push_idx_row + 0] = islb * set1;
    qp[push_idx_row + 1] = !islb * set1;
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + 2);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * 4];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 2 * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * 2 - pop_idx_row, qp, pop_idx * 2 * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      (*q) = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      (*q_val) = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      (*mus) = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * 2;
      pop_idx_row = 0;
      q_sizep *= 2;
      (*q_size) = q_sizep;
    }
    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t freql = qp[pop_idx_row + 0];
        uint64_t freqh = qp[pop_idx_row + 1];
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        int32_t i = 0;
        #if AVX_EXPERIMENT
        __m256d mu1v = _mm256_setzero_pd();
        __m256d mu2v = _mm256_setzero_pd();
        __m256d zeros = _mm256_setzero_pd();
         __m256i vv = _mm256_set_epi64x(freqh, freql, freqh, freql);
        int32_t samples2 = (samples / 2) * 2;
        for (; i < samples2; i += 2) {
          __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * 2));
          __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
          __m256d bv = _mm256_and_pd(_mm256_permute_pd(_mm256_castsi256_pd(av), 0x5), _mm256_castsi256_pd(av));
          if (!_mm256_testz_pd(bv, bv)) {
            double y0 = y[i];
            double y1 = y[i + 1];
            __m256d yv = _mm256_set_pd(0, y1, 0, y0);
            __m256d gvlt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_LT_OQ), bv);
            __m256d gvgt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_GT_OQ), bv);
            mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gvlt));
            mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(zeros, yv, gvgt));
          }
        }
        mu1 += _mm256_reduce_add_pd(mu1v);
        mu2 += _mm256_reduce_add_pd(mu2v); 
        #endif
        for (; i < samples; ++i) {
          if (((x[i * 2] & freql) == freql) && ((x[i * 2 + 1] & freqh) == freqh)) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }
        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          *best_corr_local = corr;
          bc = std::fabs(corr);
          local_b[0] = freql;
          local_b[1] = freqh;
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            bool islb = (i < 64);
            uint64_t set1 = ((uint64_t)1 << ((i & 63)));
            qp[push_idx_row + 0] = (freql | (islb * set1));
            qp[push_idx_row + 1] = (freqh | (!islb * set1));
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + 2);
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_sizep * 4];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 2 - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * 2 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              (*q) = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              (*q_val) = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              (*mus) = new_mus;
              delete[] trash_mus;
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * 2;
              pop_idx_row = 0;
              q_sizep *= 2;
              (*q_size) = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_sizep);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * (pop_idx_row + 2);  
    }
  } else {
  #endif
  
    int64_t ibits = (first_q_val >> 6);
    for (int64_t j = 0; j < ibits; ++j) {
      qp[push_idx_row + j] = 0;
    }
    qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
    for (int64_t j = ibits + 1; j < featbits; ++j) {
      qp[push_idx_row + j] = 0;
    }
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + featbits);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * featbits * 2];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * featbits * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * featbits - pop_idx_row, qp, pop_idx * featbits * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      (*q) = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      (*q_val) = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      (*mus) = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * featbits;
      pop_idx_row = 0;
      q_sizep *= 2;
      (*q_size) = q_sizep;
    }

    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t* freq = qp + pop_idx_row;
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        
        for (int64_t i = 0; i < samples; ++i) {
          int32_t j = 0;
          #if AVX_EXPERIMENT
          int32_t featbits4 = (featbits / 4) * 4;
          for (; j < featbits4; j += 4) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * featbits + j));
            __m256i vv = _mm256_loadu_si256((__m256i*)(freq + j));
            __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
            if (!_mm256_testz_si256(av, av)) {
              break;
            }
          }
          #endif
          for (; j < featbits; ++j) {
            if ((x[i * featbits + j] & freq[j]) != freq[j]) {
              break;
            }
          }
          if (j == featbits) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }

        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          *best_corr_local = corr;
          bc = std::fabs(corr);
          std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            int64_t ibits = (i >> 6);
            for (int64_t j = 0; j < ibits; ++j) {
              qp[push_idx_row + j] = freq[j];
            }
            qp[push_idx_row + ibits] = (freq[ibits] | ((uint64_t)1 << ((i & 63))));
            for (int64_t j = ibits + 1; j < featbits; ++j) {
              qp[push_idx_row + j] = freq[j];
            }
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + featbits);

            if (push_idx == pop_idx) {
              int64_t freq_idx = freq - qp;
              uint64_t* new_q = new uint64_t[q_sizep * featbits * 2];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * featbits - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * featbits - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              (*q) = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              (*q_val) = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              (*mus) = new_mus;
              delete[] trash_mus;
              if (freq_idx >= pop_idx_row) {
                freq = (qp + freq_idx) - pop_idx_row;
              } else {
                freq = (qp + freq_idx) + q_sizep * featbits - pop_idx_row;
              }
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * featbits;
              pop_idx_row = 0;
              q_sizep *= 2;
              (*q_size) = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      if (++pop_idx == q_sizep) {
        pop_idx = 0;
        pop_idx_row = 0;
      } else {
        pop_idx_row += featbits;
      }
    }

  #if FEWER_LOOPS
  }
  #endif
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmc_part_rec(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed) {
      

  int64_t local_max_queue = 1;
  int64_t sub_max_queue = 1;
  int64_t local_total_processed = 1;

      
  size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  for (int32_t i = 0; i < featbits; ++i) {
    v[i] = 0;
  }
  int64_t shifter = (ibits == featbits - 1) ? ((features - 1) & 63) : 63;
  v[ibits] = (v[ibits] | ((uint64_t)1 << (shifter - (first_q_val & 63))));
  c_real mu = std::numeric_limits<c_real>::max();
  cmc_part_rec_subcall(x, y, b, samples, features, best_corr_global,
      best_corr_local, local_b, delta, v, first_q_val, mu,
      local_max_queue, sub_max_queue + 1, local_total_processed);
      
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmc_part_rec_subcall(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed) {
      
  c_real mu1 = 0.0;
  c_real mu2 = 0.0;
  
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  max_queue = std::max(max_queue, sub_max_queue);
  ++total_processed;
  #endif
  
  c_real bcl = *best_corr_local;
  c_real bcg = best_corr_global;
  if (curr_mu <= (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
    return;
  }
  
  #if FEWER_LOOPS
  
  if (featbits == 1) {
    uint64_t freq = v[0];
    int32_t i = 0;
    #if AVX_EXPERIMENT
    __m256d mu1v = _mm256_setzero_pd();
    __m256d mu2v = _mm256_setzero_pd();
    __m256d zeros = _mm256_setzero_pd();
    __m256i vv = _mm256_set1_epi64x(freq);
    int32_t samples4 = (samples / 4) * 4;
    for (; i < samples4; i += 4) {
      __m256i xv = _mm256_loadu_si256((__m256i*)(x + i));
      __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
      if (!_mm256_testz_si256(av, av)) {
        __m256d yv = _mm256_blendv_pd(zeros, _mm256_loadu_pd(y + i), _mm256_castsi256_pd(av));
        __m256d gv = _mm256_cmp_pd(zeros, yv, _CMP_LT_OQ);
        mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gv));
        mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(yv, zeros, gv));
      }
    }
    mu1 = _mm256_reduce_add_pd(mu1v);
    mu2 = _mm256_reduce_add_pd(mu2v);
    #endif
    for (; i < samples; ++i) {
      if ((x[i] & freq) == freq) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else if (featbits == 2) {
    uint64_t freql = v[0];
    uint64_t freqh = v[1];
    int32_t i = 0;
    #if AVX_EXPERIMENT
    __m256d mu1v = _mm256_setzero_pd();
    __m256d mu2v = _mm256_setzero_pd();
    __m256d zeros = _mm256_setzero_pd();
     __m256i vv = _mm256_set_epi64x(freqh, freql, freqh, freql);
    int32_t samples2 = (samples / 2) * 2;
    for (; i < samples2; i += 2) {
      __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * 2));
      __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
      __m256d bv = _mm256_and_pd(_mm256_permute_pd(_mm256_castsi256_pd(av), 0x5), _mm256_castsi256_pd(av));
      if (!_mm256_testz_pd(bv, bv)) {
        double y0 = y[i];
        double y1 = y[i + 1];
        __m256d yv = _mm256_set_pd(0, y1, 0, y0);
        __m256d gvlt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_LT_OQ), bv);
        __m256d gvgt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_GT_OQ), bv);
        mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gvlt));
        mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(zeros, yv, gvgt));
      }
    }
    mu1 += _mm256_reduce_add_pd(mu1v);
    mu2 += _mm256_reduce_add_pd(mu2v); 
    #endif
    for (; i < samples; ++i) {
      if (((x[i * 2] & freql) == freql) && ((x[i * 2 + 1] & freqh) == freqh)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else {

  #endif

    for (int64_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      #if AVX_EXPERIMENT
      int32_t featbits4 = (featbits / 4) * 4;
      for (; j < featbits4; j += 4) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * featbits + j));
        __m256i vv = _mm256_loadu_si256((__m256i*)(v + j));
        __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
        if (!_mm256_testz_si256(av, av)) {
          break;
        }
      }
      #endif
      for (; j < featbits; ++j) {
        uint64_t vj = v[j];
        if ((x[i * featbits + j] & vj) != vj) {
          break;
        }
      }

     if (j == featbits) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  
  #if FEWER_LOOPS
  }
  #endif
  c_real corr = mu1 - mu2;
  bcg = best_corr_global;
  c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
  if (std::fabs(corr) > bc) {
    *best_corr_local = corr;
    bc = std::fabs(corr);
    std::memcpy(local_b, v, featbits * sizeof(uint64_t));
  }
  c_real max_mu = std::max(mu1, mu2);
  if (max_mu > bc + delta) {
    for (int32_t nt = top1 + 1; nt < features; ++nt) {
      int32_t ntt = (nt >> 6);
      int64_t shifter = (ntt == featbits - 1) ? ((features - 1) & 63) : 63;
      v[ntt] = (v[ntt] | ((uint64_t)1 << (shifter-(nt & 63))));
      cmc_part_rec_subcall(x, y, b, samples, features,
          best_corr_global, best_corr_local, local_b, delta, v, nt, max_mu,
          max_queue, sub_max_queue + 1, total_processed);
      v[ntt] = (v[ntt] & ~((uint64_t)1 << (shifter-(nt & 63))));
    }
  }
}

#if BIT_AVX

void cmc_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t vecbits = ((features + 255) >> 8);
  int32_t padded = (((features + 255) >> 8) << 2);
  int32_t featbits = ((features + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
		
	std::queue<std::vector<uint64_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  std::queue<c_real> mus;
  std::vector<uint64_t> first_freq(padded, 0);
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
    
		for (int64_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      for (; j < vecbits; ++j) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded + (j << 2)));
        __m256i vv = _mm256_loadu_si256((__m256i*)(freq.data() + (j << 2)));
        __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
        if (!_mm256_testz_si256(av, av)) {
          break;
        }
      }

     if (j == vecbits) {
        c_real y1 = y[i];
        if (y1 > 0) {
          mu1 += y1;
        } else {
          mu2 -= y1;
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

void cmc_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t vecbits = ((features + 255) >> 8);
  int32_t padded = (((features + 255) >> 8) << 2);
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	c_real best_corr = 0;
  
  uint64_t* qp = *q;
  int32_t* q_valp = *q_val;
  int32_t q_sizep = *q_size;

  int64_t push_idx = 0;
  int64_t pop_idx = 0;
  int64_t push_idx_row = 0;
  int64_t pop_idx_row = 0;
  
  #if FEWER_LOOPS
  if (vecbits == 1) {
    
    int64_t ibits = (first_q_val >> 6);
    std::memset(qp + push_idx_row, 0, 4 * sizeof(uint64_t));
    qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + 4);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * 4 * 2];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 4 * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * 4 - pop_idx_row, qp, pop_idx * 4 * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      (*q) = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      (*q_val) = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      (*mus) = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * 4;
      pop_idx_row = 0;
      q_sizep *= 2;
      (*q_size) = q_sizep;
    }
    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t* freq = qp + pop_idx_row;
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        
        __m256i vv  = _mm256_loadu_si256((__m256i*)(freq));
        for (int64_t i = 0; i < samples; ++i) {
          __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded));
          __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
          if (_mm256_testz_si256(av, av)) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }
        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          *best_corr_local = corr;
          bc = std::fabs(corr);
          std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            int64_t ibits = (i >> 6);
            std::memcpy(qp + push_idx_row, freq, 4 * sizeof(uint64_t));
            qp[push_idx_row + ibits] = (qp[push_idx_row + ibits] | ((uint64_t)1 << ((i & 63))));
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + 4);
            if (push_idx == pop_idx) {
              int64_t freq_idx = freq - qp;
              uint64_t* new_q = new uint64_t[q_sizep * 4 * 2];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 4 - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * 4 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              (*q) = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              (*q_val) = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              (*mus) = new_mus;
              delete[] trash_mus;
              if (freq_idx >= pop_idx_row) {
                freq = (qp + freq_idx) - pop_idx_row;
              } else {
                freq = (qp + freq_idx) + q_sizep * 4 - pop_idx_row;
              }
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * 4;
              pop_idx_row = 0;
              q_sizep *= 2;
              (*q_size) = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_sizep);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * (pop_idx_row + padded);
    }
    
  } else if (vecbits == 2) {

    int64_t ibits = (first_q_val >> 6);
    uint64_t set1 = ((uint64_t)1 << ((first_q_val & 63)));
    std::memset(qp + push_idx_row, 0, 8 * sizeof(uint64_t));
    qp[push_idx_row + ibits] = set1;
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + 8);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * 4];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 8 * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * 8 - pop_idx_row, qp, pop_idx * 8 * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      (*q) = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 8];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      (*q_val) = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      (*mus) = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * 8;
      pop_idx_row = 0;
      q_sizep *= 2;
      (*q_size) = q_sizep;
    }
    while (push_idx != pop_idx) {
      int32_t value = q_valp[pop_idx];
      c_real curr_mu = musp[pop_idx];
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t* freql = qp + pop_idx_row;
        uint64_t* freqh = qp + pop_idx_row + 4;
        c_real mu1 = 0;
        c_real mu2 = 0;
        __m256i vvl = _mm256_loadu_si256((__m256i*)(freql));
        __m256i vvh = _mm256_loadu_si256((__m256i*)(freqh));
        for (int64_t i = 0; i < samples; ++i) {
          __m256i xvl = _mm256_loadu_si256((__m256i*)(x + i * padded));
          __m256i avl = _mm256_xor_si256(_mm256_and_si256(xvl, vvl), vvl);
          __m256i xvh = _mm256_loadu_si256((__m256i*)(x + i * padded + 4));
          __m256i avh = _mm256_xor_si256(_mm256_and_si256(xvh, vvh), vvh);
          if (_mm256_testz_si256(avl, avl) && _mm256_testz_si256(avh, avh)) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }
        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          *best_corr_local = corr;
          bc = std::fabs(corr);
          std::memcpy(local_b, freql, featbits * sizeof(uint64_t));
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            int64_t freq_idx = freql - qp;
            int64_t ibits = (i >> 6);
            uint64_t set1 = ((uint64_t)1 << ((i & 63)));
            std::memcpy(qp + push_idx_row, freql, 8 * sizeof(uint64_t));
            qp[push_idx_row + ibits] = set1;
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + 8);
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_sizep * 4];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 8 - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * 8 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              (*q) = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 8];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              (*q_val) = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              (*mus) = new_mus;
              delete[] trash_mus;
              if (freq_idx >= pop_idx_row) {
                freql = (qp + freq_idx) - pop_idx_row;
              } else {
                freql = (qp + freq_idx) + q_sizep * 8 - pop_idx_row;
              }
              freqh = freql + 4;
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * 8;
              pop_idx_row = 0;
              q_sizep *= 2;
              (*q_size) = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_sizep);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * (pop_idx_row + padded);
    }    
  } else {
  #endif
  
    int64_t ibits = (first_q_val >> 6);
    for (int64_t j = 0; j < ibits; ++j) {
      qp[push_idx_row + j] = 0;
    }
    qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
    for (int64_t j = ibits + 1; j < padded; ++j) {
      qp[push_idx_row + j] = 0;
    }
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + padded);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * padded * 2];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * padded * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * padded - pop_idx_row, qp, pop_idx * padded * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      (*q) = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      (*q_val) = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      (*mus) = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * padded;
      pop_idx_row = 0;
      q_sizep *= 2;
      (*q_size) = q_sizep;
    }

    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t* freq = qp + pop_idx_row;
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        
        for (int64_t i = 0; i < samples; ++i) {
          int32_t j = 0;
          for (; j < vecbits; ++j) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded + (j << 2)));
            __m256i vv = _mm256_loadu_si256((__m256i*)(freq + (j << 2)));
            __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
            if (!_mm256_testz_si256(av, av)) {
              break;
            }
          }
         if (j == vecbits) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }

        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          *best_corr_local = corr;
          bc = std::fabs(corr);
          std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            int64_t ibits = (i >> 6);
            for (int64_t j = 0; j < ibits; ++j) {
              qp[push_idx_row + j] = freq[j];
            }
            qp[push_idx_row + ibits] = (freq[ibits] | ((uint64_t)1 << ((i & 63))));
            for (int64_t j = ibits + 1; j < featbits; ++j) {
              qp[push_idx_row + j] = freq[j];
            }
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + padded);

            if (push_idx == pop_idx) {
              int64_t freq_idx = freq - qp;
              uint64_t* new_q = new uint64_t[q_sizep * padded * 2];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * padded - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * padded - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              (*q) = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              (*q_val) = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              (*mus) = new_mus;
              delete[] trash_mus;
              if (freq_idx >= pop_idx_row) {
                freq = (qp + freq_idx) - pop_idx_row;
              } else {
                freq = (qp + freq_idx) + q_sizep * padded - pop_idx_row;
              }
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * padded;
              pop_idx_row = 0;
              q_sizep *= 2;
              (*q_size) = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      if (++pop_idx == q_sizep) {
        pop_idx = 0;
        pop_idx_row = 0;
      } else {
        pop_idx_row += padded;
      }
    }

  #if FEWER_LOOPS
  }
  #endif
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmc_part_rec_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed) {

  int64_t local_max_queue = 1;
  int64_t sub_max_queue = 1;
  int64_t local_total_processed = 1;

      
  size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  int32_t padded = (((features + 255) >> 8) << 2);
  int64_t ibits = (first_q_val >> 6);
  
  for (int32_t i = 0; i < padded; ++i) {
    v[i] = 0;
  }
  v[ibits] = (v[ibits] | ((uint64_t)1 << ((first_q_val & 63))));
  c_real mu = std::numeric_limits<c_real>::max();
  cmc_part_rec_subcall_avx(x, y, b, samples, features, best_corr_global,
      best_corr_local, local_b, delta, v, first_q_val, mu,
      local_max_queue, sub_max_queue + 1, local_total_processed);
      
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmc_part_rec_subcall_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int32_t top1, c_real max_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed) {
      
  c_real mu1 = 0.0;
  c_real mu2 = 0.0;
  
  int32_t vecbits = ((features + 255) >> 8);
  int32_t padded = (((features + 255) >> 8) << 2);
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  max_queue = std::max(max_queue, sub_max_queue);
  ++total_processed;
  #endif
  
  c_real bcl = *best_corr_local;
  c_real bcg = best_corr_global;
  if (curr_mu <= (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
    return;
  }
  
  #if FEWER_LOOPS
  
  if (vecbits == 1) {
    
    __m256i vv  = _mm256_loadu_si256((__m256i*)(v));
    for (int64_t i = 0; i < samples; ++i) {
      __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded));
      __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
      if (_mm256_testz_si256(av, av)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else if (vecbits == 2) {
    __m256i vvl = _mm256_loadu_si256((__m256i*)(v));
    __m256i vvh = _mm256_loadu_si256((__m256i*)(v + 4));
    for (int64_t i = 0; i < samples; ++i) {
      __m256i xvl = _mm256_loadu_si256((__m256i*)(x + i * padded));
      __m256i avl = _mm256_xor_si256(_mm256_and_si256(xvl, vvl), vvl);
      __m256i xvh = _mm256_loadu_si256((__m256i*)(x + i * padded + 4));
      __m256i avh = _mm256_xor_si256(_mm256_and_si256(xvh, vvh), vvh);
      if (_mm256_testz_si256(avl, avl) && _mm256_testz_si256(avh, avh)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else {

  #endif

    for (int64_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      for (; j < vecbits; ++j) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded + (j << 2)));
        __m256i vv = _mm256_loadu_si256((__m256i*)(v + (j << 2)));
        __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
        if (!_mm256_testz_si256(av, av)) {
          break;
        }
      }

     if (j == vecbits) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  
  #if FEWER_LOOPS
  }
  #endif
  c_real corr = mu1 - mu2;
  bcg = best_corr_global;
  c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
  if (std::fabs(corr) > bc) {
    *best_corr_local = corr;
    bc = std::fabs(corr);
    std::memcpy(local_b, v, featbits * sizeof(uint64_t));
  }
  c_real max_mu = std::max(mu1, mu2);
  if (max_mu > bc + delta) {
    for (int32_t nt = top1 + 1; nt < features; ++nt) {
      int32_t ntt = (nt >> 6);
      v[ntt] = (v[ntt] | ((uint64_t)1 << ((nt & 63))));
      cmc_part_rec_subcall_avx(x, y, b, samples, features,
          best_corr_global, best_corr_local, local_b, delta, v, nt, max_mu,
          max_queue, sub_max_queue + 1, total_processed);
      v[ntt] = (v[ntt] & ~((uint64_t)1 << ((nt & 63))));
    }
  }
}

#endif

void cmcpos_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	std::queue<std::vector<uint64_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  std::queue<c_real> mus;
  std::vector<uint64_t> first_freq(featbits, 0);
  
  bool go_on = true;
  
  if (first_q_val == 0) { // let the first thread handle the all-zeros case
    c_real mu1 = 0;
		c_real mu2 = 0;
    
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
		c_real corr = mu1 - mu2;

    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    c_real bc = bcg > bcl ? bcg : bcl;
		if (corr > bc) {
			*best_corr_local = corr;
      bc = corr;
      std::memset(local_b, 0, featbits * sizeof(uint64_t));
		}
		go_on = (mu1 > bc + delta);
    mus.push(mu1);
  } else {
    c_real mu1 = 0;
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      mu1 += y1 * (y1 > 0);
    }
    mus.push(mu1);
  }

  if (go_on) {
    first_freq[ibits] = ((uint64_t)1 << ((first_q_val & 63)));
    q.push(first_freq);
    q_val.push(first_q_val);

    while (!q.empty()) {
      auto freq = q.front();
      int32_t value = q_val.front();
      q.pop();
      q_val.pop();
      c_real curr_mu = mus.front();
      mus.pop();
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu <= (bcg > bcl ? bcg : bcl) + delta) {
        continue;
      }
      c_real mu1 = 0;
      c_real mu2 = 0;
      
      for (int64_t i = 0; i < samples; ++i) {
        int32_t j = 0;
        for (; j < featbits; ++j) {
          if ((x[i * featbits + j] & freq[j]) != freq[j]) {
            break;
          }
        }
        if (j == featbits) {
          c_real y1 = y[i];
          if (y1 > 0) {
            mu1 += y1;
          } else {
            mu2 -= y1;
          }
        }
      }
      c_real corr = mu1 - mu2;

      bcg = best_corr_global;
      c_real bc = bcg > bcl ? bcg : bcl;
      if (corr > bc) {
        *best_corr_local = corr;
        bc = corr;
        std::memcpy(local_b, freq.data(), featbits * sizeof(uint64_t));
      }
      if (mu1 > bc + delta) {
        for (size_t i = value + 1; i < n; i++) {
          int64_t ibits = (i >> 6);
          std::vector<uint64_t> freq_new(freq);
          freq_new[ibits] = (freq_new[ibits] | ((uint64_t)1 << ((i & 63))));
          q.push(freq_new);
          q_val.push(i);
          mus.push(mu1);
        }
        #if GET_STATS
        local_max_queue = std::max(local_max_queue, (int64_t)(q.size()));
        local_total_processed += (n - value - 1);
        #endif
      }
    }
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmcpos_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	c_real best_corr = 0;
  
  uint64_t* qp = *q;
  int32_t* q_valp = *q_val;
  c_real* musp = *mus;
  int32_t q_sizep = *q_size;

  int64_t push_idx = 0;
  int64_t pop_idx = 0;
  int64_t push_idx_row = 0;
  int64_t pop_idx_row = 0;
  
  bool go_on = true;
  
  if (first_q_val == 0) { // let the first thread handle the all-zeros case
    c_real mu1 = 0;
		c_real mu2 = 0;
    
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
		c_real corr = mu1 - mu2;

    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    c_real bc = bcg > bcl ? bcg : bcl;
		if (corr > bc) {
			*best_corr_local = corr;
      bc = corr;
      std::memset(local_b, 0, featbits * sizeof(uint64_t));
		}
		go_on = (mu1 > bc + delta);
    musp[push_idx] = mu1;
  } else {
    c_real mu1 = 0;
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      mu1 += y1 * (y1 > 0);
    }
    musp[push_idx] = mu1;
  }

  if (go_on) {
  
    #if FEWER_LOOPS
    if (featbits == 1) {
      
      qp[push_idx_row] = ((uint64_t)1 << ((first_q_val & 63)));
      q_valp[push_idx] = first_q_val;
      ++push_idx;
      bool endb = (push_idx != q_sizep);
      push_idx = endb * push_idx;
      push_idx_row = endb * push_idx_row + endb;
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_sizep * 2];
        std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * sizeof(uint64_t));
        std::memcpy(new_q + q_sizep - pop_idx_row, qp, pop_idx * sizeof(uint64_t));
        uint64_t* trash = qp;
        qp = new_q;
        (*q) = new_q;
        delete[] trash;
        int32_t* new_q_val = new int32_t[q_sizep * 2];
        std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
        std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
        int32_t* trash_val = q_valp;
        q_valp = new_q_val;
        (*q_val) = new_q_val;
        delete[] trash_val;
        c_real* new_mus = new c_real[q_sizep * 2];
        std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
        std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
        c_real* trash_mus = musp;
        musp = new_mus;
        (*mus) = new_mus;
        delete[] trash_mus;
        push_idx = q_sizep;
        pop_idx = 0;
        push_idx_row = q_sizep;
        pop_idx_row = 0;
        q_sizep *= 2;
        (*q_size) = q_sizep;
      }
      while (push_idx != pop_idx) {
        c_real curr_mu = musp[pop_idx];
        c_real bcl = *best_corr_local;
        c_real bcg = best_corr_global;
        if (curr_mu > (bcg > bcl ? bcg : bcl) + delta) {
          uint64_t freq = qp[pop_idx_row];
          int32_t value = q_valp[pop_idx];
          c_real mu1 = 0;
          c_real mu2 = 0;
          int32_t i = 0;
          #if AVX_EXPERIMENT
          __m256d mu1v = _mm256_setzero_pd();
          __m256d mu2v = _mm256_setzero_pd();
          __m256d zeros = _mm256_setzero_pd();
          __m256i vv = _mm256_set1_epi64x(freq);
          int32_t samples4 = (samples / 4) * 4;
          for (; i < samples4; i += 4) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + i));
            __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
            if (!_mm256_testz_si256(av, av)) {
              __m256d yv = _mm256_blendv_pd(zeros, _mm256_loadu_pd(y + i), _mm256_castsi256_pd(av));
              __m256d gv = _mm256_cmp_pd(zeros, yv, _CMP_LT_OQ);
              mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gv));
              mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(yv, zeros, gv));
            }
          }
          mu1 = _mm256_reduce_add_pd(mu1v);
          mu2 = _mm256_reduce_add_pd(mu2v);
          #endif
          for (; i < samples; ++i) {
            if ((x[i] & freq) == freq) {
              c_real y1 = y[i];
              bool ycz = (y1 > 0);
              mu1 += ycz * y1;
              mu2 -= !ycz * y1;
            }
          }
          c_real corr = mu1 - mu2;    
          bcg = best_corr_global;
          c_real bc = bcg > bcl ? bcg : bcl;
          if (corr > bc) {
            *best_corr_local = corr;
            bc = corr;
            local_b[0] = freq;
          }
          if (mu1 > bc + delta) {
            for (size_t i = value + 1; i < n; i++) {
              qp[push_idx_row] = (freq | ((uint64_t)1 << ((i & 63))));
              q_valp[push_idx] = i;
              musp[push_idx] = mu1;
              ++push_idx;
              bool endb = (push_idx != q_sizep);
              push_idx = endb * push_idx;
              push_idx_row = endb * push_idx_row + endb;
              if (push_idx == pop_idx) {
                uint64_t* new_q = new uint64_t[q_sizep * 2];
                std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx_row) * sizeof(uint64_t));
                std::memcpy(new_q + q_sizep - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
                uint64_t* trash = qp;
                qp = new_q;
                (*q) = new_q;
                delete[] trash;
                int32_t* new_q_val = new int32_t[q_sizep * 2];
                std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
                std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
                int32_t* trash_val = q_valp;
                q_valp = new_q_val;
                (*q_val) = new_q_val;
                delete[] trash_val;
                c_real* new_mus = new c_real[q_sizep * 2];
                std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
                std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
                c_real* trash_mus = musp;
                musp = new_mus;
                (*mus) = new_mus;
                delete[] trash_mus;
                push_idx = q_sizep;
                pop_idx = 0;
                push_idx_row = q_sizep;
                pop_idx_row = 0;
                q_sizep *= 2;
                (*q_size) = q_sizep;
              }
            }
            #if GET_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
            local_max_queue = std::max(local_max_queue, cur_qsize);
            local_total_processed += (n - value - 1);
            #endif
          }
        }
        ++pop_idx;
        bool endb2 = (pop_idx != q_sizep);
        pop_idx = endb2 * pop_idx;
        pop_idx_row = endb2 * pop_idx_row + endb2;
      }
      
    } else if (featbits == 2) {

      bool islb = (first_q_val < 64);
      uint64_t set1 = ((uint64_t)1 << ((first_q_val & 63)));
      qp[push_idx_row + 0] = islb * set1;
      qp[push_idx_row + 1] = !islb * set1;
      q_valp[push_idx] = first_q_val;
      ++push_idx;
      bool endb = (push_idx != q_sizep);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + 2);
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_sizep * 4];
        std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 2 * sizeof(uint64_t));
        std::memcpy(new_q + q_sizep * 2 - pop_idx_row, qp, pop_idx * 2 * sizeof(uint64_t));
        uint64_t* trash = qp;
        qp = new_q;
        (*q) = new_q;
        delete[] trash;
        int32_t* new_q_val = new int32_t[q_sizep * 2];
        std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
        std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
        int32_t* trash_val = q_valp;
        q_valp = new_q_val;
        (*q_val) = new_q_val;
        delete[] trash_val;
        c_real* new_mus = new c_real[q_sizep * 2];
        std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
        std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
        c_real* trash_mus = musp;
        musp = new_mus;
        (*mus) = new_mus;
        delete[] trash_mus;
        push_idx = q_sizep;
        pop_idx = 0;
        push_idx_row = q_sizep * 2;
        pop_idx_row = 0;
        q_sizep *= 2;
        (*q_size) = q_sizep;
      }
      while (push_idx != pop_idx) {
        c_real curr_mu = musp[pop_idx];
        c_real bcl = *best_corr_local;
        c_real bcg = best_corr_global;
        if (curr_mu > (bcg > bcl ? bcg : bcl) + delta) {
          uint64_t freql = qp[pop_idx_row + 0];
          uint64_t freqh = qp[pop_idx_row + 1];
          int32_t value = q_valp[pop_idx];
          c_real mu1 = 0;
          c_real mu2 = 0;
          int32_t i = 0;
          #if AVX_EXPERIMENT
          __m256d mu1v = _mm256_setzero_pd();
          __m256d mu2v = _mm256_setzero_pd();
          __m256d zeros = _mm256_setzero_pd();
           __m256i vv = _mm256_set_epi64x(freqh, freql, freqh, freql);
          int32_t samples2 = (samples / 2) * 2;
          for (; i < samples2; i += 2) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * 2));
            __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
            __m256d bv = _mm256_and_pd(_mm256_permute_pd(_mm256_castsi256_pd(av), 0x5), _mm256_castsi256_pd(av));
            if (!_mm256_testz_pd(bv, bv)) {
              double y0 = y[i];
              double y1 = y[i + 1];
              __m256d yv = _mm256_set_pd(0, y1, 0, y0);
              __m256d gvlt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_LT_OQ), bv);
              __m256d gvgt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_GT_OQ), bv);
              mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gvlt));
              mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(zeros, yv, gvgt));
            }
          }
          mu1 += _mm256_reduce_add_pd(mu1v);
          mu2 += _mm256_reduce_add_pd(mu2v); 
          #endif
          for (; i < samples; ++i) {
            if (((x[i * 2] & freql) == freql) && ((x[i * 2 + 1] & freqh) == freqh)) {
              c_real y1 = y[i];
              bool ycz = (y1 > 0);
              mu1 += ycz * y1;
              mu2 -= !ycz * y1;
            }
          }
          c_real corr = mu1 - mu2;    
          bcg = best_corr_global;
          c_real bc = bcg > bcl ? bcg : bcl;
          if (corr > bc) {
            *best_corr_local = corr;
            bc = corr;
            local_b[0] = freql;
            local_b[1] = freqh;
          }
          if (mu1 > bc + delta) {
            for (size_t i = value + 1; i < n; i++) {
              bool islb = (i < 64);
              uint64_t set1 = ((uint64_t)1 << ((i & 63)));
              qp[push_idx_row + 0] = (freql | (islb * set1));
              qp[push_idx_row + 1] = (freqh | (!islb * set1));
              q_valp[push_idx] = i;
              musp[push_idx] = mu1;
              ++push_idx;
              bool endb = (push_idx != q_sizep);
              push_idx = endb * push_idx;
              push_idx_row = endb * (push_idx_row + 2);
              if (push_idx == pop_idx) {
                uint64_t* new_q = new uint64_t[q_sizep * 4];
                std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 2 - pop_idx_row) * sizeof(uint64_t));
                std::memcpy(new_q + q_sizep * 2 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
                uint64_t* trash = qp;
                qp = new_q;
                (*q) = new_q;
                delete[] trash;
                int32_t* new_q_val = new int32_t[q_sizep * 2];
                std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
                std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
                int32_t* trash_val = q_valp;
                q_valp = new_q_val;
                (*q_val) = new_q_val;
                delete[] trash_val;
                c_real* new_mus = new c_real[q_sizep * 2];
                std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
                std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
                c_real* trash_mus = musp;
                musp = new_mus;
                (*mus) = new_mus;
                delete[] trash_mus;
                push_idx = q_sizep;
                pop_idx = 0;
                push_idx_row = q_sizep * 2;
                pop_idx_row = 0;
                q_sizep *= 2;
                (*q_size) = q_sizep;
              }
            }
            #if GET_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
            local_max_queue = std::max(local_max_queue, cur_qsize);
            local_total_processed += (n - value - 1);
            #endif
          }
        }
        ++pop_idx;
        bool endb2 = (pop_idx != q_sizep);
        pop_idx = endb2 * pop_idx;
        pop_idx_row = endb2 * (pop_idx_row + 2);
      }    
    } else {
    #endif
    
      int64_t ibits = (first_q_val >> 6);
      for (int64_t j = 0; j < ibits; ++j) {
        qp[push_idx_row + j] = 0;
      }
      qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
      for (int64_t j = ibits + 1; j < featbits; ++j) {
        qp[push_idx_row + j] = 0;
      }
      q_valp[push_idx] = first_q_val;
      ++push_idx;
      bool endb = (push_idx != q_sizep);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + featbits);
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_sizep * featbits * 2];
        std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * featbits * sizeof(uint64_t));
        std::memcpy(new_q + q_sizep * featbits - pop_idx_row, qp, pop_idx * featbits * sizeof(uint64_t));
        uint64_t* trash = qp;
        qp = new_q;
        (*q) = new_q;
        delete[] trash;
        int32_t* new_q_val = new int32_t[q_sizep * 2];
        std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
        std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
        int32_t* trash_val = q_valp;
        q_valp = new_q_val;
        (*q_val) = new_q_val;
        delete[] trash_val;
        c_real* new_mus = new c_real[q_sizep * 2];
        std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
        std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
        c_real* trash_mus = musp;
        musp = new_mus;
        (*mus) = new_mus;
        delete[] trash_mus;
        push_idx = q_sizep;
        pop_idx = 0;
        push_idx_row = q_sizep * featbits;
        pop_idx_row = 0;
        q_sizep *= 2;
        (*q_size) = q_sizep;
      }

      while (push_idx != pop_idx) {
        c_real curr_mu = musp[pop_idx];
        c_real bcl = *best_corr_local;
        c_real bcg = best_corr_global;
        if (curr_mu > (bcg > bcl ? bcg : bcl) + delta) {
          uint64_t* freq = qp + pop_idx_row;
          int32_t value = q_valp[pop_idx];
          c_real mu1 = 0;
          c_real mu2 = 0;
          for (int64_t i = 0; i < samples; ++i) {
            int32_t j = 0;
            #if AVX_EXPERIMENT
            int32_t featbits4 = (featbits / 4) * 4;
            for (; j < featbits4; j += 4) {
              __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * featbits + j));
              __m256i vv = _mm256_loadu_si256((__m256i*)(freq + j));
              __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
              if (!_mm256_testz_si256(av, av)) {
                break;
              }
            }
            #endif
            for (; j < featbits; ++j) {
              if ((x[i * featbits + j] & freq[j]) != freq[j]) {
                break;
              }
            }
            if (j == featbits) {
              c_real y1 = y[i];
              bool ycz = (y1 > 0);
              mu1 += ycz * y1;
              mu2 -= !ycz * y1;
            }
          }

          c_real corr = mu1 - mu2;
          bcg = best_corr_global;
          c_real bc = bcg > bcl ? bcg : bcl;
          if (corr > bc) {
            *best_corr_local = corr;
            bc = corr;
            std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
          }
          if (mu1 > bc + delta) {
            
            for (size_t i = value + 1; i < n; i++) {
              int64_t ibits = (i >> 6);
              for (int64_t j = 0; j < ibits; ++j) {
                qp[push_idx_row + j] = freq[j];
              }
              qp[push_idx_row + ibits] = (freq[ibits] | ((uint64_t)1 << ((i & 63))));
              for (int64_t j = ibits + 1; j < featbits; ++j) {
                qp[push_idx_row + j] = freq[j];
              }
              q_valp[push_idx] = i;
              musp[push_idx] = mu1;
              ++push_idx;
              bool endb = (push_idx != q_sizep);
              push_idx = endb * push_idx;
              push_idx_row = endb * (push_idx_row + featbits);

              if (push_idx == pop_idx) {
                int64_t freq_idx = freq - qp;
                uint64_t* new_q = new uint64_t[q_sizep * featbits * 2];
                std::memcpy(new_q, qp + pop_idx_row, (q_sizep * featbits - pop_idx_row) * sizeof(uint64_t));
                std::memcpy(new_q + q_sizep * featbits - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
                uint64_t* trash = qp;
                qp = new_q;
                (*q) = new_q;
                delete[] trash;
                int32_t* new_q_val = new int32_t[q_sizep * 2];
                std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
                std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
                int32_t* trash_val = q_valp;
                q_valp = new_q_val;
                (*q_val) = new_q_val;
                delete[] trash_val;
                c_real* new_mus = new c_real[q_sizep * 2];
                std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
                std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
                c_real* trash_mus = musp;
                musp = new_mus;
                (*mus) = new_mus;
                delete[] trash_mus;
                if (freq_idx >= pop_idx_row) {
                  freq = (qp + freq_idx) - pop_idx_row;
                } else {
                  freq = (qp + freq_idx) + q_sizep * featbits - pop_idx_row;
                }
                push_idx = q_sizep;
                pop_idx = 0;
                push_idx_row = q_sizep * featbits;
                pop_idx_row = 0;
                q_sizep *= 2;
                (*q_size) = q_sizep;
              }
            }
            #if GET_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
            local_max_queue = std::max(local_max_queue, cur_qsize);
            local_total_processed += (n - value - 1);
            #endif
          }
        }
        if (++pop_idx == q_sizep) {
          pop_idx = 0;
          pop_idx_row = 0;
        } else {
          pop_idx_row += featbits;
        }
      }

    #if FEWER_LOOPS
    }
    #endif
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif
}

void cmcpos_part_rec(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed) {
      

  int64_t local_max_queue = 1;
  int64_t sub_max_queue = 1;
  int64_t local_total_processed = 1;
      
  size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  bool go_on = true;

  c_real mu1 = 0;
  c_real mu2 = 0;  
  if (first_q_val == 0) { // let the first thread handle the all-zeros case
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
		c_real corr = mu1 - mu2;

    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    c_real bc = bcg > bcl ? bcg : bcl;
		if (corr > bc) {
			*best_corr_local = corr;
      bc = corr;
      std::memset(local_b, 0, featbits * sizeof(uint64_t));
		}
		go_on = (mu1 > bc + delta);
  } else {
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      mu1 += y1 * (y1 > 0);
    }
  }

  if (go_on) {
    for (int32_t i = 0; i < featbits; ++i) {
      v[i] = 0;
    }
    v[ibits] = (v[ibits] | ((uint64_t)1 << ((first_q_val & 63))));
    cmcpos_part_rec_subcall(x, y, b, samples, features,
        best_corr_global, best_corr_local, local_b, delta, v, first_q_val, mu1,
        local_max_queue, sub_max_queue + 1, local_total_processed);
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmcpos_part_rec_subcall(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta, uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed) {
      
  c_real mu1 = 0.0;
  c_real mu2 = 0.0;
  
  #if GET_STATS
  max_queue = std::max(max_queue, sub_max_queue);
  ++total_processed;
  #endif
  
  c_real bcl = *best_corr_local;
  c_real bcg = best_corr_global;
  if (curr_mu <= (bcg > bcl ? bcg : bcl) + delta) {
    return;
  }
  
  int32_t featbits = ((features + 63) >> 6);
  
  #if FEWER_LOOPS
  
  if (featbits == 1) {
    uint64_t freq = v[0];
    int32_t i = 0;
    #if AVX_EXPERIMENT
    __m256d mu1v = _mm256_setzero_pd();
    __m256d mu2v = _mm256_setzero_pd();
    __m256d zeros = _mm256_setzero_pd();
    __m256i vv = _mm256_set1_epi64x(freq);
    int32_t samples4 = (samples / 4) * 4;
    for (; i < samples4; i += 4) {
      __m256i xv = _mm256_loadu_si256((__m256i*)(x + i));
      __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
      if (!_mm256_testz_si256(av, av)) {
        __m256d yv = _mm256_blendv_pd(zeros, _mm256_loadu_pd(y + i), _mm256_castsi256_pd(av));
        __m256d gv = _mm256_cmp_pd(zeros, yv, _CMP_LT_OQ);
        mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gv));
        mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(yv, zeros, gv));
      }
    }
    mu1 = _mm256_reduce_add_pd(mu1v);
    mu2 = _mm256_reduce_add_pd(mu2v);
    #endif
    for (; i < samples; ++i) {
      if ((x[i] & freq) == freq) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else if (featbits == 2) {
    uint64_t freql = v[0];
    uint64_t freqh = v[1];
    int32_t i = 0;
    #if AVX_EXPERIMENT
    __m256d mu1v = _mm256_setzero_pd();
    __m256d mu2v = _mm256_setzero_pd();
    __m256d zeros = _mm256_setzero_pd();
     __m256i vv = _mm256_set_epi64x(freqh, freql, freqh, freql);
    int32_t samples2 = (samples / 2) * 2;
    for (; i < samples2; i += 2) {
      __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * 2));
      __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
      __m256d bv = _mm256_and_pd(_mm256_permute_pd(_mm256_castsi256_pd(av), 0x5), _mm256_castsi256_pd(av));
      if (!_mm256_testz_pd(bv, bv)) {
        double y0 = y[i];
        double y1 = y[i + 1];
        __m256d yv = _mm256_set_pd(0, y1, 0, y0);
        __m256d gvlt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_LT_OQ), bv);
        __m256d gvgt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_GT_OQ), bv);
        mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gvlt));
        mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(zeros, yv, gvgt));
      }
    }
    mu1 += _mm256_reduce_add_pd(mu1v);
    mu2 += _mm256_reduce_add_pd(mu2v); 
    #endif
    for (; i < samples; ++i) {
      if (((x[i * 2] & freql) == freql) && ((x[i * 2 + 1] & freqh) == freqh)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else {

  #endif

    for (int64_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      #if AVX_EXPERIMENT
      int32_t featbits4 = (featbits / 4) * 4;
      for (; j < featbits4; j += 4) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * featbits + j));
        __m256i vv = _mm256_loadu_si256((__m256i*)(v + j));
        __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
        if (!_mm256_testz_si256(av, av)) {
          break;
        }
      }
      #endif
      for (; j < featbits; ++j) {
        if ((x[i * featbits + j] & v[j]) != v[j]) {
          break;
        }
      }

     if (j == featbits) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  
  #if FEWER_LOOPS
  }
  #endif
  c_real corr = mu1 - mu2;    
  bcg = best_corr_global;
  c_real bc = bcg > bcl ? bcg : bcl;
  if (corr > bc) {
    *best_corr_local = corr;
    bc = corr;
    std::memcpy(local_b, v, featbits * sizeof(uint64_t));
  }
  if (mu1 > bc + delta) {
    for (int32_t nt = top1 + 1; nt < features; ++nt) {
      int32_t ntt = (nt >> 6);
      v[ntt] = (v[ntt] | ((uint64_t)1 << ((nt & 63))));
      cmcpos_part_rec_subcall(x, y, b, samples, features,
          best_corr_global, best_corr_local, local_b, delta, v, nt, mu1,
          max_queue, sub_max_queue + 1, total_processed);
      v[ntt] = (v[ntt] & ~((uint64_t)1 << ((nt & 63))));
    }
  }
}

#if BIT_AVX

void cmcpos_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t vecbits = ((features + 255) >> 8);
  int32_t padded = (((features + 255) >> 8) << 2);
  int32_t featbits = ((features + 63) >> 6);
  int64_t ibits = (first_q_val >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
		
	std::queue<std::vector<uint64_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  std::queue<c_real> mus;
  std::vector<uint64_t> first_freq(padded, 0);
  
  bool go_on = true;
  
  if (first_q_val == 0) { // let the first thread handle the all-zeros case
    c_real mu1 = 0;
		c_real mu2 = 0;
    
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
		c_real corr = mu1 - mu2;

    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    c_real bc = bcg > bcl ? bcg : bcl;
		if (corr > bc) {
			*best_corr_local = corr;
      bc = corr;
      std::memset(local_b, 0, featbits * sizeof(uint64_t));
		}
    mus.push(mu1);
		go_on = (mu1 > bc + delta);
  } else {
    c_real mu1 = 0;
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      mu1 += y1 * (y1 > 0);
    }
    mus.push(mu1);
  }

  if (go_on) {
    first_freq[ibits] = ((uint64_t)1 << ((first_q_val & 63)));
    q.push(first_freq);
    q_val.push(first_q_val);

    while (!q.empty()) {
      auto freq = q.front();
      int32_t value = q_val.front();
      q.pop();
      q_val.pop();
      mus.pop();
      c_real bcl = *best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu <= (bcg > bcl ? bcg : bcl) + delta) {
        continue;
      }
      c_real mu1 = 0;
      c_real mu2 = 0;
      
      for (int64_t i = 0; i < samples; ++i) {
        int32_t j = 0;
        for (; j < vecbits; ++j) {
          __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded + (j << 2)));
          __m256i vv = _mm256_loadu_si256((__m256i*)(freq.data() + (j << 2)));
          __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
          if (!_mm256_testz_si256(av, av)) {
            break;
          }
        }

       if (j == vecbits) {
          c_real y1 = y[i];
          if (y1 > 0) {
            mu1 += y1;
          } else {
            mu2 -= y1;
          }
        }
      }
      c_real corr = mu1 - mu2;
      
      bcg = best_corr_global;
      c_real bc = bcg > bcl ? bcg : bcl;
      if (corr > bc) {
        *best_corr_local = corr;
        bc = std::fabs(corr);
        std::memcpy(local_b, freq.data(), featbits * sizeof(uint64_t));
      }
      if (mu1 > bc + delta) {
        for (size_t i = value + 1; i < n; i++) {
          int64_t ibits = (i >> 6);
          std::vector<uint64_t> freq_new(freq);
          freq_new[ibits] = (freq_new[ibits] | ((uint64_t)1 << ((i & 63))));
          q.push(freq_new);
          q_val.push(i);
          mus.push(mu1);
        }
        #if GET_STATS
        local_max_queue = std::max(local_max_queue, (int64_t)(q.size()));
        local_total_processed += (n - value - 1);
        #endif
      }
    }
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmcpos_part_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t** q, int32_t** q_val, c_real** mus, int32_t* q_size,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t vecbits = ((features + 255) >> 8);
  int32_t padded = (((features + 255) >> 8) << 2);
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	c_real best_corr = 0;
  
  uint64_t* qp = *q;
  int32_t* q_valp = *q_val;
  int32_t q_sizep = *q_size;

  int64_t push_idx = 0;
  int64_t pop_idx = 0;
  int64_t push_idx_row = 0;
  int64_t pop_idx_row = 0;
  
  bool go_on = true;
  
  if (first_q_val == 0) { // let the first thread handle the all-zeros case
    c_real mu1 = 0;
		c_real mu2 = 0;
    
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
		c_real corr = mu1 - mu2;

    bcg = best_corr_global;
    c_real bc = bcg > bcl ? bcg : bcl;
		if (corr > bc) {
			*best_corr_local = corr;
      bc = corr;
      std::memset(local_b, 0, featbits * sizeof(uint64_t));
		}
		go_on = (mu1 > bc + delta);
    musp[push_idx] = mu1;
  } else {
    c_real mu1 = 0;
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      mu1 += y1 * (y1 > 0);
    }
    musp[push_idx] = mu1;
  }

  if (go_on) {
  
    #if FEWER_LOOPS
    if (vecbits == 1) {
      
      int64_t ibits = (first_q_val >> 6);
      std::memset(qp + push_idx_row, 0, 4 * sizeof(uint64_t));
      qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
      q_valp[push_idx] = first_q_val;
      ++push_idx;
      bool endb = (push_idx != q_sizep);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + 4);
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_sizep * 4 * 2];
        std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 4 * sizeof(uint64_t));
        std::memcpy(new_q + q_sizep * 4 - pop_idx_row, qp, pop_idx * 4 * sizeof(uint64_t));
        uint64_t* trash = qp;
        qp = new_q;
        (*q) = new_q;
        delete[] trash;
        int32_t* new_q_val = new int32_t[q_sizep * 2];
        std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
        std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
        int32_t* trash_val = q_valp;
        q_valp = new_q_val;
        (*q_val) = new_q_val;
        delete[] trash_val;
        c_real* new_mus = new c_real[q_sizep * 2];
        std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
        std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
        c_real* trash_mus = musp;
        musp = new_mus;
        (*mus) = new_mus;
        delete[] trash_mus;
        push_idx = q_sizep;
        pop_idx = 0;
        push_idx_row = q_sizep * 4;
        pop_idx_row = 0;
        q_sizep *= 2;
        (*q_size) = q_sizep;
      }
      while (push_idx != pop_idx) {
        c_real curr_mu = musp[pop_idx];
        c_real bcl = *best_corr_local;
        c_real bcg = best_corr_global;
        if (curr_mu > (bcg > bcl ? bcg : bcl) + delta) {
          uint64_t* freq = qp + pop_idx_row;
          int32_t value = q_valp[pop_idx];
          c_real mu1 = 0;
          c_real mu2 = 0;
          
          __m256i vv  = _mm256_loadu_si256((__m256i*)(freq));
          for (int64_t i = 0; i < samples; ++i) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded));
            __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
            if (_mm256_testz_si256(av, av)) {
              c_real y1 = y[i];
              bool ycz = (y1 > 0);
              mu1 += ycz * y1;
              mu2 -= !ycz * y1;
            }
          }
          c_real corr = mu1 - mu2;
          bcg = best_corr_global;
          c_real bc = bcg > bcl ? bcg : bcl;
          if (corr > bc) {
            *best_corr_local = corr;
            bc = corr;
            std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
          }
          
          if (mu1 > bc + delta) {
            for (size_t i = value + 1; i < n; i++) {
              int64_t ibits = (i >> 6);
              std::memcpy(qp + push_idx_row, freq, 4 * sizeof(uint64_t));
              qp[push_idx_row + ibits] = (qp[push_idx_row + ibits] | ((uint64_t)1 << ((i & 63))));
              q_valp[push_idx] = i;
              musp[push_idx] = mu1;
              ++push_idx;
              bool endb = (push_idx != q_sizep);
              push_idx = endb * push_idx;
              push_idx_row = endb * (push_idx_row + 4);
              if (push_idx == pop_idx) {
                int64_t freq_idx = freq - qp;
                uint64_t* new_q = new uint64_t[q_sizep * 4 * 2];
                std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 4 - pop_idx_row) * sizeof(uint64_t));
                std::memcpy(new_q + q_sizep * 4 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
                uint64_t* trash = qp;
                qp = new_q;
                (*q) = new_q;
                delete[] trash;
                int32_t* new_q_val = new int32_t[q_sizep * 2];
                std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
                std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
                int32_t* trash_val = q_valp;
                q_valp = new_q_val;
                (*q_val) = new_q_val;
                delete[] trash_val;
                c_real* new_mus = new c_real[q_sizep * 2];
                std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
                std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
                c_real* trash_mus = musp;
                musp = new_mus;
                (*mus) = new_mus;
                delete[] trash_mus;
                if (freq_idx >= pop_idx_row) {
                  freq = (qp + freq_idx) - pop_idx_row;
                } else {
                  freq = (qp + freq_idx) + q_sizep * 4 - pop_idx_row;
                }
                push_idx = q_sizep;
                pop_idx = 0;
                push_idx_row = q_sizep * 4;
                pop_idx_row = 0;
                q_sizep *= 2;
                (*q_size) = q_sizep;
              }
            }
            #if GET_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
            local_max_queue = std::max(local_max_queue, cur_qsize);
            local_total_processed += (n - value - 1);
            #endif
          }
        }
        ++pop_idx;
        bool endb2 = (pop_idx != q_sizep);
        pop_idx = endb2 * pop_idx;
        pop_idx_row = endb2 * (pop_idx_row + padded);
      }
      
    } else if (vecbits == 2) {

      int64_t ibits = (first_q_val >> 6);
      //~ bool islb = (first_q_val < 64);
      uint64_t set1 = ((uint64_t)1 << ((first_q_val & 63)));
      std::memset(qp + push_idx_row, 0, 8 * sizeof(uint64_t));
      qp[push_idx_row + ibits] = set1;
      q_valp[push_idx] = first_q_val;
      ++push_idx;
      bool endb = (push_idx != q_sizep);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + 8);
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_sizep * 4];
        std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 8 * sizeof(uint64_t));
        std::memcpy(new_q + q_sizep * 8 - pop_idx_row, qp, pop_idx * 8 * sizeof(uint64_t));
        uint64_t* trash = qp;
        qp = new_q;
        (*q) = new_q;
        delete[] trash;
        int32_t* new_q_val = new int32_t[q_sizep * 8];
        std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
        std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
        int32_t* trash_val = q_valp;
        q_valp = new_q_val;
        (*q_val) = new_q_val;
        delete[] trash_val;
        c_real* new_mus = new c_real[q_sizep * 2];
        std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
        std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
        c_real* trash_mus = musp;
        musp = new_mus;
        (*mus) = new_mus;
        delete[] trash_mus;
        push_idx = q_sizep;
        pop_idx = 0;
        push_idx_row = q_sizep * 8;
        pop_idx_row = 0;
        q_sizep *= 2;
        (*q_size) = q_sizep;
      }
      while (push_idx != pop_idx) {
        c_real curr_mu = musp[pop_idx];
        c_real bcl = *best_corr_local;
        c_real bcg = best_corr_global;
        if (curr_mu > (bcg > bcl ? bcg : bcl) + delta) {
          uint64_t* freql = qp + pop_idx_row;
          uint64_t* freqh = qp + pop_idx_row + 4;
          int32_t value = q_valp[pop_idx];
          c_real mu1 = 0;
          c_real mu2 = 0;
          __m256i vvl = _mm256_loadu_si256((__m256i*)(freql));
          __m256i vvh = _mm256_loadu_si256((__m256i*)(freqh));
          for (int64_t i = 0; i < samples; ++i) {
            __m256i xvl = _mm256_loadu_si256((__m256i*)(x + i * padded));
            __m256i avl = _mm256_xor_si256(_mm256_and_si256(xvl, vvl), vvl);
            __m256i xvh = _mm256_loadu_si256((__m256i*)(x + i * padded + 4));
            __m256i avh = _mm256_xor_si256(_mm256_and_si256(xvh, vvh), vvh);
            if (_mm256_testz_si256(avl, avl) && _mm256_testz_si256(avh, avh)) {
              c_real y1 = y[i];
              bool ycz = (y1 > 0);
              mu1 += ycz * y1;
              mu2 -= !ycz * y1;
            }
          }
          c_real corr = mu1 - mu2;
          bcg = best_corr_global;
          c_real bc = bcg > bcl ? bcg : bcl;
          if (corr > bc) {
            *best_corr_local = corr;
            bc = corr;
            std::memcpy(local_b, freql, featbits * sizeof(uint64_t));
          }
          
          if (mu1 > bc + delta) {
            for (size_t i = value + 1; i < n; i++) {
              int64_t freq_idx = freql - qp;
              int64_t ibits = (i >> 6);
              uint64_t set1 = ((uint64_t)1 << ((i & 63)));
              std::memcpy(qp + push_idx_row, freql, 8 * sizeof(uint64_t));
              qp[push_idx_row + ibits] = set1;
              q_valp[push_idx] = i;
              musp[push_idx] = mu1;
              ++push_idx;
              bool endb = (push_idx != q_sizep);
              push_idx = endb * push_idx;
              push_idx_row = endb * (push_idx_row + 8);
              if (push_idx == pop_idx) {
                uint64_t* new_q = new uint64_t[q_sizep * 4];
                std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 8 - pop_idx_row) * sizeof(uint64_t));
                std::memcpy(new_q + q_sizep * 8 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
                uint64_t* trash = qp;
                qp = new_q;
                (*q) = new_q;
                delete[] trash;
                int32_t* new_q_val = new int32_t[q_sizep * 8];
                std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
                std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
                int32_t* trash_val = q_valp;
                q_valp = new_q_val;
                (*q_val) = new_q_val;
                delete[] trash_val;
                c_real* new_mus = new c_real[q_sizep * 2];
                std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
                std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
                c_real* trash_mus = musp;
                musp = new_mus;
                (*mus) = new_mus;
                delete[] trash_mus;
                if (freq_idx >= pop_idx_row) {
                  freql = (qp + freq_idx) - pop_idx_row;
                } else {
                  freql = (qp + freq_idx) + q_sizep * 8 - pop_idx_row;
                }
                freqh = freql + 4;
                push_idx = q_sizep;
                pop_idx = 0;
                push_idx_row = q_sizep * 8;
                pop_idx_row = 0;
                q_sizep *= 2;
                (*q_size) = q_sizep;
              }
            }
            #if GET_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
            local_max_queue = std::max(local_max_queue, cur_qsize);
            local_total_processed += (n - value - 1);
            #endif
          }
        }
        ++pop_idx;
        bool endb2 = (pop_idx != q_sizep);
        pop_idx = endb2 * pop_idx;
        pop_idx_row = endb2 * (pop_idx_row + padded);
      }    
    } else {
    #endif
    
      int64_t ibits = (first_q_val >> 6);
      for (int64_t j = 0; j < ibits; ++j) {
        qp[push_idx_row + j] = 0;
      }
      qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
      for (int64_t j = ibits + 1; j < padded; ++j) {
        qp[push_idx_row + j] = 0;
      }
      q_valp[push_idx] = first_q_val;
      ++push_idx;
      bool endb = (push_idx != q_sizep);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + padded);
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_sizep * padded * 2];
        std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * padded * sizeof(uint64_t));
        std::memcpy(new_q + q_sizep * padded - pop_idx_row, qp, pop_idx * padded * sizeof(uint64_t));
        uint64_t* trash = qp;
        qp = new_q;
        (*q) = new_q;
        delete[] trash;
        int32_t* new_q_val = new int32_t[q_sizep * 2];
        std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
        std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
        int32_t* trash_val = q_valp;
        q_valp = new_q_val;
        (*q_val) = new_q_val;
        delete[] trash_val;
        c_real* new_mus = new c_real[q_sizep * 2];
        std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
        std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
        c_real* trash_mus = musp;
        musp = new_mus;
        (*mus) = new_mus;
        delete[] trash_mus;
        push_idx = q_sizep;
        pop_idx = 0;
        push_idx_row = q_sizep * padded;
        pop_idx_row = 0;
        q_sizep *= 2;
        (*q_size) = q_sizep;
      }

      while (push_idx != pop_idx) {
        c_real curr_mu = musp[pop_idx];
        c_real bcl = *best_corr_local;
        c_real bcg = best_corr_global;
        if (curr_mu > (bcg > bcl ? bcg : bcl) + delta) {
          uint64_t* freq = qp + pop_idx_row;
          int32_t value = q_valp[pop_idx];
          c_real mu1 = 0;
          c_real mu2 = 0;
          
          for (int64_t i = 0; i < samples; ++i) {
            int32_t j = 0;
            for (; j < vecbits; ++j) {
              __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded + (j << 2)));
              __m256i vv = _mm256_loadu_si256((__m256i*)(freq + (j << 2)));
              __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
              if (!_mm256_testz_si256(av, av)) {
                break;
              }
            }
           if (j == vecbits) {
              c_real y1 = y[i];
              bool ycz = (y1 > 0);
              mu1 += ycz * y1;
              mu2 -= !ycz * y1;
            }
          }

          c_real corr = mu1 - mu2;
          bcg = best_corr_global;
          c_real bc = bcg > bcl ? bcg : bcl;
          if (corr > bc) {
            *best_corr_local = corr;
            bc = corr;
            std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
          }
          
         if (mu1 > bc + delta) {
            
            for (size_t i = value + 1; i < n; i++) {
              int64_t ibits = (i >> 6);
              for (int64_t j = 0; j < ibits; ++j) {
                qp[push_idx_row + j] = freq[j];
              }
              qp[push_idx_row + ibits] = (freq[ibits] | ((uint64_t)1 << ((i & 63))));
              for (int64_t j = ibits + 1; j < featbits; ++j) {
                qp[push_idx_row + j] = freq[j];
              }
              q_valp[push_idx] = i;
              musp[push_idx] = mu1;
              ++push_idx;
              bool endb = (push_idx != q_sizep);
              push_idx = endb * push_idx;
              push_idx_row = endb * (push_idx_row + padded);

              if (push_idx == pop_idx) {
                int64_t freq_idx = freq - qp;
                uint64_t* new_q = new uint64_t[q_sizep * padded * 2];
                std::memcpy(new_q, qp + pop_idx_row, (q_sizep * padded - pop_idx_row) * sizeof(uint64_t));
                std::memcpy(new_q + q_sizep * padded - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
                uint64_t* trash = qp;
                qp = new_q;
                (*q) = new_q;
                delete[] trash;
                int32_t* new_q_val = new int32_t[q_sizep * 2];
                std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
                std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
                int32_t* trash_val = q_valp;
                q_valp = new_q_val;
                (*q_val) = new_q_val;
                delete[] trash_val;
                c_real* new_mus = new c_real[q_sizep * 2];
                std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
                std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
                c_real* trash_mus = musp;
                musp = new_mus;
                (*mus) = new_mus;
                delete[] trash_mus;
                if (freq_idx >= pop_idx_row) {
                  freq = (qp + freq_idx) - pop_idx_row;
                } else {
                  freq = (qp + freq_idx) + q_sizep * padded - pop_idx_row;
                }
                push_idx = q_sizep;
                pop_idx = 0;
                push_idx_row = q_sizep * padded;
                pop_idx_row = 0;
                q_sizep *= 2;
                (*q_size) = q_sizep;
              }
            }
            #if GET_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
            local_max_queue = std::max(local_max_queue, cur_qsize);
            local_total_processed += (n - value - 1);
            #endif
          }
        }
        if (++pop_idx == q_sizep) {
          pop_idx = 0;
          pop_idx_row = 0;
        } else {
          pop_idx_row += padded;
        }
      }

    #if FEWER_LOOPS
    }
    #endif
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmcpos_part_rec_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, uint64_t* local_b, double delta,
    uint64_t* v, int64_t* max_queue, int64_t* total_processed) {
      

  int64_t local_max_queue = 1;
  int64_t sub_max_queue = 1;
  int64_t local_total_processed = 1;
      
  size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  int32_t padded = (((features + 255) >> 8) << 2);
  int64_t ibits = (first_q_val >> 6);
  
  bool go_on = true;
  
  c_real mu1 = 0;
  c_real mu2 = 0;
  if (first_q_val == 0) { // let the first thread handle the all-zeros case
   
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      if (y1 > 0) {
        mu1 += y1;
      } else {
        mu2 -= y1;
      }
    }
		c_real corr = mu1 - mu2;

    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    c_real bc = bcg > bcl ? bcg : bcl;
		if (corr > bc) {
			*best_corr_local = corr;
      bc = corr;
      std::memset(local_b, 0, featbits * sizeof(uint64_t));
		}
		go_on = (mu1 > bc + delta);
  } else {
		for (int64_t i = 0; i < samples; ++i) {
      c_real y1 = y[i];
      mu1 += y1 * (y1 > 0);
    }
  }

  if (go_on) {
    for (int32_t i = 0; i < padded; ++i) {
      v[i] = 0;
    }
    v[ibits] = (v[ibits] | ((uint64_t)1 << ((first_q_val & 63))));
    cmcpos_part_rec_subcall_avx(x, y, b, samples, features,
        best_corr_global, best_corr_local, local_b, delta, v, first_q_val, mu1,
        local_max_queue, sub_max_queue + 1, local_total_processed);
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmcpos_part_rec_subcall_avx(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, c_real& best_corr_global,
    c_real* best_corr_local, uint64_t* local_b, double delta, uint64_t* v, int32_t top1, c_real curr_mu,
    int64_t& max_queue, int64_t sub_max_queue, int64_t& total_processed) {
      
  #if GET_STATS
  max_queue = std::max(max_queue, sub_max_queue);
  ++total_processed;
  #endif
  
  c_real bcl = *best_corr_local;
  c_real bcg = best_corr_global;
  if (curr_mu <= (bcg > bcl ? bcg : bcl) + delta) {
    return;
  }
      
  c_real mu1 = 0.0;
  c_real mu2 = 0.0;
  
  int32_t vecbits = ((features + 255) >> 8);
  int32_t padded = (((features + 255) >> 8) << 2);
  int32_t featbits = ((features + 63) >> 6);
  
  #if FEWER_LOOPS
  
  if (vecbits == 1) {
    
    __m256i vv  = _mm256_loadu_si256((__m256i*)(v));
    for (int64_t i = 0; i < samples; ++i) {
      __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded));
      __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
      if (_mm256_testz_si256(av, av)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else if (vecbits == 2) {
    __m256i vvl = _mm256_loadu_si256((__m256i*)(v));
    __m256i vvh = _mm256_loadu_si256((__m256i*)(v + 4));
    for (int64_t i = 0; i < samples; ++i) {
      __m256i xvl = _mm256_loadu_si256((__m256i*)(x + i * padded));
      __m256i avl = _mm256_xor_si256(_mm256_and_si256(xvl, vvl), vvl);
      __m256i xvh = _mm256_loadu_si256((__m256i*)(x + i * padded + 4));
      __m256i avh = _mm256_xor_si256(_mm256_and_si256(xvh, vvh), vvh);
      if (_mm256_testz_si256(avl, avl) && _mm256_testz_si256(avh, avh)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else {

  #endif

    for (int64_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      for (; j < vecbits; ++j) {
        __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * padded + (j << 2)));
        __m256i vv = _mm256_loadu_si256((__m256i*)(v + (j << 2)));
        __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
        if (!_mm256_testz_si256(av, av)) {
          break;
        }
      }

     if (j == vecbits) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  
  #if FEWER_LOOPS
  }
  #endif
  c_real corr = mu1 - mu2;    
  bcg = best_corr_global;
  c_real bc = bcg > bcl ? bcg : bcl;
  if (corr > bc) {
    *best_corr_local = corr;
    bc = corr;
    std::memcpy(local_b, v, featbits * sizeof(uint64_t));
  }
  if (mu1 > bc + delta) {
    for (int32_t nt = top1 + 1; nt < features; ++nt) {
      int32_t ntt = (nt >> 6);
      v[ntt] = (v[ntt] | ((uint64_t)1 << ((nt & 63))));
      cmcpos_part_rec_subcall_avx(x, y, b, samples, features,
          best_corr_global, best_corr_local, local_b, delta, v, nt, mu1,
          max_queue, sub_max_queue + 1, total_processed);
      v[ntt] = (v[ntt] & ~((uint64_t)1 << ((nt & 63))));
    }
  }
}

#endif

void update_freq_range(uint64_t* x, uint64_t* b, bool* fb, size_t begin_idx, size_t end_idx, int32_t featbits) {
  for (size_t i = begin_idx; i < end_idx; i++) {
    int32_t j = 0;
    for (; j < featbits; ++j) {
      if ((x[i * featbits + j] & b[j]) != b[j]) {
        break;
      }
    }
    if (j == featbits) {
      fb[i] = 1;
    }
    else {
      fb[i] = 0;
    }
  }
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void cmc_part(uint64_t* x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real& best_corr_local, uint64_t* local_b, double delta,
    uint64_t* q, int32_t* q_val, c_real* mus, int32_t& q_size,
    int64_t& max_queue, int64_t& total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	c_real best_corr = 0;
  
  uint64_t* qp = q;
  int32_t* q_valp = q_val;
  c_real* musp = mus;
  int32_t q_sizep = q_size;

  int64_t push_idx = 0;
  int64_t pop_idx = 0;
  int64_t push_idx_row = 0;
  int64_t pop_idx_row = 0;
  
  #if FEWER_LOOPS
  if (featbits == 1) {
    
    qp[push_idx_row] = ((uint64_t)1 << ((first_q_val & 63)));
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * push_idx_row + endb;
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * 2];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep - pop_idx_row, qp, pop_idx * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      q = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      q_val = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      mus = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep;
      pop_idx_row = 0;
      q_sizep *= 2;
      q_size = q_sizep;
    }
    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t freq = qp[pop_idx_row];
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        int32_t i = 0;
        #if AVX_EXPERIMENT
        __m256d mu1v = _mm256_setzero_pd();
        __m256d mu2v = _mm256_setzero_pd();
        __m256d zeros = _mm256_setzero_pd();
        __m256i vv = _mm256_set1_epi64x(freq);
        int32_t samples4 = (samples / 4) * 4;
        for (; i < samples4; i += 4) {
          __m256i xv = _mm256_loadu_si256((__m256i*)(x + i));
          __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
          if (!_mm256_testz_si256(av, av)) {
            __m256d yv = _mm256_blendv_pd(zeros, _mm256_loadu_pd(y + i), _mm256_castsi256_pd(av));
            __m256d gv = _mm256_cmp_pd(zeros, yv, _CMP_LT_OQ);
            mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gv));
            mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(yv, zeros, gv));
          }
        }
        mu1 = _mm256_reduce_add_pd(mu1v);
        mu2 = _mm256_reduce_add_pd(mu2v);
        #endif
        for (; i < samples; ++i) {
          if ((x[i] & freq) == freq) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }
        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          best_corr_local = corr;
          bc = std::fabs(corr);
          local_b[0] = freq;
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            qp[push_idx_row] = (freq | ((uint64_t)1 << ((i & 63))));
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * push_idx_row + endb;
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_sizep * 2];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              q = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              q_val = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              mus = new_mus;
              delete[] trash_mus;
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep;
              pop_idx_row = 0;
              q_sizep *= 2;
              q_size = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_sizep);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * pop_idx_row + endb2;
    }
    
  } else if (featbits == 2) {

    bool islb = (first_q_val < 64);
    uint64_t set1 = ((uint64_t)1 << ((first_q_val & 63)));
    qp[push_idx_row + 0] = islb * set1;
    qp[push_idx_row + 1] = !islb * set1;
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + 2);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * 4];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * 2 * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * 2 - pop_idx_row, qp, pop_idx * 2 * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      q = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      q_val = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      mus = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * 2;
      pop_idx_row = 0;
      q_sizep *= 2;
      q_size = q_sizep;
    }
    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t freql = qp[pop_idx_row + 0];
        uint64_t freqh = qp[pop_idx_row + 1];
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        int32_t i = 0;
        #if AVX_EXPERIMENT
        __m256d mu1v = _mm256_setzero_pd();
        __m256d mu2v = _mm256_setzero_pd();
        __m256d zeros = _mm256_setzero_pd();
         __m256i vv = _mm256_set_epi64x(freqh, freql, freqh, freql);
        int32_t samples2 = (samples / 2) * 2;
        for (; i < samples2; i += 2) {
          __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * 2));
          __m256i av = _mm256_cmpeq_epi64(_mm256_and_si256(xv, vv), vv);
          __m256d bv = _mm256_and_pd(_mm256_permute_pd(_mm256_castsi256_pd(av), 0x5), _mm256_castsi256_pd(av));
          if (!_mm256_testz_pd(bv, bv)) {
            double y0 = y[i];
            double y1 = y[i + 1];
            __m256d yv = _mm256_set_pd(0, y1, 0, y0);
            __m256d gvlt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_LT_OQ), bv);
            __m256d gvgt = _mm256_and_pd(_mm256_cmp_pd(zeros, yv, _CMP_GT_OQ), bv);
            mu1v = _mm256_add_pd(mu1v, _mm256_blendv_pd(zeros, yv, gvlt));
            mu2v = _mm256_sub_pd(mu2v, _mm256_blendv_pd(zeros, yv, gvgt));
          }
        }
        mu1 += _mm256_reduce_add_pd(mu1v);
        mu2 += _mm256_reduce_add_pd(mu2v); 
        #endif
        for (; i < samples; ++i) {
          if (((x[i * 2] & freql) == freql) && ((x[i * 2 + 1] & freqh) == freqh)) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }
        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          best_corr_local = corr;
          bc = std::fabs(corr);
          local_b[0] = freql;
          local_b[1] = freqh;
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            bool islb = (i < 64);
            uint64_t set1 = ((uint64_t)1 << ((i & 63)));
            qp[push_idx_row + 0] = (freql | (islb * set1));
            qp[push_idx_row + 1] = (freqh | (!islb * set1));
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + 2);
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_sizep * 4];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * 2 - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * 2 - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              q = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              q_val = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              mus = new_mus;
              delete[] trash_mus;
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * 2;
              pop_idx_row = 0;
              q_sizep *= 2;
              q_size = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_sizep);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * (pop_idx_row + 2);  
    }
  } else {
  #endif
  
    int64_t ibits = (first_q_val >> 6);
    for (int64_t j = 0; j < ibits; ++j) {
      qp[push_idx_row + j] = 0;
    }
    qp[push_idx_row + ibits] = ((uint64_t)1 << ((first_q_val & 63)));
    for (int64_t j = ibits + 1; j < featbits; ++j) {
      qp[push_idx_row + j] = 0;
    }
    q_valp[push_idx] = first_q_val;
    musp[push_idx] = std::numeric_limits<c_real>::max();
    ++push_idx;
    bool endb = (push_idx != q_sizep);
    push_idx = endb * push_idx;
    push_idx_row = endb * (push_idx_row + featbits);
    if (push_idx == pop_idx) {
      uint64_t* new_q = new uint64_t[q_sizep * featbits * 2];
      std::memcpy(new_q, qp + pop_idx_row, (q_sizep - pop_idx) * featbits * sizeof(uint64_t));
      std::memcpy(new_q + q_sizep * featbits - pop_idx_row, qp, pop_idx * featbits * sizeof(uint64_t));
      uint64_t* trash = qp;
      qp = new_q;
      q = new_q;
      delete[] trash;
      int32_t* new_q_val = new int32_t[q_sizep * 2];
      std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
      std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
      int32_t* trash_val = q_valp;
      q_valp = new_q_val;
      q_val = new_q_val;
      delete[] trash_val;
      c_real* new_mus = new c_real[q_sizep * 2];
      std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
      std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
      c_real* trash_mus = musp;
      musp = new_mus;
      mus = new_mus;
      delete[] trash_mus;
      push_idx = q_sizep;
      pop_idx = 0;
      push_idx_row = q_sizep * featbits;
      pop_idx_row = 0;
      q_sizep *= 2;
      q_size = q_sizep;
    }

    while (push_idx != pop_idx) {
      c_real curr_mu = musp[pop_idx];
      c_real bcl = best_corr_local;
      c_real bcg = best_corr_global;
      if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
        uint64_t* freq = qp + pop_idx_row;
        int32_t value = q_valp[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        
        for (int64_t i = 0; i < samples; ++i) {
          int32_t j = 0;
          #if AVX_EXPERIMENT
          int32_t featbits4 = (featbits / 4) * 4;
          for (; j < featbits4; j += 4) {
            __m256i xv = _mm256_loadu_si256((__m256i*)(x + i * featbits + j));
            __m256i vv = _mm256_loadu_si256((__m256i*)(freq + j));
            __m256i av = _mm256_xor_si256(_mm256_and_si256(xv, vv), vv);
            if (!_mm256_testz_si256(av, av)) {
              break;
            }
          }
          #endif
          for (; j < featbits; ++j) {
            if ((x[i * featbits + j] & freq[j]) != freq[j]) {
              break;
            }
          }
          if (j == featbits) {
            c_real y1 = y[i];
            bool ycz = (y1 > 0);
            mu1 += ycz * y1;
            mu2 -= !ycz * y1;
          }
        }

        c_real corr = mu1 - mu2;
        bcg = best_corr_global;
        c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
        if (std::fabs(corr) > bc) {
          best_corr_local = corr;
          bc = std::fabs(corr);
          std::memcpy(local_b, freq, featbits * sizeof(uint64_t));
        }
        
        c_real max_mu = std::max(mu1, mu2);
        if (max_mu > bc + delta) {
          for (size_t i = value + 1; i < n; i++) {
            int64_t ibits = (i >> 6);
            for (int64_t j = 0; j < ibits; ++j) {
              qp[push_idx_row + j] = freq[j];
            }
            qp[push_idx_row + ibits] = (freq[ibits] | ((uint64_t)1 << ((i & 63))));
            for (int64_t j = ibits + 1; j < featbits; ++j) {
              qp[push_idx_row + j] = freq[j];
            }
            q_valp[push_idx] = i;
            musp[push_idx] = max_mu;
            ++push_idx;
            bool endb = (push_idx != q_sizep);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + featbits);

            if (push_idx == pop_idx) {
              int64_t freq_idx = freq - qp;
              uint64_t* new_q = new uint64_t[q_sizep * featbits * 2];
              std::memcpy(new_q, qp + pop_idx_row, (q_sizep * featbits - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_sizep * featbits - pop_idx_row, qp, pop_idx_row * sizeof(uint64_t));
              uint64_t* trash = qp;
              qp = new_q;
              q = new_q;
              delete[] trash;
              int32_t* new_q_val = new int32_t[q_sizep * 2];
              std::memcpy(new_q_val, q_valp + pop_idx, (q_sizep - pop_idx) * sizeof(int32_t));
              std::memcpy(new_q_val + q_sizep - pop_idx, q_valp, pop_idx * sizeof(int32_t));
              int32_t* trash_val = q_valp;
              q_valp = new_q_val;
              q_val = new_q_val;
              delete[] trash_val;
              c_real* new_mus = new c_real[q_sizep * 2];
              std::memcpy(new_mus, musp + pop_idx, (q_sizep - pop_idx) * sizeof(c_real));
              std::memcpy(new_mus + q_sizep - pop_idx, musp, pop_idx * sizeof(c_real));
              c_real* trash_mus = musp;
              musp = new_mus;
              mus = new_mus;
              delete[] trash_mus;
              if (freq_idx >= pop_idx_row) {
                freq = (qp + freq_idx) - pop_idx_row;
              } else {
                freq = (qp + freq_idx) + q_sizep * featbits - pop_idx_row;
              }
              push_idx = q_sizep;
              pop_idx = 0;
              push_idx_row = q_sizep * featbits;
              pop_idx_row = 0;
              q_sizep *= 2;
              q_size = q_sizep;
            }
          }
          #if GET_STATS
          int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_sizep - pop_idx;
          local_max_queue = std::max(local_max_queue, cur_qsize);
          local_total_processed += (n - value - 1);
          #endif
        }
      }
      if (++pop_idx == q_sizep) {
        pop_idx = 0;
        pop_idx_row = 0;
      } else {
        pop_idx_row += featbits;
      }
    }

  #if FEWER_LOOPS
  }
  #endif
  
  #if GET_STATS
  max_queue = std::max(max_queue, local_max_queue);
  total_processed += local_total_processed;
  #endif

}

#if STRUCT_EXPERIMENT

void cmc_part(SetX x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, SetFreq* local_b, double delta,
    SetQueue* queue,
    int64_t* max_queue, int64_t* total_processed) {

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  
  #if GET_STATS
  int64_t local_max_queue = 1;
  int64_t local_total_processed = 1;
  #endif
  
	c_real best_corr = 0;
  //~ std::cout << "*reset*\n";
  queue->reset();
  queue->push_first(first_q_val);
  while (queue->not_empty()) {
    c_real curr_mu = queue->get_curr_mu();
    c_real bcl = *best_corr_local;
    c_real bcg = best_corr_global;
    if (curr_mu > (std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl)) + delta) {
      SetFreq freq = queue->front();
      c_real mu1 = 0;
      c_real mu2 = 0;
      int32_t i;
      for (i = 0; i < samples; ++i) {
        if (leq(freq, x, i)) {
          c_real y1 = y[i];
          bool ycz = (y1 > 0);
          mu1 += ycz * y1;
          mu2 -= !ycz * y1;
        }
      }
      c_real corr = mu1 - mu2;
      bcg = best_corr_global;
      c_real bc = std::fabs(bcg) > std::fabs(bcl) ? std::fabs(bcg) : std::fabs(bcl);
      if (std::fabs(corr) > bc) {
        *best_corr_local = corr;
        bc = std::fabs(corr);
        local_b->memcpy(freq);
      }
      c_real max_mu = std::max(mu1, mu2);
      //~ std::cout << "MAX_MU: " + std::to_string(max_mu) + "\n";
      if(std::isnan(max_mu)) {
        //~ std::cout << "mu1 " + std::to_string(mu1) + "\n";
        //~ std::cout << "mu2 " + std::to_string(mu2) + "\n";
        //~ for (int32_t i = 0; i < samples; ++i) {
          //~ std::cout << std::to_string(y[i]) + " ";
        //~ }
        //~ std::cout << "\n";
        exit(-1);
      }
      if (max_mu > bc + delta) {
        queue->push_next(freq, max_mu);
        #if GET_STATS
        int64_t cur_qsize = queue->get_cur_qsize();
        local_max_queue = std::max(local_max_queue, cur_qsize);
        local_total_processed += (n - freq.value - 1);
        //~ std::cout << std::to_string(local_max_queue) + " " + std::to_string(freq.value) + "\n";
        //~ std::cout << std::to_string(queue->push_idx) + " " + std::to_string(queue->pop_idx) + "\n";
        #endif
      }
    }
    queue->pop();
     //~ std::cout << std::to_string(queue->push_idx) + " " + std::to_string(queue->pop_idx) + "\n";
  }
  
  #if GET_STATS
  (*max_queue) = std::max((*max_queue), local_max_queue);
  (*total_processed) += local_total_processed;
  #endif

}

void cmc_part_pos(SetX x, c_real* y, uint64_t* b,
    int32_t samples, int32_t features, int32_t first_q_val,
    c_real& best_corr_global, c_real* best_corr_local, SetFreq* local_b, double delta,
    SetQueue* queue,
    int64_t* max_queue, int64_t* total_processed) {}

#endif
