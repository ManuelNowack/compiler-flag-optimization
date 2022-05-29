#include "correlation.hpp"

uint64_t opt_it_ctr;

c_real compute_max_correlation_bitwise(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
    int32_t samples, int32_t features) {

  #if PRINT_STATS
  int32_t n_processed = 0;
  int32_t n_updates = 0;
  int32_t max_queue_size = 0;
  #endif

	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  #if BIT_AVX
  int32_t rowbits = (((features + 255) >> 8) << 2);
  #else
  int32_t rowbits = featbits;
  #endif
  
	c_real best_corr = 0;
		
	std::queue<std::vector<uint64_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  std::queue<c_real> mus;
  
  for (size_t i = 0; i < n; i++) {
    int64_t ibits = (i >> 6);
    std::vector<uint64_t> freq_new(featbits, 0);
    freq_new[ibits] = ((uint64_t)1 << ((i & 63)));
    q.push(freq_new);
    q_val.push(i);
    mus.push(std::numeric_limits<c_real>::max());
    #if PRINT_STATS
    if (q.size() > max_queue_size) {
      max_queue_size = q.size();
    }
    ++n_processed;
    #endif
  }

	while (!q.empty()) {
     //~ ++opt_it_ctr;
		auto freq = q.front();
    int32_t value = q_val.front();
    c_real curr_mu = mus.front();
		q.pop();
    q_val.pop();
    mus.pop();
		c_real mu1 = 0;
		c_real mu2 = 0;
    if (curr_mu <= std::fabs(best_corr)) {
      continue;
    }
    
		for (int32_t i = 0; i < samples; ++i) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * rowbits + j] & freq[j]) != freq[j]) {
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
		if (std::fabs(corr) > std::fabs(best_corr)) {
			best_corr = corr;
      // best freq is a vector with highest correlation (will be a Node in graph setup)
			std::memcpy(b, freq.data(), featbits * sizeof(uint64_t));
      #if PRINT_STATS
      ++n_updates;
      #endif
		}
		if (std::max(mu1, mu2) > std::fabs(best_corr)) {
			
			for (size_t i = value + 1; i < n; i++) {
        int64_t ibits = (i >> 6);
				std::vector<uint64_t> freq_new(freq);
        freq_new[ibits] = (freq_new[ibits] | ((uint64_t)1 << ((i & 63))));
				q.push(freq_new);
				q_val.push(i);
        mus.push(std::max(mu1, mu2));
        #if PRINT_STATS
        if (q.size() > max_queue_size) {
          max_queue_size = q.size();
        }
        ++n_processed;
        #endif
			}
		}
	}
  
  #if PRINT_STATS
  std::cout << "Max queue:\t" << max_queue_size << "\tProcessed:\t" << n_processed << "\tUpdated:\t" << n_updates << "\n";
  #endif

	for (size_t i = 0; i < samples; i++) {
		int32_t j = 0;
    for (; j < featbits; ++j) {
      if ((x[i * rowbits + j] & b[j]) != b[j]) {
        break;
      }
    }
    if (j == featbits) {
			fb[i] = 1;
		} else {
			fb[i] = 0;
		}
	}
	
	return best_corr;
}

c_real compute_max_correlation_bitwise(uint64_t* x, c_real* y, uint64_t* b, bool* fb,
    int32_t samples, int32_t features, uint64_t*& q, int32_t*& q_val, c_real*& mus, int32_t& q_size) {
	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  #if BIT_AVX
  int32_t rowbits = (((features + 255) >> 8) << 2);
  #else
  int32_t rowbits = featbits;
  #endif
  
	c_real best_corr = 0;
  
  int32_t push_idx = 0;
  int32_t pop_idx = 0;
  int32_t push_idx_row = 0;
  int32_t pop_idx_row = 0;
  
  #if PRINT_STATS
  int32_t n_processed = 0;
  int32_t n_updates = 0;
  int32_t max_queue_size = 0;
  #endif
  
  //~ clocktime start1, end1, start2, end2;
  //~ get_time(&start1); 
  
  #if FEWER_LOOPS
  if (featbits == 1) {
    
    for (size_t i = 0; i < n; i++) {
      q[push_idx_row] = ((uint64_t)1 << ((i & 63)));
      q_val[push_idx] = i;
      mus[push_idx] = std::numeric_limits<c_real>::max();
      //~ std::cout << "( ";
      //~ for (int32_t j = 0; j < featbits; ++j) {
        //~ std::cout << q[push_idx_row + j] << " ";
      //~ }
      //~ std::cout << " ) at IDX " << i << "\n";
      ++push_idx;
      bool endb = (push_idx != q_size);
      push_idx = endb * push_idx;
      push_idx_row = endb * push_idx_row + endb;
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_size * 2];
        std::memcpy(new_q, q + pop_idx_row, (q_size - pop_idx) * sizeof(uint64_t));
        std::memcpy(new_q + q_size - pop_idx_row, q, pop_idx * sizeof(uint64_t));
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
        push_idx_row = q_size;
        pop_idx_row = 0;
        q_size *= 2;
      }
      #if PRINT_STATS
      int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
      if (cur_qsize > max_queue_size) {
        max_queue_size = cur_qsize;
      }
      ++n_processed;
      #endif
    }
    while (push_idx != pop_idx) {
      //~ ++opt_it_ctr;
      c_real curr_mu = mus[pop_idx];
      if (curr_mu > std::fabs(best_corr)) {
        uint64_t freq = q[pop_idx_row];
        int32_t value = q_val[pop_idx];
        c_real mu1 = 0;
        c_real mu2 = 0;
        //~ for (int32_t i = 0; i < samples; ++i) {
          //~ if ((x[i * rowbits] & freq) == freq) {
            //~ c_real y1 = y[i];
            //~ bool ycz = (y1 > 0);
            //~ mu1 += ycz * y1;
            //~ mu2 -= !ycz * y1;
          //~ }
        //~ }
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
        if (std::fabs(corr) > std::fabs(best_corr)) {
          best_corr = corr;
          b[0] = freq;
          #if PRINT_STATS
          ++n_updates;
          #endif
        }
        if (std::max(mu1, mu2) > std::fabs(best_corr)) {
          for (size_t i = value + 1; i < n; i++) {
            q[push_idx_row] = (freq | ((uint64_t)1 << ((i & 63))));
            q_val[push_idx] = i;
            mus[push_idx] = std::max(mu1, mu2);
            //~ std::cout << "( ";
            //~ for (int32_t j = 0; j < featbits; ++j) {
              //~ std::cout << q[push_idx_row + j] << " ";
            //~ }
            //~ std::cout << " ) at IDX " << i << "\n";
            ++push_idx;
            bool endb = (push_idx != q_size);
            push_idx = endb * push_idx;
            push_idx_row = endb * push_idx_row + endb;
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_size * 2];
              std::memcpy(new_q, q + pop_idx_row, (q_size - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_size - pop_idx_row, q, pop_idx_row * sizeof(uint64_t));
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
              push_idx_row = q_size;
              pop_idx_row = 0;
              q_size *= 2;
            }
            #if PRINT_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
            if (cur_qsize > max_queue_size) {
              max_queue_size = cur_qsize;
            }
            ++n_processed;
            #endif
          }
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_size);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * pop_idx_row + endb2;
    }
    uint64_t b0 = b[0];
    for (size_t i = 0; i < samples; i++) {
      fb[i] = ((x[i] & b0) == b0);
    }
    
  } else if (featbits == 2) {

    for (size_t i = 0; i < n; i++) {
      bool islb = (i < 64);
      uint64_t set1 = ((uint64_t)1 << ((i & 63)));
      q[push_idx_row + 0] = islb * set1;
      q[push_idx_row + 1] = !islb * set1;
      q_val[push_idx] = i;
      mus[push_idx] = std::numeric_limits<c_real>::max();
      //~ std::cout << "( ";
      //~ for (int32_t j = 0; j < featbits; ++j) {
        //~ std::cout << q[push_idx_row + j] << " ";
      //~ }
      //~ std::cout << " ) at IDX " << i << "\n";
      ++push_idx;
      bool endb = (push_idx != q_size);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + 2);
      if (push_idx == pop_idx) {
        uint64_t* new_q = new uint64_t[q_size * 4];
        std::memcpy(new_q, q + pop_idx_row, (q_size - pop_idx) * 2 * sizeof(uint64_t));
        std::memcpy(new_q + q_size * 2 - pop_idx_row, q, pop_idx * 2 * sizeof(uint64_t));
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
        push_idx_row = q_size * 2;
        pop_idx_row = 0;
        q_size *= 2;
      }
      #if PRINT_STATS
      int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
      if (cur_qsize > max_queue_size) {
        max_queue_size = cur_qsize;
      }
      ++n_processed;
      #endif
    }
    while (push_idx != pop_idx) {
       //~ ++opt_it_ctr;
      c_real curr_mu = mus[pop_idx];
      if (curr_mu > std::fabs(best_corr)) {
        uint64_t freql = q[pop_idx_row + 0];
        uint64_t freqh = q[pop_idx_row + 1];
        int32_t value = q_val[pop_idx];
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
        //~ for (int32_t i = 0; i < samples; ++i) {
          //~ if (((x[i * 2] & freql) == freql) && ((x[i * 2 + 1] & freqh) == freqh)) {
            //~ c_real y1 = y[i];
            //~ bool ycz = (y1 > 0);
            //~ mu1 += ycz * y1;
            //~ mu2 -= !ycz * y1;
          //~ }
        //~ }
        c_real corr = mu1 - mu2;
        if (std::fabs(corr) > std::fabs(best_corr)) {
          best_corr = corr;
          b[0] = freql;
          b[1] = freqh;
          #if PRINT_STATS
          ++n_updates;
          #endif
        }
        if (std::max(mu1, mu2) > std::fabs(best_corr)) {
          for (size_t i = value + 1; i < n; i++) {
            bool islb = (i < 64);
            uint64_t set1 = ((uint64_t)1 << ((i & 63)));
            q[push_idx_row + 0] = (freql | (islb * set1));
            q[push_idx_row + 1] = (freqh | (!islb * set1));
            q_val[push_idx] = i;
            mus[push_idx] = std::max(mu1, mu2);
            //~ std::cout << "( ";
            //~ for (int32_t j = 0; j < featbits; ++j) {
              //~ std::cout << q[push_idx_row + j] << " ";
            //~ }
            //~ std::cout << " ) at IDX " << i << "\n";
            ++push_idx;
            bool endb = (push_idx != q_size);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + 2);
            if (push_idx == pop_idx) {
              uint64_t* new_q = new uint64_t[q_size * 4];
              std::memcpy(new_q, q + pop_idx_row, (q_size * 2 - pop_idx_row) * sizeof(uint64_t));
              std::memcpy(new_q + q_size * 2 - pop_idx_row, q, pop_idx_row * sizeof(uint64_t));
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
              push_idx_row = q_size * 2;
              pop_idx_row = 0;
              q_size *= 2;
            }
            #if PRINT_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
            if (cur_qsize > max_queue_size) {
              max_queue_size = cur_qsize;
            }
            ++n_processed;
            #endif
          }
        }
      }
      ++pop_idx;
      bool endb2 = (pop_idx != q_size);
      pop_idx = endb2 * pop_idx;
      pop_idx_row = endb2 * (pop_idx_row + 2);
    }
    uint64_t b0 = b[0];
    uint64_t b1 = b[1];
    for (size_t i = 0; i < samples; i++) {
      fb[i] = ((x[i * rowbits] & b0) == b0) && ((x[i * rowbits + 1] & b1) == b1);
    }
    
  } else {
  #endif
  
    for (size_t i = 0; i < n; i++) {
      int64_t ibits = (i >> 6);
      for (int32_t j = 0; j < ibits; ++j) {
        q[push_idx_row + j] = 0;
      }
      q[push_idx_row + ibits] = ((uint64_t)1 << ((i & 63)));
      for (int32_t j = ibits + 1; j < featbits; ++j) {
        q[push_idx_row + j] = 0;
      }
      q_val[push_idx] = i;
      mus[push_idx] = std::numeric_limits<c_real>::max();
      
      //~ std::cout << "( ";
      //~ for (int32_t j = 0; j < featbits; ++j) {
        //~ std::cout << q[push_idx_row + j] << " ";
      //~ }
      //~ std::cout << " ) at IDX " << i << "\n";
      
      ++push_idx;
      bool endb = (push_idx != q_size);
      push_idx = endb * push_idx;
      push_idx_row = endb * (push_idx_row + featbits);
      
      //~ if (++push_idx == q_size) {
        //~ push_idx = 0;
        //~ push_idx_row = 0;
      //~ } else {
        //~ push_idx_row += featbits;
      //~ }
      if (push_idx == pop_idx) {
        //~ std::cout << "RESIZE:\n";
        //~ std::cout << "Old size: " << q_size << "\n";
        //~ std::cout << "New size: " << q_size * 2 << "\n";
        //~ std::cout << "1) Copy " << (q_size - pop_idx) << " elements from idx " << pop_idx << " to idx 0\n";
        //~ std::cout << "2) Copy " << pop_idx << " elements from idx 0 to idx " << (q_size - pop_idx) << "\n";
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
      #if PRINT_STATS
      int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
      if (cur_qsize > max_queue_size) {
        max_queue_size = cur_qsize;
      }
      ++n_processed;
      #endif
    }

    while (push_idx != pop_idx) {
       //~ ++opt_it_ctr;
      c_real curr_mu = mus[pop_idx];
      if (curr_mu > std::fabs(best_corr)) {
        uint64_t* freq = q + pop_idx_row;
        int32_t value = q_val[pop_idx];
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
        
        //~ for (int32_t i = 0; i < samples; ++i) {
          //~ int32_t j = 0;
          //~ for (; j < featbits; ++j) {
            //~ if ((x[i * rowbits + j] & freq[j]) != freq[j]) {
              //~ break;
            //~ }
          //~ }
          //~ if (j == featbits) {
            //~ c_real y1 = y[i];
            //~ bool ycz = (y1 > 0);
            //~ mu1 += ycz * y1;
            //~ mu2 -= !ycz * y1;
          //~ }
        //~ }

        c_real corr = mu1 - mu2;
        if (std::fabs(corr) > std::fabs(best_corr)) {
          best_corr = corr;
          // best freq is a vector with highest correlation (will be a Node in graph setup)
          std::memcpy(b, freq, featbits * sizeof(uint64_t));
          #if PRINT_STATS
          ++n_updates;
          #endif
        }
        if (std::max(mu1, mu2) > std::fabs(best_corr)) {
          
          for (size_t i = value + 1; i < n; i++) {
            int64_t ibits = (i >> 6);
            for (int32_t j = 0; j < ibits; ++j) {
              q[push_idx_row + j] = freq[j];
            }
            q[push_idx_row + ibits] = (freq[ibits] | ((uint64_t)1 << ((i & 63))));
            for (int32_t j = ibits + 1; j < featbits; ++j) {
              q[push_idx_row + j] = freq[j];
            }
            q_val[push_idx] = i;
            mus[push_idx] = std::max(mu1, mu2);
            
            //~ std::cout << "( ";
            //~ for (int32_t j = 0; j < featbits; ++j) {
              //~ std::cout << q[push_idx_row + j] << " ";
            //~ }
            //~ std::cout << " ) at IDX " << i << "\n";
        
            ++push_idx;
            bool endb = (push_idx != q_size);
            push_idx = endb * push_idx;
            push_idx_row = endb * (push_idx_row + featbits);
            
            //~ if (++push_idx == q_size) {
              //~ push_idx = 0;
              //~ push_idx_row = 0;
            //~ } else {
              //~ push_idx_row += featbits;
            //~ }
            if (push_idx == pop_idx) {
              //~ std::cout << "RESIZE:\n";
              //~ std::cout << "Old size: " << q_size << "\n";
              //~ std::cout << "New size: " << q_size * 2 << "\n";
              //~ std::cout << "1) Copy " << (q_size - pop_idx) << " elements from idx " << pop_idx << " to idx 0\n";
              //~ std::cout << "2) Copy " << pop_idx << " elements from idx 0 to idx " << (q_size - pop_idx) << "\n";
              //~ std::cout << "Old freq: ";
              //~ for (int32_t j = 0; j < featbits; ++j) {
                //~ std::cout << freq[j] << " ";
              //~ }
              //~ std::cout << "\n";
              int64_t freq_idx = freq - q;
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
                freq = (q + freq_idx) - pop_idx_row;
              } else {
                freq = (q + freq_idx) + q_size * featbits - pop_idx_row;
              }
              push_idx = q_size;
              pop_idx = 0;
              push_idx_row = q_size * featbits;
              pop_idx_row = 0;
              q_size *= 2;
              //~ std::cout << "New freq: ";
              //~ for (int32_t j = 0; j < featbits; ++j) {
                //~ std::cout << freq[j] << " ";
              //~ }
              //~ std::cout << "\n";
            }
            #if PRINT_STATS
            int64_t cur_qsize = push_idx >= pop_idx ? push_idx - pop_idx : push_idx + q_size - pop_idx;
            if (cur_qsize > max_queue_size) {
              max_queue_size = cur_qsize;
            }
            ++n_processed;
            #endif
          }
        }
      }
      if (++pop_idx == q_size) {
        pop_idx = 0;
        pop_idx_row = 0;
      } else {
        pop_idx_row += featbits;
      }
    }
    
    #if PRINT_STATS
    std::cout << "Max queue:\t" << max_queue_size << "\tProcessed:\t" << n_processed << "\tUpdated:\t" << n_updates << "\n";
    #endif
    
    //~ get_time(&end1);
    
    //~ std::cout << "Max queue:\t" << max_queue_size << "\n";
    //~ std::cout << "Processed:\t" << n_processed << "\n";

    //~ get_time(&start2);
    for (size_t i = 0; i < samples; i++) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if ((x[i * rowbits + j] & b[j]) != b[j]) {
          break;
        }
      }
      fb[i] = (j == featbits);
    }
    //~ get_time(&end2);
    
    //~ std::cout << "One:\t" << get_time_difference(&start1, &end1) << "\tTwo:\t" << get_time_difference(&start2, &end2) << "\n";
  
  #if FEWER_LOOPS
  }
  #endif
	
  return best_corr;
}


c_real compute_max_correlation_bitwise(int32_t* x, c_real* y, int32_t* b, bool* fb,
    int32_t samples, int32_t features) {

	size_t n = features;
  int32_t featbits = ((features + 31) >> 5);
  
	c_real best_corr = 0;
		
	std::queue<std::vector<int32_t>> q;
	std::queue<int32_t> q_val; // current max value from which we start pushing
  
  for (size_t i = 0; i < n; i++) {
    int32_t ibits = (i >> 5);
    std::vector<int32_t> freq_new(featbits, 0);
    freq_new[ibits] = ((int32_t)1 << ((i & 31)));
    q.push(freq_new);
    q_val.push(i);
  }
  

	while (!q.empty()) {
		auto freq = q.front();
    int32_t value = q_val.front();
		q.pop();
    q_val.pop();
		c_real mu1 = 0;
		c_real mu2 = 0;
    
		for (int32_t i = 0; i < samples; ++i) {
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
		if (std::fabs(corr) > std::fabs(best_corr)) {
			best_corr = corr;
      // best freq is a vector with highest correlation (will be a Node in graph setup)
			std::memcpy(b, freq.data(), featbits * sizeof(int32_t));
		}
		if (std::max(mu1, mu2) > std::fabs(best_corr)) {
			
			for (size_t i = value + 1; i < n; i++) {
        int32_t ibits = (i >> 5);
				std::vector<int32_t> freq_new(freq);
        freq_new[ibits] = (freq_new[ibits] | ((int32_t)1 << ((i & 31))));
				q.push(freq_new);
				q_val.push(i);
			}
		}
	}

	for (size_t i = 0; i < samples; i++) {
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
	
	return best_corr;
}

//~ #pragma inline_recursion(on)
inline void compute_max_correlation_bitwise_rec_subcall(uint64_t* x, c_real* y,
    uint64_t* b, bool* fb, uint64_t* v, int32_t top1, c_real curr_mu,
    int32_t samples, int32_t features, c_real& max_corr) {
  
  c_real mu1 = 0.0;
  c_real mu2 = 0.0;
  
  if (curr_mu <= std::fabs(max_corr)) {
    return;
  }
  
  int32_t featbits = ((features + 63) >> 6);
  #if BIT_AVX
  int32_t rowbits = (((features + 255) >> 8) << 2);
  #else
  int32_t rowbits = featbits;
  #endif
  
  //~ for (int32_t i = 0; i < featbits; ++i) {
    //~ std::cout << v[i] << " ";
  //~ }
  //~ std::cout << "\n";
  
  //~ ++opt_it_ctr;
  
  //~ std::vector<bool> active_new(samples, 0);
  bool any_active = false;
  
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
      if ((x[i * rowbits] & freq) == freq) {
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
      if (((x[i * rowbits] & freql) == freql)
          && ((x[i * rowbits + 1] & freqh) == freqh)) {
        c_real y1 = y[i];
        bool ycz = (y1 > 0);
        mu1 += ycz * y1;
        mu2 -= !ycz * y1;
      }
    }
  } else {

  #endif

       //~ if (v[0] == 1 && v[1] == 0) {
          //~ for (int32_t i = 0; i < samples; ++i) {
            //~ std::cout << (x[i * rowbits] & v[0]) << " "
              //~ << (x[i * rowbits + 1] & v[1]) << "\n";
          //~ }
       //~ }
       
    for (int32_t i = 0; i < samples; ++i) {
      //~ if (active[i]) {
        //~ any_active = true;
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
          if ((x[i * rowbits + j] & v[j]) != v[j]) {
            break;
          }
        }


       if (j == featbits) {
          c_real y1 = y[i];
          bool ycz = (y1 > 0);
          mu1 += ycz * y1;
          mu2 -= !ycz * y1;
          //~ active_new[i] = 1;
          //~ std::cout << "is in " << i << "\n";
        }
        
            
        //~ if (j == featbits) {
          //~ c_real y1 = y[i];
          //~ if (y1 > 0) {
            //~ mu1 += y1;
          //~ } else {
            //~ mu2 -= y1;
          //~ }
        //~ }
      //~ }
    }
  
  #if FEWER_LOOPS
  }
  #endif
  //~ if (any_active) {
    c_real new_corr = mu1 - mu2;
    if (std::fabs(new_corr) > std::fabs(max_corr)) {
      max_corr = new_corr;
      std::memcpy(b, v, featbits * sizeof(uint64_t));
    }
    //~ std::cout << "< ";
    //~ for (int32_t j = 0; j < featbits; ++j) {
      //~ std::cout << v[j] << " ";
    //~ }
    //~ std::cout << "> " << mu1 << " " << mu2 << " " << max_corr << "\n";
    c_real max_mu = std::max(mu1, mu2);
    if (max_mu > std::fabs(max_corr)) {
      for (int32_t nt = top1 + 1; nt < features; ++nt) {
        int32_t ntt = (nt >> 6);
        int64_t shifter = (ntt == featbits - 1) ? ((features - 1) & 63) : 63;
        //~ std::cout << shifter << "\n";
        v[ntt] = (v[ntt] | ((uint64_t)1 << (shifter-(nt & 63))));
        //~ for (int32_t k = 0; k < features; ++k) {
          //~ for (int32_t kk = 0; kk < 64 && k * 64 + kk < features; ++kk) {
              //~ std::cout << ((v[k] >> kk) & 1);
          //~ }
        //~ }
        //~ std::cout << "\n";
        //~ std::cout << "vntt " << v[ntt] << " (" << ((uint64_t)1 << (63 - (nt & 63))) << ") " << "\n";
        compute_max_correlation_bitwise_rec_subcall(x, y, b, fb, v, nt, max_mu, samples, features, max_corr);
        v[ntt] = (v[ntt] & ~((uint64_t)1 << (shifter-(nt & 63))));
        //~ std::cout << "~vntt " << v[ntt] << " (" << ~((uint64_t)1 << (63 - (nt & 63))) << ") " << "\n";
      }
      //~ for (int32_t nt = top1 + 1; nt < features; ++nt) {
        //~ int32_t ntt = (nt >> 6);
        //~ v[ntt] = (v[ntt] & ~((uint64_t)1 << ((nt & 63))));
      //~ }
    }
     
  //~ }
}

c_real compute_max_correlation_bitwise_rec(uint64_t* x, c_real* y, uint64_t* b, bool* fb, uint64_t* v, int32_t samples, int32_t features) {
  
  //~ std::cout << "*--------------------------*\n";
  
	size_t n = features;
  int32_t featbits = ((features + 63) >> 6);
  #if BIT_AVX
  int32_t rowbits = (((features + 255) >> 8) << 2);
  #else
  int32_t rowbits = featbits;
  #endif
  
	c_real best_corr = 0;
  
  for (int32_t i = 0; i < featbits; ++i) {
    v[i] = 0;
  }
  
  //~ std::vector<bool> active(samples, 1);
  
  c_real mu = std::numeric_limits<c_real>::max();
  /*for (int32_t nt = 0; nt < features; ++nt) {
    int32_t ntt = (nt >> 6);
    v[ntt] = (v[ntt] | ((uint64_t)1 << ((nt & 63))));
    for (int32_t k = 0; k < features; ++k) {
      for (int32_t kk = 0; kk < 64 && k * 64 + kk < features; ++kk) {
          std::cout << ((v[k] >> kk) & 1);
      }
    }
    std::cout << "\n";
    compute_max_correlation_bitwise_rec_subcall(x, y, b, fb, v, nt, mu,
        samples, features, best_corr, active);
    //~ v[ntt] = (v[ntt] & ~((uint64_t)1 << ((nt & 63))));
  }
  for (int32_t nt = 0; nt < features; ++nt) {
    int32_t ntt = (nt >> 6);
    v[ntt] = (v[ntt] & ~((uint64_t)1 << ((nt & 63))));
  }*/
  
  compute_max_correlation_bitwise_rec_subcall(x, y, b, fb, v, -1, mu,
        samples, features, best_corr);
  
	for (size_t i = 0; i < samples; i++) {
    int32_t j = 0;
    for (; j < featbits; ++j) {
      if ((x[i * rowbits + j] & b[j]) != b[j]) {
        break;
      }
    }
    fb[i] = (j == featbits);
  }
	return best_corr;
  
  
}
