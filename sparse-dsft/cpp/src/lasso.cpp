#include "lasso.hpp"

// FOR NOW, THIS IS A STRAIGHTFORWARD REWRITTEN VERSION OF THE PYTHON CODE

inline c_real prox(c_real w0, c_real C) {
  return std::max(1.0 - C / std::fabs(w0), 0.0) * w0;
}

// m samples
// n features
Estimator* coordinate_descent(bool* x, c_real* y, int32_t m, int32_t n, bool* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, Estimator* est, bool center, char function,
    DependencyGraph* dg, int64_t& corrtime, bool recursive) {
  std::map<std::vector<bool>, c_real> params;
  std::map<std::vector<bool>, bool*> feature_vecs;
  std::vector<bool> blocked_key(n, 0);
  c_real* residual = new c_real[m];
  c_real* a_j = new c_real[m];
  if (!est) {
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
  } else {
    int32_t fsize = est->freqs.size();
    for (int32_t s = 0; s < fsize; ++s) {
      std::vector<bool>& kfreq = est->freqs[s];
      params[kfreq] = est->coefs[s];
      bool* mvv = new bool[m];
      int32_t freq_sum = 0;
      for (int32_t j = 0; j < n; ++j) {
        freq_sum += kfreq[j];
      }
      for (int32_t i = 0; i < m; ++i) {
        int32_t temp_sum = 0;
        for (int32_t j = 0; j < n; ++j) {
          temp_sum += x[i * n + j] * kfreq[j];
        }
        mvv[i] = (temp_sum == freq_sum);
      }
      //~ for (int32_t i = 0; i < m; ++i) {
        //~ mvv[i] = true;
        //~ for (int32_t j = 0; j < n; ++j) {
          //~ if ((x[i * n + j] & kfreq[j]) != kfreq[j]) {
              //~ mvv[i] = false;
              //~ break;
          //~ }
        //~ }
      //~ }
      if (feature_vecs.count(kfreq) > 0) {
        delete[] feature_vecs[kfreq];
      }
      //~ std::cout << "Pushing ";
      //~ for (int32_t i = 0; i < n; ++i) {
        //~ std::cout << kfreq[i];
      //~ }
      //~ std::cout << "\n";
      feature_vecs[kfreq] = mvv;
    }
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
    for (int32_t s = 0; s < fsize; ++s) {
      for (int32_t i = 0; i < m; ++i) {
        bool active = true;
        for (int32_t j = 0; j < n; ++j) {
          if (est->freqs[s][j] * (1 - x[i * n + j]) != 0) {
            active = false;
            break;
          }
        }
        if (active) {
          residual[i] -= est->coefs[s];
                    //~ std::cout << est->coefs[s] << "\n";
        }
      }
    }
  }

  std::vector<std::vector<bool>> remove_idx_1;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    if (feature_vecs.count(blocked_key) > 0) {
      //~ std::cout << "Deleting ";
      //~ for (int32_t i = 0; i < n; ++i) {
        //~ std::cout << blocked_key[i];
      //~ }
      //~ std::cout << "\n";
      params.erase(blocked_key);
      delete[] feature_vecs[blocked_key];
      feature_vecs.erase(blocked_key);
    }
    for (auto& fv : feature_vecs) {
      if (center) {
        c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_1.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_1) {
      //~ std::cout << "Deleting ";
      //~ for (int32_t i = 0; i < n; ++i) {
        //~ std::cout << key[i];
      //~ }
      //~ std::cout << "\n";
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_1.clear();
  }

  int32_t no_improvement_since = 0;
  for (int32_t step = 0; step < steps; ++step) {
  
    c_real gain;
    clocktime start, end;
    get_time(&start);
    if (recursive) {
      gain = compute_max_correlation_recursive_baseline(x, residual, freq, fB, m, n);
    } else {
      gain = compute_max_correlation_baseline(x, residual, freq, fB, m, n);
    }
    get_time(&end);
    corrtime += get_time_difference(&start, &end);
    //~ std::cout << gain << "\t";
    //~ for (int32_t i = 0; i < n; ++i) {
      //~ std::cout << freq[i];
    //~ }
    //~ std::cout << "\t";
    //~ for (int32_t i = 0; i < m; ++i) {
      //~ std::cout << fB[i];
    //~ }
    //~ std::cout << "\n";
    bool found_key = false;
    for (auto& par : params) {
      int32_t j = 0;
      for (; j < n; ++j) {
        if (freq[j] != par.first[j]) {
          break;
        }
      }
      if (j == n) {
        ++no_improvement_since;
        found_key = true;
        break;
      }
    }
    if (!found_key) {
      no_improvement_since = 0;
    } else if (no_improvement_since > patience) {
      break;
    }
    std::vector<bool> key(freq, freq + n);
    bool* nfb = new bool[m];
    if (center) {
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        mean += fbval;
        nfb[i] = fbval;
      }
      mean /= m;
      for (int32_t i = 0; i < m; ++i) {
        a_j[i] -= mean;
      }
    } else {
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        nfb[i] = fbval;
      }
    }
    c_real h_jj = 1.0 / m * squared_norm(a_j, m);
    c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
    c_real w_j;
    if (params.count(key) > 0) {
      w_j = params[key];
    } else {
      w_j = 0.0;
    }
    c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
    c_real w_j_diff = w_j_opt - w_j;
    for (int32_t i = 0; i < m; ++i) {
      residual[i] -= w_j_diff * a_j[i];
    }
    params[key] = w_j_opt;
    if (feature_vecs.count(key) > 0) {
      delete[] feature_vecs[key];
    }
    //~ std::cout << "Pushing ";
      //~ for (int32_t i = 0; i < n; ++i) {
        //~ std::cout << key[i];
      //~ }
      //~ std::cout << "\n";
    feature_vecs[key] = nfb;
  }
  
  std::vector<std::vector<bool>> remove_idx_2;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    for (auto& fv : feature_vecs) {
      if (center) {
       c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_2.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_2) {
      //~ std::cout << "Deleting ";
      //~ for (int32_t i = 0; i < n; ++i) {
        //~ std::cout << key[i];
      //~ }
      //~ std::cout << "\n";
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_2.clear();
  }
  
  if (center) {
    c_real correction = 0.0;
    for (auto& fv : feature_vecs) {
                  //~ std::cout << "Processing: ";;
            //~ for(int32_t i = 0; i < n; ++i) {
              //~ std::cout << fv.first[i];
            //~ }
            //~ std::cout << "\n";
      auto& val = fv.second;
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        mean += val[i];
      }
      mean /= m;
      correction -= params[fv.first] * mean;
    }
    if (params.count(blocked_key) == 0) {
      params[blocked_key] = correction;
    } else {
      params[blocked_key] += correction;
    }
  }
    //~ std::cout << "------------------------\n";
  
  delete[] residual;
  delete[] a_j;
  for (auto& fv : feature_vecs) {
    delete[] fv.second;
  }
  
  Estimator* new_est = new Estimator();
  for (auto& par : params) {
    new_est->freqs.push_back(par.first);
    new_est->coefs.push_back(par.second);
  }
  return new_est;
  
}

// writes results to models and lams, so they can be used later on
void coordinate_descent_regularization_path(bool* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center, char function,
    Estimator**& models, c_real*& lams, DependencyGraph* dg, int64_t& corrtime, bool recursive) {
      //~ bas_it_ctr = 0;
  c_real* y_centered = new c_real[m];
  c_real sum = 0.0;
  for (int32_t i = 0; i < m; ++i) {
    sum += y[i];
  }
  sum /= m;
  for (int32_t i = 0; i < m; ++i) {
    y_centered[i] = y[i] - sum;
  }
  bool* freq = new bool[n]; // this is B
  bool* fB = new bool[m];
  c_real corr_max;
  clocktime start, end;
  get_time(&start); 
  if (recursive) {
    corr_max = compute_max_correlation_recursive_baseline(x, y_centered, freq, fB, m, n);
  } else {
    corr_max = compute_max_correlation_baseline(x, y_centered, freq, fB, m, n);
  }
  get_time(&end);
  corrtime += get_time_difference(&start, &end);

  lams = new c_real[n_lambda];
  models = new Estimator*[n_lambda];
  c_real lam_max = (1.0 / m) * std::fabs(corr_max);
  c_real lam_min = (c == -1 ? eps * lam_max : c);
  c_real lam_step = (std::log10(lam_max) - std::log10(lam_min)) / (n_lambda - 1);
  c_real d = std::log10(lam_max);
  Estimator* est_prev = nullptr;
  Estimator* est = nullptr;
  for (int32_t i = 0; i < n_lambda; ++i) {
    c_real lam = std::pow(10.0, d);
    est_prev = est;
    est = coordinate_descent(x, y_centered, m, n, freq, fB, steps, lam, 1e-4, patience,
        10, est_prev, center, function, dg, corrtime, recursive);
    models[i] = est;
    lams[i] = lam;
    d -= lam_step;
  }
  delete[] freq;
  delete[] fB;
  delete[] y_centered;
    //~ std::cout << "Enumerated: " << bas_it_ctr << "\n";
}

inline int32_t get_set_bits(uint64_t number) {
  return 0;
}

// ALTERNATIVE IMPLEMENTATION OF FUNCTION BELOW
// this version is only iterative, but the recursive function is added so we don't have issues with conflicting flags
BitwiseEstimator64* coordinate_descent(uint64_t* x, c_real* y, uint64_t* v, int32_t m, int32_t n, uint64_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator64* est, bool center, int64_t& corrtime, TP_TYPE* tp,
    uint64_t*& q, int32_t*& q_val, c_real*& mus, int32_t& q_size, bool recursive) {
  #if BIT_AVX
  int64_t nbits = (((n + 255) >> 8) << 2);
  #else
  int64_t nbits = ((n + 63) >> 6);
  #endif
  int64_t featbits = ((n + 63) >> 6);
  std::map<std::vector<uint64_t>, c_real> params;
  std::map<std::vector<uint64_t>, c_real*> feature_vecs;
  std::vector<uint64_t> blocked_key(featbits, 0);
  c_real* residual = new c_real[m];
  c_real* a_j = new c_real[m];
  if (!est) {
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
  } else {
    int32_t fsize = est->freqs.size();
    for (int32_t s = 0; s < fsize; ++s) {
      std::vector<uint64_t>& kfreq = est->freqs[s];
      params[kfreq] = est->coefs[s];
      c_real* mvv = new c_real[m];
      int32_t freq_sum = 0;
      for (int32_t i = 0; i < m; ++i) {
        mvv[i] = true;
        for (int32_t j = 0; j < featbits; ++j) {
          if ((x[i * nbits + j] & kfreq[j]) != kfreq[j]) {
              mvv[i] = false;
              break;
          }
        }
      }
      if (feature_vecs.count(kfreq) > 0) {
        delete[] feature_vecs[kfreq];
      }
      feature_vecs[kfreq] = mvv;
    }
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
    for (int32_t s = 0; s < fsize; ++s) {
      for (int32_t i = 0; i < m; ++i) {
        bool active = true;
        for (int32_t j = 0; j < featbits; ++j) {
          if ((est->freqs[s][j] & (~x[i * nbits + j])) != 0) {
            active = false;
            break;
          }
        }
        if (active) {
          residual[i] -= est->coefs[s];
        }
      }
    }
  }

  std::vector<std::vector<uint64_t>> remove_idx_1;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    if (feature_vecs.count(blocked_key) > 0) {
      params.erase(blocked_key);
      delete[] feature_vecs[blocked_key];
      feature_vecs.erase(blocked_key);
    }
    for (auto& fv : feature_vecs) {
      if (center) {
        c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_1.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_1) {
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_1.clear();
  }

  int32_t no_improvement_since = 0;
  for (int32_t step = 0; step < steps; ++step) {
    c_real gain;
    clocktime start, end;
    get_time(&start); 
    if (tp) {
      gain = tp->cmc_full(x, residual, freq, fB, m, n, recursive);
    } else {
      if (recursive) {
        gain = compute_max_correlation_bitwise_rec(x, residual, freq, fB, v, m, n);
      } else {
        gain = compute_max_correlation_bitwise(x, residual, freq, fB, m, n, q, q_val, mus, q_size);
      }
    }
    get_time(&end);
    corrtime += get_time_difference(&start, &end);
    //~ std::cout << gain << "\n";
    bool found_key = false;
    for (auto& par : params) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if (freq[j] != par.first[j]) {
          break;
        }
      }
      if (j == featbits) {
        ++no_improvement_since;
        found_key = true;
        break;
      }
    }
    if (!found_key) {
      no_improvement_since = 0;
    } else if (no_improvement_since > patience) {
      break;
    }
    std::vector<uint64_t> key(freq, freq + featbits);
    c_real* nfb = new c_real[m];
    if (center) {
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        mean += fbval;
        nfb[i] = fbval;
      }
      mean /= m;
      for (int32_t i = 0; i < m; ++i) {
        a_j[i] -= mean;
      }
    } else {
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        nfb[i] = fbval;
      }
    }
    c_real h_jj = 1.0 / m * squared_norm(a_j, m);
    c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
    c_real w_j;
    if (params.count(key) > 0) {
      w_j = params[key];
    } else {
      w_j = 0.0;
    }
    c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
    c_real w_j_diff = w_j_opt - w_j;
    for (int32_t i = 0; i < m; ++i) {
      residual[i] -= w_j_diff * a_j[i];
    }
    params[key] = w_j_opt;
    if (feature_vecs.count(key) > 0) {
      delete[] feature_vecs[key];
    }
    feature_vecs[key] = nfb;
  }
  
  std::vector<std::vector<uint64_t>> remove_idx_2;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    for (auto& fv : feature_vecs) {
      if (center) {
       c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_2.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_2) {
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_2.clear();
  }
  
  if (center) {
    c_real correction = 0.0;
    for (auto& fv : feature_vecs) {
      auto& val = fv.second;
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        mean += val[i];
      }
      mean /= m;
      correction -= params[fv.first] * mean;
    }
    if (params.count(blocked_key) == 0) {
      params[blocked_key] = correction;
    } else {
      params[blocked_key] += correction;
    }
  }
  
  delete[] residual;
  delete[] a_j;
  for (auto& fv : feature_vecs) {
    delete[] fv.second;
  }
  
  BitwiseEstimator64* new_est = new BitwiseEstimator64();
  for (auto& par : params) {
    new_est->freqs.push_back(par.first);
    new_est->coefs.push_back(par.second);
  }
  return new_est;
  
}

// writes results to models and lams, so they can be used later on
void coordinate_descent_regularization_path(uint64_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator64**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp,
    uint64_t*& q, int32_t*& q_val, c_real*& mus, int32_t& q_size, bool recursive) {
  #if BIT_AVX
  int64_t nbits = (((n + 255) >> 8) << 2);
  #else
  int64_t nbits = ((n + 63) >> 6);
  #endif
  int64_t featbits = ((n + 63) >> 6);
  c_real* y_centered = new c_real[m];
  c_real sum = 0.0;
  for (int32_t i = 0; i < m; ++i) {
    sum += y[i];
  }
  sum /= m;
  for (int32_t i = 0; i < m; ++i) {
    y_centered[i] = y[i] - sum;
  }
  uint64_t* freq = new uint64_t[featbits]; // this is B
  bool* fB = new bool[m];
  uint64_t* v = new uint64_t[nbits];
  c_real corr_max;
  clocktime start, end;

  get_time(&start); 
  if (tp) {
    corr_max = tp->cmc_full(x, y_centered, freq, fB, m, n, recursive);
  } else {
    if (recursive) {
      corr_max = compute_max_correlation_bitwise_rec(x, y_centered, freq, fB, v, m, n);
    } else {
      corr_max = compute_max_correlation_bitwise(x, y_centered, freq, fB, m, n, q, q_val, mus, q_size);
    }
  }
  get_time(&end);
  corrtime += get_time_difference(&start, &end);
  
  lams = new c_real[n_lambda];
  models = new BitwiseEstimator64*[n_lambda];
  c_real lam_max = (1.0 / m) * std::fabs(corr_max);
  c_real lam_min = (c == -1 ? eps * lam_max : c);
  c_real lam_step = (std::log10(lam_max) - std::log10(lam_min)) / (n_lambda - 1);
  c_real d = std::log10(lam_max);
  BitwiseEstimator64* est_prev = nullptr;
  BitwiseEstimator64* est = nullptr;
  for (int32_t i = 0; i < n_lambda; ++i) {
    c_real lam = std::pow(10.0, d);
    est_prev = est;
    est = coordinate_descent(x, y_centered, v, m, n, freq, fB, steps, lam, 1e-4,
        patience, 10, est_prev, center, corrtime, tp, q, q_val, mus, q_size, recursive);
    models[i] = est;
    lams[i] = lam;
    d -= lam_step;
  }

  delete[] freq;
  delete[] fB;
  delete[] v;
  delete[] y_centered;
}

// m samples
// n features
BitwiseEstimator64* coordinate_descent(uint64_t* x, c_real* y, uint64_t* v, int32_t m, int32_t n, uint64_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator64* est, bool center, int64_t& corrtime, TP_TYPE* tp, bool recursive) {
  #if BIT_AVX
  int64_t nbits = (((n + 255) >> 8) << 2);
  #else
  int64_t nbits = ((n + 63) >> 6);
  #endif
  int64_t featbits = ((n + 63) >> 6);
  std::map<std::vector<uint64_t>, c_real> params;
  std::map<std::vector<uint64_t>, bool*> feature_vecs;
  std::vector<uint64_t> blocked_key(featbits, 0);
  c_real* residual = new c_real[m];
  c_real* a_j = new c_real[m];
  if (!est) {
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
  } else {
    int32_t fsize = est->freqs.size();
    for (int32_t s = 0; s < fsize; ++s) {
      std::vector<uint64_t>& kfreq = est->freqs[s];
      params[kfreq] = est->coefs[s];
      bool* mvv = new bool[m];
      int32_t freq_sum = 0;
      for (int32_t i = 0; i < m; ++i) {
        mvv[i] = true;
        for (int32_t j = 0; j < featbits; ++j) {
          if ((x[i * nbits + j] & kfreq[j]) != kfreq[j]) {
              mvv[i] = false;
              break;
          }
        }
      }
      if (feature_vecs.count(kfreq) > 0) {
        delete[] feature_vecs[kfreq];
      }
      //~ std::cout << "Pushing ";
        //~ std::cout << kfreq[0];
      //~ std::cout << "\n";
      feature_vecs[kfreq] = mvv;
    }
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
    for (int32_t s = 0; s < fsize; ++s) {
      for (int32_t i = 0; i < m; ++i) {
        bool active = true;
        for (int32_t j = 0; j < featbits; ++j) {
          if ((est->freqs[s][j] & (~x[i * nbits + j])) != 0) {
            active = false;
            break;
          }
        }
        if (active) {
          residual[i] -= est->coefs[s];
        }
      }
    }
  }

  std::vector<std::vector<uint64_t>> remove_idx_1;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    if (feature_vecs.count(blocked_key) > 0) {
      //~ std::cout << "Deleting ";
        //~ std::cout << blocked_key[0];
      //~ std::cout << "\n";
      params.erase(blocked_key);
      delete[] feature_vecs[blocked_key];
      feature_vecs.erase(blocked_key);
    }
    for (auto& fv : feature_vecs) {
      if (center) {
        c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_1.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_1) {
      //~ std::cout << "Deleting ";
        //~ std::cout << key[0];
      //~ std::cout << "\n";
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_1.clear();
  }

  int32_t no_improvement_since = 0;
  for (int32_t step = 0; step < steps; ++step) {
    c_real gain;
    clocktime start, end;
    get_time(&start); 
    if (tp) {
      gain = tp->cmc_full(x, residual, freq, fB, m, n, recursive);
    } else {
      if (recursive) {
        gain = compute_max_correlation_bitwise_rec(x, residual, freq, fB, v, m, n);
      } else {
        gain = compute_max_correlation_bitwise(x, residual, freq, fB, m, n);
      }
    }
    get_time(&end);
    corrtime += get_time_difference(&start, &end);
    //~ std::cout << gain << "\t";
    //~ for (int32_t j = 0; j < n; ++j) {
        //~ std::cout << ((freq[0] >> j) & 1);
    //~ }
    //~ ////~ std::cout << " (" << freq[0] << ")\t";
    //~ std::cout << "\t";
    //~ for (int32_t j = 0; j < m; ++j) {
        //~ std::cout << fB[j];
    //~ }
    //~ std::cout << "\n";
    
    
    bool found_key = false;
    for (auto& par : params) {
      int32_t j = 0;
      for (; j < featbits; ++j) {
        if (freq[j] != par.first[j]) {
          break;
        }
      }
      if (j == featbits) {
        ++no_improvement_since;
        found_key = true;
        break;
      }
    }
    if (!found_key) {
      no_improvement_since = 0;
    } else if (no_improvement_since > patience) {
      break;
    }
    std::vector<uint64_t> key(freq, freq + featbits);
    bool* nfb = new bool[m];
    if (center) {
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        bool fbval = fB[i];
        a_j[i] = fbval;
        mean += fbval;
        nfb[i] = fbval;
      }
      mean /= m;
      for (int32_t i = 0; i < m; ++i) {
        a_j[i] -= mean;
      }
    } else {
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        nfb[i] = fbval;
      }
    }
    c_real h_jj = 1.0 / m * squared_norm(a_j, m);
    c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
    c_real w_j;
    if (params.count(key) > 0) {
      w_j = params[key];
    } else {
      w_j = 0.0;
    }
    c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
    c_real w_j_diff = w_j_opt - w_j;
    for (int32_t i = 0; i < m; ++i) {
      residual[i] -= w_j_diff * a_j[i];
    }
    params[key] = w_j_opt;
    if (feature_vecs.count(key) > 0) {
      delete[] feature_vecs[key];
    }
    //~ std::cout << "Pushing ";
        //~ std::cout << key[0];
      //~ std::cout << "\n";
    feature_vecs[key] = nfb;
  }
  
  std::vector<std::vector<uint64_t>> remove_idx_2;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    for (auto& fv : feature_vecs) {
      if (center) {
       c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_2.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_2) {
       //~ std::cout << "Deleting ";
        //~ std::cout << key[0];
      //~ std::cout << "\n";
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_2.clear();
  }
  
  if (center) {
    c_real correction = 0.0;
    for (auto& fv : feature_vecs) {
            //~ std::cout << "Processing: " << fv.first[0] << "\n";
      auto& val = fv.second;
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        mean += val[i];
      }
      mean /= m;
      correction -= params[fv.first] * mean;
    }
    if (params.count(blocked_key) == 0) {
      params[blocked_key] = correction;
    } else {
      params[blocked_key] += correction;
    }
  }
    //~ std::cout << "------------------------\n";
  
  delete[] residual;
  delete[] a_j;
  for (auto& fv : feature_vecs) {
    delete[] fv.second;
  }
  
  BitwiseEstimator64* new_est = new BitwiseEstimator64();
  for (auto& par : params) {
    new_est->freqs.push_back(par.first);
    new_est->coefs.push_back(par.second);
  }
  return new_est;

}

// writes results to models and lams, so they can be used later on
void coordinate_descent_regularization_path(uint64_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator64**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp, bool recursive) {
  //~ opt_it_ctr = 0;
  #if BIT_AVX
  int64_t nbits = (((n + 255) >> 8) << 2);
  #else
  int64_t nbits = ((n + 63) >> 6);
  #endif
  int64_t featbits = ((n + 63) >> 6);
  c_real* y_centered = new c_real[m];
  c_real sum = 0.0;
  for (int32_t i = 0; i < m; ++i) {
    sum += y[i];
  }
  sum /= m;
  for (int32_t i = 0; i < m; ++i) {
    y_centered[i] = y[i] - sum;
  }
  uint64_t* freq = new uint64_t[featbits]; // this is B
  bool* fB = new bool[m];
  uint64_t* v = new uint64_t[nbits];
  c_real corr_max;
  clocktime start, end;

  get_time(&start); 
  if (tp) {
    corr_max = tp->cmc_full(x, y_centered, freq, fB, m, n, recursive);
  } else {
    if (recursive) {
      corr_max = compute_max_correlation_bitwise_rec(x, y_centered, freq, fB, v, m, n);
    } else {
      corr_max = compute_max_correlation_bitwise(x, y_centered, freq, fB, m, n);
    }
  }
  get_time(&end);
  corrtime += get_time_difference(&start, &end);
  
  lams = new c_real[n_lambda];
  models = new BitwiseEstimator64*[n_lambda];
  c_real lam_max = (1.0 / m) * std::fabs(corr_max);
  c_real lam_min = (c == -1 ? eps * lam_max : c);
  c_real lam_step = (std::log10(lam_max) - std::log10(lam_min)) / (n_lambda - 1);
  c_real d = std::log10(lam_max);
  BitwiseEstimator64* est_prev = nullptr;
  BitwiseEstimator64* est = nullptr;
  for (int32_t i = 0; i < n_lambda; ++i) {
    c_real lam = std::pow(10.0, d);
    est_prev = est;
    est = coordinate_descent(x, y_centered, v, m, n, freq, fB, steps, lam, 1e-4,
        patience, 10, est_prev, center, corrtime, tp, recursive);
    models[i] = est;
    lams[i] = lam;
    d -= lam_step;
  }
  delete[] freq;
  delete[] fB;
  delete[] v;
  delete[] y_centered;
  //~ std::cout << "Enumerated: " << opt_it_ctr << "\n";
}

// This variant is forsaken for now, I might remove it later
BitwiseEstimator32* coordinate_descent(int32_t* x, c_real* y, int32_t m, int32_t n, int32_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator32* est, bool center, int64_t& corrtime, TP_TYPE* tp) {
  
  int64_t nbits = ((n + 31) >> 5);
  std::map<std::vector<int32_t>, c_real> params;
  std::map<std::vector<int32_t>, c_real*> feature_vecs;
  std::vector<int32_t> blocked_key(nbits, 0);
  c_real* residual = new c_real[m];
  c_real* a_j = new c_real[m];
  if (!est) {
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
  } else {
    int32_t fsize = est->freqs.size();
    for (int32_t s = 0; s < fsize; ++s) {
      std::vector<int32_t>& kfreq = est->freqs[s];
      params[kfreq] = est->coefs[s];
      c_real* mvv = new c_real[m];
      int32_t freq_sum = 0;
      for (int32_t i = 0; i < m; ++i) {
        mvv[i] = true;
        for (int32_t j = 0; j < nbits; ++j) {
          if ((x[i * nbits + j] & kfreq[j]) != kfreq[j]) {
              mvv[i] = false;
              break;
          }
        }
      }
      if (feature_vecs.count(kfreq) > 0) {
        delete[] feature_vecs[kfreq];
      }
      feature_vecs[kfreq] = mvv;
    }
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
    for (int32_t s = 0; s < fsize; ++s) {
      for (int32_t i = 0; i < m; ++i) {
        bool active = true;
        for (int32_t j = 0; j < nbits; ++j) {
          if ((est->freqs[s][j] & (~x[i * nbits + j])) != 0) {
            active = false;
            break;
          }
        }
        if (active) {
          residual[i] -= est->coefs[s];
        }
      }
    }
  }

  std::vector<std::vector<int32_t>> remove_idx_1;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    if (feature_vecs.count(blocked_key) > 0) {
      params.erase(blocked_key);
      delete[] feature_vecs[blocked_key];
      feature_vecs.erase(blocked_key);
    }
    for (auto& fv : feature_vecs) {
      if (center) {
        c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_1.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_1) {
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_1.clear();
  }

  int32_t no_improvement_since = 0;
  for (int32_t step = 0; step < steps; ++step) {
    c_real gain;
    clocktime start, end;
    get_time(&start); 
    gain = compute_max_correlation_bitwise(x, residual, freq, fB, m, n);
    get_time(&end);
    corrtime += get_time_difference(&start, &end);
    bool found_key = false;
    for (auto& par : params) {
      int32_t j = 0;
      for (; j < nbits; ++j) {
        if (freq[j] != par.first[j]) {
          break;
        }
      }
      if (j == nbits) {
        ++no_improvement_since;
        found_key = true;
        break;
      }
    }
    if (!found_key) {
      no_improvement_since = 0;
    } else if (no_improvement_since > patience) {
      break;
    }
    std::vector<int32_t> key(freq, freq + nbits);
    c_real* nfb = new c_real[m];
    if (center) {
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        mean += fbval;
        nfb[i] = fbval;
      }
      mean /= m;
      for (int32_t i = 0; i < m; ++i) {
        a_j[i] -= mean;
      }
    } else {
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        nfb[i] = fbval;
      }
    }
    c_real h_jj = 1.0 / m * squared_norm(a_j, m);
    c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
    c_real w_j;
    if (params.count(key) > 0) {
      w_j = params[key];
    } else {
      w_j = 0.0;
    }
    c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
    c_real w_j_diff = w_j_opt - w_j;
    for (int32_t i = 0; i < m; ++i) {
      residual[i] -= w_j_diff * a_j[i];
    }
    params[key] = w_j_opt;
    if (feature_vecs.count(key) > 0) {
      delete[] feature_vecs[key];
    }
    feature_vecs[key] = nfb;
  }
  
  std::vector<std::vector<int32_t>> remove_idx_2;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    for (auto& fv : feature_vecs) {
      if (center) {
       c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv.second[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv.second[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[fv.first];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[fv.first] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_2.push_back(fv.first);
      }
    }
    for (auto& key : remove_idx_2) {
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_2.clear();
  }
  
  if (center) {
    c_real correction = 0.0;
    for (auto& fv : feature_vecs) {
      auto& val = fv.second;
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        mean += val[i];
      }
      mean /= m;
      correction -= params[fv.first] * mean;
    }
    if (params.count(blocked_key) == 0) {
      params[blocked_key] = correction;
    } else {
      params[blocked_key] += correction;
    }
  }
  
  delete[] residual;
  delete[] a_j;
  for (auto& fv : feature_vecs) {
    delete[] fv.second;
  }

  BitwiseEstimator32* new_est = new BitwiseEstimator32();
  for (auto& par : params) {
    new_est->freqs.push_back(par.first);
    new_est->coefs.push_back(par.second);
  }
  return new_est;
  
}

// writes results to models and lams, so they can be used later on
void coordinate_descent_regularization_path(int32_t* x, c_real* y, int32_t m, int32_t n,
    c_real eps, int32_t n_lambda, c_real c, int32_t patience, int32_t steps, bool center,
    BitwiseEstimator32**& models, c_real*& lams, int64_t& corrtime, TP_TYPE* tp) {
  int64_t nbits = ((n + 31) >> 5);
  c_real* y_centered = new c_real[m];
  c_real sum = 0.0;
  for (int32_t i = 0; i < m; ++i) {
    sum += y[i];
  }
  sum /= m;
  for (int32_t i = 0; i < m; ++i) {
    y_centered[i] = y[i] - sum;
  }
  int32_t* freq = new int32_t[nbits]; // this is B
  bool* fB = new bool[m];
  c_real corr_max;
  clocktime start, end;
  get_time(&start); 
  corr_max = compute_max_correlation_bitwise(x, y_centered, freq, fB, m, n);
  get_time(&end);
  corrtime += get_time_difference(&start, &end);
  lams = new c_real[n_lambda];
  models = new BitwiseEstimator32*[n_lambda];
  c_real lam_max = (1.0 / m) * std::fabs(corr_max);
  c_real lam_min = (c == -1 ? eps * lam_max : c);
  c_real lam_step = (std::log10(lam_max) - std::log10(lam_min)) / (n_lambda - 1);
  c_real d = std::log10(lam_max);
  BitwiseEstimator32* est_prev = nullptr;
  BitwiseEstimator32* est = nullptr;
  for (int32_t i = 0; i < n_lambda; ++i) {
    c_real lam = std::pow(10.0, d);
    est_prev = est;
    est = coordinate_descent(x, y_centered, m, n, freq, fB, steps, lam, 1e-4,
        patience, 10, est_prev, center, corrtime, tp);
    models[i] = est;
    lams[i] = lam;
    d -= lam_step;
  }
  delete[] freq;
  delete[] fB;
  delete[] y_centered;
}

// A version of coordinate_descent in which the map elements are processed in lexicographic order
// This involves some lexicographic sorts and indirect indexing, so it might be slow
// Use only to verify if the bitwise representation gives the same results as the boolean one
// (this function also has some unfixed memory leaks)
/*BitwiseEstimator64* coordinate_descent(uint64_t* x, c_real* y, uint64_t* v, int32_t m, int32_t n, uint64_t* freq, bool* fB, int32_t steps,
    c_real c, c_real tres, int32_t patience, int32_t n_polish, BitwiseEstimator64* est, bool center, int64_t& corrtime, TP_TYPE* tp, bool recursive) {
  int64_t nbits = ((n + 63) >> 6);
  std::map<std::vector<uint64_t>, c_real> params;
  std::map<std::vector<uint64_t>, bool*> feature_vecs;
  std::vector<uint64_t> blocked_key(nbits, 0);
  c_real* residual = new c_real[m];
  c_real* a_j = new c_real[m];
  if (!est) {
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
  } else {
    int32_t fsize = est->freqs.size();
    for (int32_t s = 0; s < fsize; ++s) {
      std::vector<uint64_t>& kfreq = est->freqs[s];
      params[kfreq] = est->coefs[s];
      bool* mvv = new bool[m];
      int32_t freq_sum = 0;
      for (int32_t i = 0; i < m; ++i) {
        mvv[i] = true;
        for (int32_t j = 0; j < nbits; ++j) {
          if ((x[i * nbits + j] & kfreq[j]) != kfreq[j]) {
              mvv[i] = false;
              break;
          }
        }
      }
      if (feature_vecs.count(kfreq) > 0) {
        delete[] feature_vecs[kfreq];
      }
      feature_vecs[kfreq] = mvv;
    }
    for (int32_t i = 0; i < m; ++i) {
      residual[i] = y[i];
    }
    for (int32_t s = 0; s < fsize; ++s) {
      for (int32_t i = 0; i < m; ++i) {
        bool active = true;
        for (int32_t j = 0; j < nbits; ++j) {
          if ((est->freqs[s][j] & (~x[i * nbits + j])) != 0) {
            active = false;
            break;
          }
        }
        if (active) {
          residual[i] -= est->coefs[s];
        }
      }
    }
  }
  
  std::vector<std::pair<std::string, std::vector<uint64_t>>> words;
  for (auto& fv : feature_vecs) {
    std::string s = "";
    uint64_t ctr = 1;
    uint64_t ppppp = fv.first[0];
    for(int32_t i = 0; i < n; ++i) {
      s += std::to_string(((ppppp & ctr) >> i));
      ctr = (ctr << 1);
    }
    std::pair<std::string, std::vector<uint64_t>> wordpair(s, fv.first);
    words.push_back(wordpair);
  }
  std::sort(words.begin(), words.end());

  std::vector<std::vector<uint64_t>> remove_idx_1;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    if (feature_vecs.count(blocked_key) > 0) {
      params.erase(blocked_key);
      delete[] feature_vecs[blocked_key];
      feature_vecs.erase(blocked_key);
    }
    for (int32_t i = 0; i < words.size(); ++i) {
      if (feature_vecs.count(words[i].second)) {
      auto& fv = feature_vecs[words[i].second];
      if (center) {
        c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[words[i].second];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[words[i].second] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_1.push_back(words[i].second);
      }
    }
  }
    for (auto& key : remove_idx_1) {
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_1.clear();
  }

  int32_t no_improvement_since = 0;
  for (int32_t step = 0; step < steps; ++step) {
    c_real gain;
    clocktime start, end;
    get_time(&start); 
    if (tp) {
      gain = tp->cmc_full(x, residual, freq, fB, m, n, recursive);
    } else {
      if (recursive) {
        gain = compute_max_correlation_bitwise_rec(x, residual, freq, fB, v, m, n);
      } else {
        gain = compute_max_correlation_bitwise(x, residual, freq, fB, m, n);
      }
    }
    get_time(&end);
    corrtime += get_time_difference(&start, &end);

    bool found_key = false;
    for (int32_t i = 0; i < words.size(); ++i) {
      if (params.count(words[i].second)) {
      auto& par = params[words[i].second];
      int32_t j = 0;
      for (; j < nbits; ++j) {
        if (freq[j] != words[i].second[j]) {
          break;
        }
      }
      if (j == nbits) {
        ++no_improvement_since;
        found_key = true;
        break;
      }
    }
    }
    if (!found_key) {
      no_improvement_since = 0;
    } else if (no_improvement_since > patience) {
      break;
    }
    std::vector<uint64_t> key(freq, freq + nbits);
    bool* nfb = new bool[m];
    if (center) {
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        bool fbval = fB[i];
        a_j[i] = fbval;
        mean += fbval;
        nfb[i] = fbval;
      }
      mean /= m;
      for (int32_t i = 0; i < m; ++i) {
        a_j[i] -= mean;
      }
    } else {
      for (int32_t i = 0; i < m; ++i) {
        c_real fbval = fB[i];
        a_j[i] = fbval;
        nfb[i] = fbval;
      }
    }
    c_real h_jj = 1.0 / m * squared_norm(a_j, m);
    c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
    c_real w_j;
    bool i_haz_key = true;
    if (params.count(key) > 0) {
      w_j = params[key];
    } else {
      w_j = 0.0;
      i_haz_key = false;
    }
    c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
    c_real w_j_diff = w_j_opt - w_j;
    for (int32_t i = 0; i < m; ++i) {
      residual[i] -= w_j_diff * a_j[i];
    }
    params[key] = w_j_opt;
    if (feature_vecs.count(key) > 0) {
      delete[] feature_vecs[key];
    }
    feature_vecs[key] = nfb;
    if (!i_haz_key) {
      std::string s = "";
      uint64_t ctr = 1;
      uint64_t ppppp = key[0];
      for(int32_t i = 0; i < n; ++i) {
        s += std::to_string(((ppppp & ctr) >> i));
        ctr = (ctr << 1);
      }
      std::pair<std::string, std::vector<uint64_t>> wordpair(s, key);
      words.push_back(wordpair);
      std::sort(words.begin(), words.end());
    }
  }
  
  std::vector<std::vector<uint64_t>> remove_idx_2;
  for (int32_t polish = 0; polish < n_polish; ++polish) {
    for (int32_t i = 0; i < words.size(); ++i) {
      if (feature_vecs.count(words[i].second)) {
      auto& fv = feature_vecs[words[i].second];
      if (center) {
       c_real mean = 0.0;
        for (int32_t i = 0; i < m; ++i) {
          c_real fbval = fv[i];
          a_j[i] = fbval;
          mean += fbval;
        }
        mean /= m;
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] -= mean;
        }
      } else {
        for (int32_t i = 0; i < m; ++i) {
          a_j[i] = fv[i];
        }
      }
      c_real h_jj = 1.0 / m * squared_norm(a_j, m);
      c_real grad_j = 1.0 / m * dot_neg(a_j, residual, m);
      c_real w_j = params[words[i].second];
      c_real w_j_opt = prox(w_j - grad_j / h_jj, c / h_jj);
      c_real w_j_diff = w_j_opt - w_j;
      for (int32_t i = 0; i < m; ++i) {
        residual[i] -= w_j_diff * a_j[i];
      }
      params[words[i].second] = w_j_opt;
      if (std::fabs(w_j_opt < tres)) {
        remove_idx_2.push_back(words[i].second);
      }
    }
  }
    for (auto& key : remove_idx_2) {
      delete[] feature_vecs[key];
      feature_vecs.erase(key);
      params.erase(key);
    }
    remove_idx_2.clear();
  }
  
  if (center) {
    c_real correction = 0.0;
    for (int32_t i = 0; i < words.size(); ++i) {
      if (feature_vecs.count(words[i].second)) {
      auto& fv = feature_vecs[words[i].second];
      auto& val = fv;
      c_real mean = 0.0;
      for (int32_t i = 0; i < m; ++i) {
        mean += val[i];
      }
      mean /= m;
      correction -= params[words[i].second] * mean;
    }
  }
    if (params.count(blocked_key) == 0) {
      params[blocked_key] = correction;
    } else {
      params[blocked_key] += correction;
    }
  }
  
  delete[] residual;
  delete[] a_j;
  for (auto& fv : feature_vecs) {
    delete[] fv.second;
  }
  
  BitwiseEstimator64* new_est = new BitwiseEstimator64();
  for (auto& par : params) {
    new_est->freqs.push_back(par.first);
    new_est->coefs.push_back(par.second);
  }
  return new_est;
}
*/
