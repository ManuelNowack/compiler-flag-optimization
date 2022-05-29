#include "export.hpp"

void test_me() {
  std::cout << "Me tested\n";
}

void c_fit(bool* x, double* yd, int32_t samples, int32_t features,
           std::vector<std::vector<std::vector<bool>>>& freqs,
           std::vector<std::vector<double>>& coefs,
           double C, int32_t Nlams, double Steps, bool Recursive) {

  int32_t nn_lams = Nlams;
  c_real c = C;
  int32_t steps = Steps;

  c_real* y = new c_real[samples];
  for (int32_t i = 0; i < samples; ++i) {
    y[i] = yd[i];
  }

  #if BITWISE
  
  #if _64_BIT
  int64_t featbits = ((features + 63) >> 6);
  #if BIT_AVX
  int64_t padded = (((features + 255) >> 8) << 2);
  int64_t rowbits = padded;
  #else
  int64_t rowbits = featbits;
  #endif
  uint64_t* x_bits = new uint64_t[samples * rowbits];
  for (int32_t i = 0; i < samples; ++i) {
    for (int64_t j = 0; j < featbits; ++j) {
      uint64_t x_bit = 0;
        for (int32_t jj = 0; jj < 64; ++jj) {
          int64_t jjdx = j * 64 + jj;
          if (jjdx >= features) break;
          x_bit = (x_bit | ((uint64_t)x[i * features + jjdx] << jj));
        }
      x_bits[i * rowbits + j] = x_bit;
    }
    for (int64_t j = featbits; j < rowbits; ++j) {
      x_bits[i * rowbits + j] = 0;
    }
  }
  c_real max_corr = 0.0;
  
  #if CYCLIC_BUFFER
  #if !PARALLEL
  int32_t q_size = INIT_CYCLIC_SIZE;
  uint64_t* q = new uint64_t[q_size * rowbits];
  int32_t* q_val = new int32_t[q_size];
  c_real* mus = new c_real[q_size];
  #endif
  #endif
  
  #else
  int32_t featbits = ((features + 31) >> 5);
  int32_t* x_bits = new int32_t[samples * featbits];
  for (int32_t i = 0; i < samples; ++i) {
    for (int32_t j = 0; j < featbits; ++j) {
      int32_t x_bit = 0;
        for (int32_t jj = 0; jj < 32; ++jj) {
          int32_t jjdx = j * 32 + jj;
          if (jjdx >= features) break;
          x_bit = (x_bit | ((int32_t)x[i * features + jjdx] << jj));
        }
      x_bits[i * featbits + j] = x_bit;
    }
  }
  c_real max_corr = 0.0;
  
  #endif
  #endif
  
  freqs.clear();
  coefs.clear();
  int64_t corr_time = 0;

  // normal
  #if !BITWISE
  
  int32_t nlams = nn_lams;
  Estimator** models = nullptr;
  c_real* lams = nullptr;
  coordinate_descent_regularization_path(x, y, samples, features, 1e-3, nlams,
      c, 3, steps, true, 5, models, lams, nullptr, corr_time, Recursive);
  
  for (int32_t nl = 0; nl < nlams; ++nl) {
    std::vector<std::vector<bool>> v_freqs;
    std::vector<double> v_coefs;
    for (int32_t i = 0; i < models[nl]->coefs.size(); ++i) {
      std::vector<bool> fr(features, 0);
      for (int32_t j = 0; j < models[nl]->freqs[i].size(); ++j) {
        fr[j] = models[nl]->freqs[i][j];
      }
      v_freqs.push_back(fr);
      v_coefs.push_back(models[nl]->coefs[i]);
    }
    freqs.push_back(v_freqs);
    coefs.push_back(v_coefs);
  }

  for (int32_t nl = 0; nl < nlams; ++nl) {
      delete models[nl];
  }
  delete[] models;
  delete[] lams;

  //bitwise
  #else
  
  #if PARALLEL
  TP_TYPE tp(std::thread::hardware_concurrency());
  #endif

  int32_t nlams = nn_lams;
  
  #if _64_BIT
  BitwiseEstimator64** models = nullptr;
  #else
  BitwiseEstimator32** models = nullptr;
  #endif
  
  c_real* lams = nullptr;
  
  #if CYCLIC_BUFFER
  #if PARALLEL
  coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
      c, 3, steps, true, models, lams, corr_time, &tp, Recursive);
  #else
  coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
      c, 3, steps, true, models, lams, corr_time, nullptr, q, q_val, mus, q_size, Recursive);    
  #endif
  #else
  #if PARALLEL
  coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
      c, 3, steps, true, models, lams, corr_time, &tp, Recursive);
  #else
  coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
      c, 3, steps, true, models, lams, corr_time, nullptr, Recursive);
  #endif
  #endif
  
  #if _64_BIT
  int32_t offset = 64;
  #else
  int32_t offset = 32;
  #endif

  for (int32_t nl = 0; nl < nlams; ++nl) {
    std::vector<std::vector<bool>> v_freqs;
    std::vector<double> v_coefs;
    for (int32_t i = 0; i < models[nl]->coefs.size(); ++i) {
      std::vector<bool> fr(features, 0);
      for (int32_t j = 0; j < models[nl]->freqs[i].size(); ++j) {
        uint64_t f = models[nl]->freqs[i][j];
        uint64_t ctr = 1;
        for(int32_t k = 0; k < ((j == models[nl]->freqs[i].size() - 1) ? features % offset : features); ++k) {
          fr[j * offset + k] = ((f & ctr) >> k);
          ctr = (ctr << 1);
        }
      }
      v_freqs.push_back(fr);
      v_coefs.push_back(models[nl]->coefs[i]);
    }
    freqs.push_back(v_freqs);
    coefs.push_back(v_coefs);
  }

  //~ for (int32_t nl = nlams - 1; nl < nlams; ++nl) {
    //~ std::cout << "LAM: " << lams[nl] << "\n";
    //~ for (int32_t i = 0; i < models[nl]->coefs.size(); ++i) {
      //~ for (int32_t j = 0; j < models[nl]->freqs[i].size(); ++j) {
        //~ uint64_t f = models[nl]->freqs[i][j];
        //~ uint64_t ctr = 1;
        //~ for(int32_t k = 0; k < ((j == models[nl]->freqs[i].size() - 1) ? features % offset : features); ++k) {
          //~ std::cout << ((f & ctr) >> k);
          //~ ctr = (ctr << 1);
        //~ }
      //~ }
      //~ std::cout << " ( ";
      //~ for (int32_t j = 0; j < models[nl]->freqs[i].size(); ++j) {
        //~ std::cout << models[nl]->freqs[i][j] << " ";
      //~ }
      //~ std::cout << ")\t" << models[nl]->coefs[i] << "\n";
    //~ }
    //~ std::cout << "-----------------------------------\n";
  //~ }

  for (int32_t nl = 0; nl < nlams; ++nl) {
    delete models[nl];
  }
  delete[] models;
  delete[] lams;

  #if PARALLEL
  tp.stop();
  #endif
  #endif
  
  #if BITWISE
  delete[] x_bits;
  #endif

  #if CYCLIC_BUFFER
  #if !PARALLEL
  delete[] q;
  delete[] q_val;
  delete[] mus;
  #endif
  #endif

}
