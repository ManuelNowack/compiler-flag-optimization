#include <iostream>
#include <fstream>

#include "common.hpp"
#include "baseline.hpp"
#include "threadpool.hpp"
#include "lasso.hpp"
#include "correlation.hpp"

int32_t tries = 10;
int32_t nn_lams = 20;

// Read from binary file - currently, has to be preprocessed by python first.
// File format:
// 1 * int32_t for #samples, 1 * int32_t for #features,
// #samples * #features of bools for X, #features of c_reals for Y
void read_from_file(int32_t& s, int32_t& f, bool*& x, c_real*& y, const std::string& filename);

int main(int argc, char** argv) {
  
  if (argc < 5) {
    err("Usage: ./estimator <data file name> <c> <steps> <threads>");
  }

  std::string filename = argv[1];
  c_real c = std::stod(argv[2]);
  int32_t steps = std::stoi(argv[3]);
  int32_t no_threads = std::stoi(argv[4]);
  no_threads = (no_threads == 0)
      ? std::thread::hardware_concurrency()
      : no_threads;
  bool* x;
  c_real* y;
  int32_t samples, features;
  read_from_file(samples, features, x, y, filename);
  
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
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t x_bit = 0;
        for (int32_t jj = 0; jj < 64; ++jj) {
          int32_t jjdx = j * 64 + jj;
          if (jjdx >= features) break;
          x_bit = (x_bit | ((uint64_t)x[i * features + jjdx] << jj));
        }
      x_bits[i * rowbits + j] = x_bit;
    }
    for (int32_t j = featbits; j < rowbits; ++j) {
      x_bits[i * rowbits + j] = 0;
    }
  }
  
  //~ std::cout << "***\n";
  //~ for (int32_t i = 0; i < samples; ++i) {
    //~ for (int32_t j = 0; j < features; ++j) {
      //~ std::cout << x[i * features + j];
    //~ }
    //~ std::cout << " | ";
    //~ for (int32_t j = 0; j < rowbits; ++j) {
      //~ for (int32_t k = 0; k < 64; ++k) {
        //~ if (j * 64 + k >= features) {
          //~ break;
        //~ }
        //~ std::cout << ((x_bits[i * rowbits + j] >> k) & 1);
      //~ }
    //~ }
    //~ std::cout << "\n";
  //~ }
  
  uint64_t* v = new uint64_t[rowbits];
  std::memset(v, 0, sizeof(uint64_t) * rowbits);
  int32_t top1 = -1;
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
  //~ std::cout << featbits << "\n";
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
  int32_t* v = new int32_t[featbits];
  std::memset(v, 0, sizeof(int32_t) * featbits);
  int32_t top1 = -1;
  c_real max_corr = 0.0;
  
  #endif
  
  clocktime s1, e1, s2, e2, s3, e3, s4, e4;
  
  //normal
  #if !BITWISE
  
  //~ std::cout << "NLAMS: " << nn_lams << "\n";
  
  for (int64_t i = 0; i < samples; ++i) {
    int64_t featbits = (features + 63) / 64;
    for (int64_t j = 0; j < featbits; ++j) {
      int64_t start = i * features + j * 64;
      int64_t end = i * features + std::min((j + 1) * 64, (int64_t)features);
      std::reverse(x + start, x + end);
    }
  }
  
  get_time(&s4);
  int64_t corr_time = 0;
  for (int32_t t = 0; t < tries; ++t) {
    //~ std::cout << t << "\n";
    int32_t nlams = nn_lams;
    Estimator** models = nullptr;
    c_real* lams = nullptr;
    coordinate_descent_regularization_path(x, y, samples, features, 1e-3, nlams,
        c, 3, steps, true, 5, models, lams, nullptr, corr_time, !ITERATIVE);
    
    //~ for (int32_t nl = nlams - 1; nl < nlams; ++nl) {
      //~ std::cout << "LAM: " << lams[nl] << "\n";
      //~ for (int32_t i = 0; i < models[nl]->coefs.size(); ++i) {
        //~ std::cout << "(";
        //~ for (int32_t j = 0; j < models[nl]->freqs[i].size(); ++j) {
          //~ std::cout << models[nl]->freqs[i][j];
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
  }
  get_time(&e4);
  
  //bitwise
  #else 
  
  #if PARALLEL
  //~ TP_TYPE tp(std::thread::hardware_concurrency());
  TP_TYPE tp(no_threads);
  #endif
  get_time(&s4);
  int64_t corr_time = 0;
  for (int32_t t = 0; t < tries; ++t) {
    //~ std::cout << t << "\n";
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
        c, 3, steps, true, models, lams, corr_time, &tp, !ITERATIVE);
    #else
    coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
        c, 3, steps, true, models, lams, corr_time, nullptr, q, q_val, mus, q_size, !ITERATIVE);    
    #endif
    #else
    #if PARALLEL
    coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
        c, 3, steps, true, models, lams, corr_time, &tp, !ITERATIVE);
    #else
    coordinate_descent_regularization_path(x_bits, y, samples, features, 1e-3, nlams,
        c, 3, steps, true, models, lams, corr_time, nullptr, !ITERATIVE);
    #endif
    #endif
    
    #if _64_BIT
    int32_t offset = 64;
    #else
    int32_t offset = 32;
    #endif

    //~ for (int32_t nl = nlams - 1; nl < nlams; ++nl) {
      //~ std::cout << "LAM: " << lams[nl] << "\n";
      //~ for (int32_t i = 0; i < models[nl]->coefs.size(); ++i) {
        //~ for (int32_t j = 0; j < models[nl]->freqs[i].size(); ++j) {
          //~ uint64_t f = models[nl]->freqs[i][j];
          //~ uint64_t ctr = 1;
          //~ for(int32_t k = 0; k < ((j == models[nl]->freqs[i].size() - 1) ? features % offset : offset); ++k) {
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
  }
  get_time(&e4);

  #if PARALLEL
  tp.stop();
  #endif
  
  #endif
  
  
  std::cout << "TOT;\t\t t = " << get_time_difference(&s4, &e4) / tries << " ns\n";
  std::cout << "MAXC;\t\t t = " << corr_time / tries << " ns\n";
  
  #if BITWISE
  delete[] v;
  delete[] x_bits;
  #endif
  delete[] x;
  delete[] y;
  #if CYCLIC_BUFFER
  #if !PARALLEL
  delete[] q;
  delete[] q_val;
  delete[] mus;
  #endif
  #endif
  
  //~ tp.stop();
  return 0;
}

void read_from_file(int32_t& s, int32_t& f, bool*& x, c_real*& y, const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if(!file.is_open()) {
    err("Failed to read file " + filename);
  } else {
    file.read((char*)&s, sizeof(int32_t));
    file.read((char*)&f, sizeof(int32_t));
    x = new bool[s * f];
    y = new c_real[s];
    double* yd = new double[s];
    file.read((char*)x, sizeof(bool) * s * f); 
    file.read((char*)yd, sizeof(double) * s);
    for (int32_t i = 0; i < s; ++i) {
      y[i] = (c_real)yd[i];
    }
    delete[] yd;
    //~ for (int32_t i = 0; i < s; ++i) {
      //~ std::cout << i << ": ";
      //~ for (int32_t j = 0; j < f; ++j) {
        //~ std::cout << x[i * f + j];
      //~ }
      //~ std::cout << " | " << y[i] << "\n";
    //~ }
    file.close();
  }
}
