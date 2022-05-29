#include <iostream>
#include "src/export.hpp"
#include <vector>
#include <future>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;   

class CorrExecutor {
public:

  ThreadPool* tp;
  uint64_t* x_bits;
  uint64_t* b;
  int32_t samples;
  int32_t features;
  int64_t last_max_queue;
  int64_t last_total_processed;
  double delta;
  
  CorrExecutor(py::array_t<bool> X, int32_t n_threads) {
    py::buffer_info xp = X.request();
    bool* x = (bool*)xp.ptr;
    samples = xp.shape[0];
    features = xp.shape[1];
    int32_t featbits = ((features + 63) >> 6);
    #if BIT_AVX
    int32_t padded = (((features + 255) >> 8) << 2);
    int32_t rowbits = padded;
    #else
    int32_t rowbits = featbits;
    #endif
    x_bits = new uint64_t[samples * rowbits];
    tp = nullptr;
    for (int32_t i = 0; i < samples; ++i) {
      for (int32_t j = 0; j < featbits; ++j) {
        uint64_t x_bit = 0;
          for (int32_t jj = 0; jj < 64; ++jj) {
            int32_t jjdx = j * 64 + jj;
            if (jjdx >= features) break;
            x_bit = (x_bit | ((uint64_t)x[i * features + jjdx] << jjdx));
          }
        x_bits[i * rowbits + j] = x_bit;
      }
      for (int32_t j = featbits; j < rowbits; ++j) {
        x_bits[i * rowbits + j] = 0;
      }
    }
    b = new uint64_t[featbits];
    if (n_threads == 0) {
      tp = new ThreadPool(std::thread::hardware_concurrency());
    } else {
      tp = new ThreadPool(n_threads);
    }
    delta = 0.0;
    last_max_queue = 0;
    last_total_processed = 0;
  }
    
  CorrExecutor(py::array_t<bool> X, int32_t n_threads, double delta) {
    py::buffer_info xp = X.request();
    bool* x = (bool*)xp.ptr;
    samples = xp.shape[0];
    features = xp.shape[1];
    int32_t featbits = ((features + 63) >> 6);
    #if BIT_AVX
    int32_t padded = (((features + 255) >> 8) << 2);
    int32_t rowbits = padded;
    #else
    int32_t rowbits = featbits;
    #endif
    x_bits = new uint64_t[samples * rowbits];
    tp = nullptr;
    for (int32_t i = 0; i < samples; ++i) {
      for (int32_t j = 0; j < featbits; ++j) {
        uint64_t x_bit = 0;
          for (int32_t jj = 0; jj < 64; ++jj) {
            int32_t jjdx = j * 64 + jj;
            if (jjdx >= features) break;
            x_bit = (x_bit | ((uint64_t)x[i * features + jjdx] << jjdx));
          }
        x_bits[i * rowbits + j] = x_bit;
      }
      for (int32_t j = featbits; j < rowbits; ++j) {
        x_bits[i * rowbits + j] = 0;
      }
    }
    b = new uint64_t[featbits];
    if (n_threads == 0) {
      tp = new ThreadPool(std::thread::hardware_concurrency());
    } else {
      tp = new ThreadPool(n_threads);
    }
    this->delta = delta;
    last_max_queue = 0;
    last_total_processed = 0;
  }
  
  CorrExecutor(bool* x, int32_t samples, int32_t features, int32_t n_threads) {
    this->samples = samples;
    this->features = features;
    int32_t featbits = ((features + 63) >> 6);
    #if BIT_AVX
    int32_t padded = (((features + 255) >> 8) << 2);
    int32_t rowbits = padded;
    #else
    int32_t rowbits = featbits;
    #endif
    x_bits = new uint64_t[samples * rowbits];
    tp = nullptr;
    for (int32_t i = 0; i < samples; ++i) {
      for (int32_t j = 0; j < featbits; ++j) {
        uint64_t x_bit = 0;
          for (int32_t jj = 0; jj < 64; ++jj) {
            int32_t jjdx = j * 64 + jj;
            if (jjdx >= features) break;
            x_bit = (x_bit | ((uint64_t)x[i * features + jjdx] << jjdx));
          }
        x_bits[i * rowbits + j] = x_bit;
      }
      for (int32_t j = featbits; j < rowbits; ++j) {
        x_bits[i * rowbits + j] = 0;
      }
    }
    b = new uint64_t[featbits];
    if (n_threads == 0) {
      tp = new ThreadPool(std::thread::hardware_concurrency());
    } else {
      tp = new ThreadPool(n_threads);
    }
    delta = 0.0;
    last_max_queue = 0;
    last_total_processed = 0;
  }
    
  CorrExecutor(bool* x, int32_t samples, int32_t features, int32_t n_threads, double delta) {
    this->samples = samples;
    this->features = features;
    int32_t featbits = ((features + 63) >> 6);
    #if BIT_AVX
    int32_t padded = (((features + 255) >> 8) << 2);
    int32_t rowbits = padded;
    #else
    int32_t rowbits = featbits;
    #endif
    x_bits = new uint64_t[samples * rowbits];
    tp = nullptr;
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
    b = new uint64_t[featbits];
    if (n_threads == 0) {
      tp = new ThreadPool(std::thread::hardware_concurrency());
    } else {
      tp = new ThreadPool(n_threads);
    }
    this->delta = delta;
    last_max_queue = 0;
    last_total_processed = 0;
  }
  
  ~CorrExecutor() {
    if (x_bits) {
      delete[] x_bits;
    }
    if (tp) {
      tp->stop();
      delete tp;
    }
    if (b) {
      delete[] b;
    }
    x_bits = nullptr;
    tp = nullptr;
    b = nullptr;
  }
  
  //~ c_real compute_max_correlation(c_real* y, int64_t* b, bool* fb) {
  c_real compute_max_correlation(py::array_t<double> Y,
      py::array_t<bool>& B, py::array_t<bool>& fB) {
    int32_t featbits = ((features + 63) >> 6);
    c_real* y = (c_real*)(Y.request()).ptr;
    bool* fb = (bool*)(fB.request()).ptr;
    bool* b_bool = (bool*)(B.request()).ptr;
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmc_full(x_bits, y, b, fb, samples, features,
        false, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1);
    }
    return res;
  }
  
  c_real compute_max_positive_correlation(py::array_t<double> Y,
      py::array_t<bool>& B, py::array_t<bool>& fB) {
    int32_t featbits = ((features + 63) >> 6);
    c_real* y = (c_real*)(Y.request()).ptr;
    bool* fb = (bool*)(fB.request()).ptr;
    bool* b_bool = (bool*)(B.request()).ptr;
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmcpos_full(x_bits, y, b, fb, samples, features,
        false, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1);
    }
    return res;
  }
  
  c_real compute_max_correlation_recursive(py::array_t<double> Y,
      py::array_t<bool>& B, py::array_t<bool>& fB) {
    int32_t featbits = ((features + 63) >> 6);
    c_real* y = (c_real*)(Y.request()).ptr;
    bool* fb = (bool*)(fB.request()).ptr;
    bool* b_bool = (bool*)(B.request()).ptr;
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmc_full(x_bits, y, b, fb, samples, features,
        true, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1);
    }
    return res;
  }
  
  c_real compute_max_positive_correlation_recursive(py::array_t<double> Y,
      py::array_t<bool>& B, py::array_t<bool>& fB) {
    int32_t featbits = ((features + 63) >> 6);
    c_real* y = (c_real*)(Y.request()).ptr;
    bool* fb = (bool*)(fB.request()).ptr;
    bool* b_bool = (bool*)(B.request()).ptr;
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmcpos_full(x_bits, y, b, fb, samples, features,
        true, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1); 
    }
    return res;
  }
  
    
  // TODO merge with functions above?
  c_real a_compute_max_correlation(c_real* y,
      bool* b_bool, bool* fb) {
    int32_t featbits = ((features + 63) >> 6);
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmc_full(x_bits, y, b, fb, samples, features,
        false, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1);
    }
    return res;
  }
  

  c_real a_compute_max_positive_correlation(c_real* y,
      bool* b_bool, bool* fb) {
    int32_t featbits = ((features + 63) >> 6);
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmcpos_full(x_bits, y, b, fb, samples, features,
        false, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1);
    }
    return res;
  }
  
  c_real a_compute_max_correlation_recursive(c_real* y,
      bool* b_bool, bool* fb) {
    int32_t featbits = ((features + 63) >> 6);
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmc_full(x_bits, y, b, fb, samples, features,
        true, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1);
    }
    return res;
  }
  
  c_real a_compute_max_positive_correlation_recursive(c_real* y,
      bool* b_bool, bool* fb) {
    int32_t featbits = ((features + 63) >> 6);
    for (int32_t j = 0; j < featbits; ++j) {
      uint64_t b_bit = 0;
      for (int32_t jj = 0; jj < 64; ++jj) {
        int32_t jjdx = j * 64 + jj;
        if (jjdx >= features) break;
        b_bit = (b_bit | ((uint64_t)b_bool[jjdx] << jjdx));
      }
      b[j] = b_bit;
    }
    last_max_queue = 0;
    last_total_processed = 0;
    c_real res = tp->cmcpos_full(x_bits, y, b, fb, samples, features,
        true, delta, last_max_queue, last_total_processed);
    for (int32_t j = 0; j < features; ++j) {
      int64_t idx = (j >> 6);
      uint64_t bval = b[idx];
      int64_t pos = (j & 63);
      b_bool[j] = ((bval >> pos) & 1); 
    }
    return res;
  }
  
};

class SuperCorrExecutor {
  
  std::vector<CorrExecutor*> executors;
  std::vector<int32_t> ms;
  std::vector<int32_t> ns;
  std::vector<std::future<c_real>> futures;
  int32_t n_executors;
  int32_t m;
  int32_t n;
  c_real* my_y;
  bool* my_b;
  bool* my_fb;
  
  public:
  
  std::vector<c_real> results;
  
  SuperCorrExecutor(py::array_t<bool> X_Batch, int32_t n_threads) {
    py::buffer_info xp = X_Batch.request();
    n_executors = xp.shape[0];
    m = xp.shape[1];
    n = xp.shape[2];
    bool* x = (bool*)xp.ptr;
    for (int32_t i = 0; i < n_executors; ++i) {
      executors.push_back(new CorrExecutor(x + m * n * i, m, n, n_threads));
    }
    results.resize(n_executors);
//     my_y = new c_real[n_executors * m];
//     my_b = new bool[n_executors * n];
//     my_fb = new bool[n_executors * m];
  }
  
  SuperCorrExecutor(py::array_t<bool> X_Batch, int32_t n_threads, double delta) {
    py::buffer_info xp = X_Batch.request();
    n_executors = xp.shape[0];
    m = xp.shape[1];
    n = xp.shape[2];
    bool* x = (bool*)xp.ptr;
    for (int32_t i = 0; i < n_executors; ++i) {
      executors.push_back(new CorrExecutor(x + m * n * i, m, n, n_threads, delta));
    }
    results.resize(n_executors);
//     my_y = new c_real[n_executors * m];
//     my_b = new bool[n_executors * n];
//     my_fb = new bool[n_executors * m];
  }
  
  ~SuperCorrExecutor() {
    for (int32_t i = 0; i < n_executors; ++i) {
      delete executors[i];
    }
//     delete[] my_y;
//     delete[] my_b;
//     delete[] my_fb;
  }
  
  
  void compute_max_correlation(py::array_t<double> Y_Batch,
      py::array_t<bool>& B_Batch, py::array_t<bool>& fB_Batch) {
    c_real* y = (c_real*)(Y_Batch.request()).ptr;
    bool* b = (bool*)(B_Batch.request()).ptr;
    bool* fb = (bool*)(fB_Batch.request()).ptr;
    for (int32_t i = 0; i < n_executors; ++i) {
      futures.push_back(std::async([=]{
        int64_t midx = i * m;
        int64_t nidx = i * n;
        return executors[i]->a_compute_max_correlation(y + midx, b + nidx, fb + midx);
      }));
    }
    for (int32_t i = 0; i < n_executors; ++i) {
      results[i] = futures[i].get();
    }
    futures.clear();
  }
  
  //////////////////////////////////////
  void compute_max_positive_correlation(py::array_t<double> Y_Batch,
      py::array_t<bool>& B_Batch, py::array_t<bool>& fB_Batch) {
    c_real* y = (c_real*)(Y_Batch.request()).ptr;
    bool* b = (bool*)(B_Batch.request()).ptr;
    bool* fb = (bool*)(fB_Batch.request()).ptr;
    //std::memcpy(my_y, y, n_executors * m * sizeof(c_real));
    //std::memcpy(my_b, b, n_executors * n * sizeof(bool));
    //std::memcpy(my_fb, fb, n_executors * m * sizeof(bool));
    //py::gil_scoped_release release;
    for (int32_t i = 0; i < n_executors; ++i) {
      futures.push_back(std::async([=]{
        int64_t midx = i * m;
        int64_t nidx = i * n;
        return executors[i]->a_compute_max_positive_correlation(y + midx, b + nidx, fb + midx);
      }));
    }
    for (int32_t i = 0; i < n_executors; ++i) {
      results[i] = futures[i].get();
    }
    futures.clear();
    //py::gil_scoped_acquire acquire;
    //std::memcpy(y, my_y, n_executors * m * sizeof(c_real));
    //std::memcpy(b, my_b, n_executors * n * sizeof(bool));
    //std::memcpy(fb, my_fb, n_executors * m * sizeof(bool));
  }
  
  void compute_max_correlation_recursive(py::array_t<double> Y_Batch,
      py::array_t<bool>& B_Batch, py::array_t<bool>& fB_Batch) {
    c_real* y = (c_real*)(Y_Batch.request()).ptr;
    bool* b = (bool*)(B_Batch.request()).ptr;
    bool* fb = (bool*)(fB_Batch.request()).ptr;
    for (int32_t i = 0; i < n_executors; ++i) {
      futures.push_back(std::async([=]{
        int64_t midx = i * m;
        int64_t nidx = i * n;
        return executors[i]->a_compute_max_correlation_recursive(y + midx, b + nidx, fb + midx);
      }));
    }
    for (int32_t i = 0; i < n_executors; ++i) {
      results[i] = futures[i].get();
    }
    futures.clear();
  }
  
  void compute_max_positive_correlation_recursive(py::array_t<double> Y_Batch,
      py::array_t<bool>& B_Batch, py::array_t<bool>& fB_Batch) {
    c_real* y = (c_real*)(Y_Batch.request()).ptr;
    bool* b = (bool*)(B_Batch.request()).ptr;
    bool* fb = (bool*)(fB_Batch.request()).ptr;
    for (int32_t i = 0; i < n_executors; ++i) {
      futures.push_back(std::async([=]{
        int64_t midx = i * m;
        int64_t nidx = i * n;
        return executors[i]->a_compute_max_positive_correlation_recursive(y + midx, b + nidx, fb + midx);
      }));
    }
    for (int32_t i = 0; i < n_executors; ++i) {
      results[i] = futures[i].get();
    }
    futures.clear();
  }
  
};

struct cpp_est {
  std::vector<std::vector<bool>> freqs;
  std::vector<double> coefs;
  cpp_est(std::vector<std::vector<bool>>& f, std::vector<double>& c): freqs(f), coefs(c) {}
};

std::vector<cpp_est> cpp_fit(py::array_t<bool> X, py::array_t<double> Y,
             double C, int32_t Nlams, double Steps, bool Recursive) {
  py::buffer_info xp = X.request();
  py::buffer_info yp = Y.request();
  int32_t samples = xp.shape[0];
  int32_t features = xp.shape[1];
  bool* x = (bool*)xp.ptr;
  c_real* y = new c_real[samples];
  for (int32_t i = 0; i < samples; ++i) {
    y[i] = ((double*)yp.ptr)[i];
  }
  std::vector<std::vector<std::vector<bool>>> freqs;
  std::vector<std::vector<double>> coefs;
  c_fit(x, y, samples, features, freqs, coefs, C, Nlams, Steps, Recursive);
  std::vector<cpp_est> estimators;
  for (int32_t i = 0; i < Nlams; ++i) {
    cpp_est e(freqs[i], coefs[i]);
    estimators.push_back(e);
  }
  delete[] y;
  return estimators;
}

PYBIND11_MODULE(_fit, m) {
	m.def("cpp_fit", &cpp_fit, "",
        py::arg("X"), py::arg("Y"), py::arg("C") = 1,
        py::arg("Nlams") = 20, py::arg("Steps") = 10000,
        py::arg("Recursive") = false
  );
  
  py::class_<cpp_est>(m, "cpp_est")
    .def_readwrite("freqs", &cpp_est::freqs)
    .def_readwrite("coefs", &cpp_est::coefs);
    
  py::class_<CorrExecutor>(m, "CorrExecutor")
    .def(py::init<py::array_t<bool>, int32_t, double>())
    .def(py::init<py::array_t<bool>, int32_t>())
    .def_readwrite("last_max_queue", &CorrExecutor::last_max_queue)
    .def_readwrite("last_total_processed", &CorrExecutor::last_total_processed)
    .def_readwrite("delta", &CorrExecutor::delta)
    .def("compute_max_correlation", &CorrExecutor::compute_max_correlation)
    .def("compute_max_positive_correlation", &CorrExecutor::compute_max_positive_correlation)
    .def("compute_max_correlation_recursive", &CorrExecutor::compute_max_correlation_recursive)
    .def("compute_max_positive_correlation_recursive", &CorrExecutor::compute_max_positive_correlation_recursive);
    
  py::class_<SuperCorrExecutor>(m, "SuperCorrExecutor")
    .def(py::init<py::array_t<bool>, int32_t, double>())
    .def(py::init<py::array_t<bool>, int32_t>())
    .def("compute_max_correlation", &SuperCorrExecutor::compute_max_correlation)
    .def("compute_max_positive_correlation", &SuperCorrExecutor::compute_max_positive_correlation)
    .def("compute_max_correlation_recursive", &SuperCorrExecutor::compute_max_correlation_recursive)
    .def("compute_max_positive_correlation_recursive", &SuperCorrExecutor::compute_max_positive_correlation_recursive)
    .def_readwrite("results", &SuperCorrExecutor::results);
  
}
