#ifndef SET_HPP
#define SET_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

using namespace std;

double sgn(double val);

double maximize_fourier_sparse(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
                             bool headstart) ;

double compute_max_positive_correlation(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB, int32_t max_cardinality) ;
                             
void traverse_rec_positive(vector<int> & freq, double & best_corr, vector<int> & best_freq, vector<int>& active, vector<vector<bool>>& x, vector<double>& y);

double compute_max_positive_correlation_recursive(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) ;


double compute_max_correlation(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) ;


double compute_max_correlation_parallel(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) ;

void traverse_rec(vector<int> & freq, double & best_corr, vector<int> & best_freq, vector<int>& active, vector<vector<bool>>& x, vector<double>& y);

double compute_max_correlation_recursive(py::array_t<bool> X, 
							 py::array_t<double> Y,
							 py::array_t<bool>& B,
							 py::array_t<bool>& fB) ;
                             
#endif
