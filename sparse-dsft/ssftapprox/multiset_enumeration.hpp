#ifndef MULTISET_HPP
#define MULTISET_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

double ms_compute_max_correlation(py::array_t<int> X, 
							 py::array_t<double> Y,
							 py::array_t<int>& B,
							 py::array_t<bool>& fB,
                             py::array_t<int> M);

#endif
