#include <iostream>
#include "powerset_enumeration.hpp"
#include "multiset_enumeration.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(_powerset_enum, m) {
    m.def("maximize_fourier_sparse", &maximize_fourier_sparse, "find maximimum of dsft3 sparse function");
	m.def("compute_max_positive_correlation_recursive", &compute_max_positive_correlation_recursive, "find the maximally positively correlated feature vector");// not all improvements implemented
	m.def("compute_max_positive_correlation", &compute_max_positive_correlation, "find the maximally positively correlated feature vector");
    m.def("compute_max_correlation", &compute_max_correlation, "find the maximally correlated feature vector");
	m.def("compute_max_correlation_parallel", &compute_max_correlation_parallel, "find the maximally correlated feature vector");// not all improvements implemented
	m.def("compute_max_correlation_recursive", &compute_max_correlation_recursive, "find the maximally correlated feature vector");// not all improvements implemented
    m.def("ms_compute_max_correlation", &ms_compute_max_correlation, "find maximally correlated feature vector for the multiset basis",
          py::arg("X"),py::arg("Y"),py::arg("freq"),py::arg("fB"),py::arg("M"));
}
