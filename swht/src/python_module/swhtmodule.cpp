//=========================
// Python module interface
//=========================


#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "swht.h"

#include <cstring>
#include <limits>
#include <iostream>
#include <sstream>


/** Unsigned long converter
 * Parses a Python argument and turns it into an unsigned long if it is safe to
 * do so.
 * Success: dest assigned -> returns false
 * Failure: sets pyerr    -> returns true
 */
static bool unsignedlong_converter(const char *param, PyObject *obj, unsigned long &dest) {
    std::stringstream error_builder;
    error_builder << param;
    if (!PyNumber_Check(obj)) {
        error_builder << " must be an integer number (" << obj->ob_type->tp_name << " given).";
        PyErr_SetString(PyExc_ValueError, error_builder.str().c_str());
        return true;
    }
    if (PyLong_Check(obj)) {
        dest = PyLong_AsUnsignedLong(obj);
        if (PyErr_Occurred() != NULL) {
            error_builder << " is not safely convertible to unsigned long.";
            PyErr_SetString(PyExc_OverflowError, error_builder.str().c_str());
            return true;
        }
        return false;
    }
    PyObject *rounded = PyNumber_Long(obj);
    if (PyObject_RichCompareBool(rounded, obj, Py_NE)) {
        error_builder << " must be an integer (non-integer number given).";
        PyErr_SetString(PyExc_ValueError, error_builder.str().c_str());
        return true;
    }
    dest = PyLong_AsUnsignedLong(rounded);
    Py_DECREF(rounded);
    if (PyErr_Occurred() != NULL) {
        error_builder << " is not safely convertible to unsigned long.";
        PyErr_SetString(PyExc_OverflowError, error_builder.str().c_str());
        return true;
    }
    return false;
}


/** Double converter
 * Parses a Python argument and turns it into an double if it is safe to do so.
 * Success: dest assigned -> returns false
 * Failure: sets pyerr    -> returns true
 */
static bool double_converter(const char *param, PyObject *obj, double &dest) {
    std::stringstream error_builder;
    error_builder << param;
    if (!PyNumber_Check(obj)) {
        error_builder << " must be a number (" << obj->ob_type->tp_name << " given).";
        PyErr_SetString(PyExc_ValueError, error_builder.str().c_str());
        return true;
    }
    if (PyFloat_Check(obj)) {
        dest = PyFloat_AsDouble(obj);
        if (PyErr_Occurred() != NULL) {
            error_builder << " is not safely convertible to double.";
            PyErr_SetString(PyExc_OverflowError, error_builder.str().c_str());
            return true;
        }
        return false;
    }
    PyObject *floated = PyNumber_Float(obj);
    if (PyObject_RichCompareBool(floated, obj, Py_NE)) {
        error_builder << " must be a rational number.";
        PyErr_SetString(PyExc_ValueError, error_builder.str().c_str());
        return true;
    }
    dest = PyFloat_AsDouble(floated);
    Py_DECREF(floated);
    return false;
}


/** SWHT_basic Python interface
 * Defines the Python interface and argument parsing process for the basic
 * SWHT function.
 */
static PyObject *swht_swht(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void) self;
    
    // Arguments
    PyObject *signal            = nullptr;
    char *raw_cs_algorithm      = nullptr;
    PyObject *py_n              = nullptr;
    PyObject *py_K              = nullptr;
    PyObject *py_C              = nullptr;
    PyObject *py_ratio          = nullptr;
    PyObject *py_robust_iter    = nullptr;
    PyObject *py_cs_bins        = nullptr;
    PyObject *py_cs_iterations  = nullptr;
    PyObject *py_cs_ratio       = nullptr;
    PyObject *py_degree         = nullptr;

    // Arguments keyword names
    static const char *kwlist[12] = {
        "signal",
        "cs_algorithm",
        "n",
        "K",
        "C",
        "ratio",
        "robust_iterations",
        "cs_bins",
        "cs_iterations",
        "cs_ratio",
        "degree",
        NULL
    };
    
    // Parsing
    if (
        !PyArg_ParseTupleAndKeywords(
            args, kwargs, "OsOO|$OOOOOOO", (char **)kwlist,
            &signal,
            &raw_cs_algorithm,
            &py_n,
            &py_K,
            &py_C,
            &py_ratio,
            &py_robust_iter,
            &py_cs_bins,
            &py_cs_iterations,
            &py_cs_ratio,
            &py_degree
        )
    ) return NULL;

    // Argument processing: cs_algorithm
    std::string cs_algorithm(raw_cs_algorithm);

    // Argument processing: n
    unsigned long n;
    if (unsignedlong_converter("n", py_n, n)) return NULL;
    
    // Argument processing: K
    unsigned long K;
    if (unsignedlong_converter("K", py_K, K)) return NULL;
    
    // Argument processing: C
    double C = 1.3;
    if (py_C != NULL) {
        if (double_converter("C", py_C, C)) return NULL;
    }
    
    // Argument processing: ratio
    double ratio = 1.4;
    if (py_ratio != NULL) {
        if (double_converter("ratio", py_ratio, ratio)) return NULL;
    }

    // Argument processing: robust_iter
    unsigned long robust_iter = 1ul;
    if (py_robust_iter != NULL) {
        if (unsignedlong_converter("robust_iterations", py_robust_iter, robust_iter))
            return NULL;
    }

    // Argument processing: cs_bins
    unsigned long cs_bins = 0ul;
    if (py_cs_bins != NULL) {
        if (unsignedlong_converter("cs_bins", py_cs_bins, cs_bins))
            return NULL;
    }

    // Argument processing: cs_iterations
    unsigned long cs_iterations = 1ul;
    if (py_cs_iterations != NULL) {
        if (unsignedlong_converter("cs_iterations", py_cs_iterations, cs_iterations))
            return NULL;
    }

    // Argument processing: cs_ratio
    double cs_ratio = 2.0;
    if (py_cs_ratio != NULL) {
        if (double_converter("cs_ratio", py_cs_ratio, cs_ratio))
            return NULL;
    }

    // Argument processing: degree (optional)
    unsigned long degree = 0ul;
    if (py_degree != NULL) {
        if (unsignedlong_converter("degree", py_degree, degree))
            return NULL;
    }

    // Call C/C++ function
    frequency_map out;
    try {
        out = swht(signal, cs_algorithm, n, K, robust_iter, C, ratio,
            cs_bins, cs_iterations, cs_ratio, degree);
    } 
    catch(const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    // Return result
    PyObject *result = PyDict_New();
    for (auto &&item: out) {
        PyObject *bits_key = PyTuple_New(item.first.size());
        for (size_t i = 0; i < item.first.size(); i++) {
            PyTuple_SetItem(bits_key, i, PyLong_FromLong(item.first[i]));
        }
        PyObject *py_value = PyFloat_FromDouble(item.second);
        PyDict_SetItem(result, bits_key, py_value);
        Py_DECREF(py_value);
        Py_DECREF(bits_key);
    }
    return result;
}


PyDoc_STRVAR(swht_func_doc,
    "swht(signal, cs_algorithm, n, K, C, ratio, /, robust_iterations=1, cs_bins=None, cs_iterations=1, cs_ratio=2.0, degree=None)\n"
    "--\n\n"
    "Performs a non-robust Sparse Walsh-Hadamard Transform of the input signal.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "signal: Python callable\n"
    "\tsignal to be transformed\n"
    "cs_algorithm: str\n"
    "\tcompressive sensing algorithm to be used for frequency retrieval (NAIVE, RANDOM_BINNING or REED_SOLOMON)\n"
    "n: int\n"
    "\tindex size of the input signal\n"
    "K: int\n"
    "\tsignal sparsity\n"
    "C: float (optional)\n"
    "\tbucket constant\n"
    "ratio: float (optional)\n"
    "\tbuckets reduction ratio\n"
    "robust_iterations: int (optional)\n"
    "\tnumber of robust iterations to perform (1 = default non-robust)\n"
    "cs_bins: int (required - random binning)\n"
    "\tnumber of bins to hash into before binary search\n"
    "cs_iterations: int (optional - random binning)\n"
    "\tnumber of hashing and binary search rounds to perform\n"
    "cs_ratio: int (optional - random binning)\n"
    "\treduction ratio of the random binning bins across iterations\n"
    "degree: int (required - Reed-Solomon)\n"
    "\tdegree of the signal\n"
    "\n"
    "Returns\n"
    "-------\n"
    "A dictionary containing the frequency-amplitude mapping of the transform."
);
/** Module methods
 * Registry of module methods.
 */
static PyMethodDef SwhtMethods[] = {
    {"swht",  (PyCFunction)(void(*)(void))swht_swht, METH_VARARGS | METH_KEYWORDS, swht_func_doc},
    {NULL, NULL, 0, NULL}
};


/** Module properties
 * Module properties structure definition.
 */
PyDoc_STRVAR(swht_doc,
    "SWHT package\n"
    "--\n\n"
    "This package offers the swht function for fast sparse Walsh-Hadamard transforms.\n"
    "It also defines a overridable binary_signal class and the NAIVE, RANDOM_BINNING\n"
    "and REED_SOLOMON constants."
);
static struct PyModuleDef swhtmodule = {
    PyModuleDef_HEAD_INIT,
    "swht.swht",         // module name
    swht_doc,       // module documentation
    -1,             // state size per interpreter (-1 = global state)
    SwhtMethods,    // module methods
    NULL,           // m_slots
    NULL,           // m_traverse
    NULL,           // m_clear
    NULL            // m_free
};

/** Module initialization
 * Generates and returns module pointer when calling 'import'.
 */
PyMODINIT_FUNC PyInit_swht(void) {

    // Attempt to initialize the module
    PyObject *py_module = PyModule_Create(&swhtmodule);
    if (py_module == NULL)
        return NULL;
    
    // Add NAIVE constant to the module
    PyObject *NAIVE_cst = PyUnicode_FromString("naive");
    if (PyModule_AddObject(py_module, "NAIVE", NAIVE_cst) < 0) {
        Py_DECREF(NAIVE_cst);
        Py_DECREF(py_module);
        return NULL;
    }

    // Add RANDOM_BINNING constant to the module
    PyObject *RANDOM_BINNING_cst = PyUnicode_FromString("random binning");
    if (PyModule_AddObject(py_module, "RANDOM_BINNING", RANDOM_BINNING_cst) < 0) {
        Py_DECREF(RANDOM_BINNING_cst);
        Py_DECREF(py_module);
        return NULL;
    }

    // Add REED_SOLOMON constant to the module
    PyObject *REED_SOLOMON_cst = PyUnicode_FromString("reed-solomon");
    if (PyModule_AddObject(py_module, "REED_SOLOMON", REED_SOLOMON_cst) < 0) {
        Py_DECREF(REED_SOLOMON_cst);
        Py_DECREF(py_module);
        return NULL;
    }

    return py_module;
}
