# Adding compressive sensing providers

This is a short manual explaining how to implement a new compressive sensing technique and use it with the library.

## Adding the provider
The CS provider must be put in the [src/cs_providers](src/cs_providers) folder, in a class implementing the `finite_field_cs` interface (provided in [finite_field_cs.h](src/cs_providers/finite_field_cs.h)) like so:
```c++
#include "finite_field_cs.h"

struct new_cs: public finite_field_cs {

    /** Update
     * Implement if the provider must be updated at each iteration.
     */
    void update();

    /** Measurement matrix
     * Returns the required vector from the measurement matrix.
     * 
     * @param i: index of the measurement matrix row to return
     * @return: a row of the measurement matrix
     */
    const bits &measurement_matrix_row(unsigned long i);

    /** Frequency recovery
     * The frequency recovery algorithm from a measurement must
     * be implemented here.
     * 
     * @param measurement: bit sequence returned as-is
     */
    void recover_frequency(bits &measurement);
};
```

## Linking and compiling the provider
**Step 1:** In [src/cs_providers/CMakeLists.txt](src/cs_providers/CMakeLists.txt), add your provider files like so (supposing a header file with implementation file):
```cmake
# Gather files
set(CS_PROVIDERS_FILES
    ...
    ${CS_PROVIDERS_PATH}/my_new_cs.cpp
    ${CS_PROVIDERS_PATH}/my_new_cs.h
)
```
**Step 2:** If the provider needs linking to external libraries, you can add then to the kernel in [src/CMakeLists.txt](src/CMakeLists.txt):
```cmake
# Build basic SWHT
add_library(swht_kernel SHARED swht_basic.cpp swht_robust.cpp swht_kernel.h ${UTILS} ${CS_PROVIDERS} "${CMAKE_BINARY_DIR}/include/build_info.h")
target_include_directories(swht_kernel PUBLIC
    ...
    <path-to-required-includes-folder>
)
target_link_libraries(swht_kernel PUBLIC ... <required-libraries>)
```

## Making the provider usable with `swht`
Now that the provider is being built and linked, to make it a standard parameter of `swht` you must instantiate its version of the `swht_basic` and `swht_robust` kernel functions.

**Step 1:** In the [cs_factory.h](src/cs_providers/cs_factory.h) header, declare the factory for your provider:
```c++
#define NEW_CS <unique-id>
template <> finite_field_cs *get_finite_field_cs<NEW_CS, <required-args-types> ...>(unsigned long n, <required-args> ...);
```
then define it in [cs_factory.cpp](src/cs_providers/cs_factory.cpp)
```c++
template <> finite_field_cs *get_finite_field_cs<NEW_CS, <required-args-types> ...>(unsigned long n, <required-args> ...) {
    ...
    return <pointer-to-provider-instance>;
}
```

**Step 2:** Instantiate the transform kernel functions with your provider, by adding the following at the bottom of the files:

In [swht_basic.cpp](src/swht_basic.cpp):
```c++
template int swht_basic<NEW_CS, <required-args-types> ...>(PyObject *signal, frequency_map &out, unsigned long n, unsigned long K, double C, double ratio, <required-args> ...);
```
In [swht_robust.cpp](src/swht_robust.cpp):
```c++
template int swht_robust<NEW_CS, <required-args-types> ...>(PyObject *signal, frequency_map &out, unsigned long n, unsigned long K, double C, double ratio, unsigned long robust_iterations, <required-args> ...);
```

**Step 3:** Add your new kernel calls to the C++ interface, in [swht.h](src/swht.h):
```c++
/** CS algorithms
 * Record of the available compressive sensing algorithms.
 */
const static std::unordered_map<std::string, int> cs_algorithms = {
    ...
    {"<provider-name>", <provider-id>}
};

/** SWHT
 * General C++ interface for the SWHT transform.
 */
frequency_map swht(
    ...
    <your-additional-args> // MUST PROVIDE DEFAULTS !!!
);
```
and in [swht.cpp](src/swht.cpp):
```c++
/** SWHT
 * General C++ interface for the SWHT transform.
 */
frequency_map swht(..., <your-additional-args>) {

    ...

    // Additional checks for your parameters
    if (<error-check>) {
        throw std::invalid_argument("<error-message>");
    }

    // Call swht
    frequency_map out;
    bool use_basic = robust_iterations == 1ul;
    switch (cs_algo_num) {
    ...
    case <provider-id>:
        if (use_basic) swht_basic<<provider-id>>(signal, out, n, K, C, ratio, degree, <required-args>);
        else swht_robust<<provider-id>>(signal, out, n, K, C, ratio, robust_iterations, degree, <required-args>);
        break;
    ...
    }
    ...
}
```

**NOTE:** You only need to add `<your-additional-args>` if your provider needs arguments that are not already used by another provider.

**Step 4:** Finally, you need to add the additional arguments required by your provider (if there are any) to the Python interface in [swhtmodule.cpp](src/python_module/swhtmodule.cpp).  
1. Declare the additional arguments:
```c++
static PyObject *swht_swht(PyObject *self, PyObject *args, PyObject *kwargs) {
    ...
    
    // Arguments
    ...
    PyObject *py_my_arg = nullptr;
    ...

    // Arguments keyword names
    static const char *kwlist[12 + <#-added-args>] = {
        ...
        "my_arg",
        ...
        NULL
    };
    
    // Parsing
    if (
        !PyArg_ParseTupleAndKeywords(
            args, kwargs, "OsOO|$OOOOOOO<O * <#-added-args>>", (char **)kwlist,
            ...
            &py_my_arg,
            ...
        )
    ) return NULL;
    ...
}
```
2. Convert them to C++ types and add them to the `swht` C++ call:
```c++
static PyObject *swht_swht(PyObject *self, PyObject *args, PyObject *kwargs) {
    
    ...

    // Argument processing: my_arg (optional)
    <type> my_arg = <default-value>;
    if (py_my_arg != NULL) {
        my_arg = <converter>(py_my_arg);
        if (<bad-conversion>)
            return NULL;
    }

    ...

    // Call C/C++ function
    frequency_map out;
    try {
        out = swht(signal, cs_algorithm, n, K, robust_iter, C, ratio,
            cs_bins, cs_iterations, cs_ratio, degree, ..., my_arg, ...);
    } 

    ...
}
```
3. (Optional but recommended) Add a constant for your algorithm's name:
```c++
/** Module initialization
 * Generates and returns module pointer when calling 'import'.
 */
PyMODINIT_FUNC PyInit_swht(void) {

    ...
    
    // Add NEW_CS constant to the module
    PyObject *NEW_CS_cst = PyUnicode_FromString("<provider-name>");
    if (PyModule_AddObject(py_module, "NEW_CS", NEW_CS_cst) < 0) {
        Py_DECREF(NEW_CS_cst);
        Py_DECREF(py_module);
        return NULL;
    }

    ...
}
```
4. (Optional but recommended) Make the constant accessible from the general package in [\_\_init__.py](src/python_module/\_\_init__.py):
```Python
...
__all__ = [
    ...
    "NEW_CS"
]
```

5. (Optional but recommended) Update the docstring `swht_func_doc` in the file and the interface doc in [\_\_init__.pyi](src/python_module/\_\_init__.pyi) to get pretty and complete information about your new arguments. And declare your provider name constant (here you can literally put the three dots):
```Python
NEW_CS: str = ...
```
**NOTE:** For the conversion of Python objects to C++, a converter for `unsigned long` is already provided as well as one for `double`. For other types it is recommended to take inspiration from these ones and read the [CPython documentation](https://docs.python.org/3.7/c-api/index.html) for an API function reference.
