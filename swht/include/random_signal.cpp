//=====================
// Random signal class
//=====================


#include "random_signal.h"

#include <gmpxx.h>

#if __GNUC__ > 7
#include <algorithm>
#else
#include <experimental/algorithm>
#endif
#include <numeric>
#include <vector>
#include <list>

#include "debugging.h" // also used in the unit tests


/** Boolean inner-product
 * And on each index then reduce with xor.
 */
inline int bool_inner_product(const vbits &v, PyObject **w) {
    size_t n = v.size();
    int out = 0;
    for (unsigned long i = 0ul; i < n; i++) {
        int bit = ((PyLongObject *) w[i])->ob_digit[0];
        out ^= v[i] && bit;
    }
    return out;
}


/**
 * Compute and store a query result.
 */
PyObject *init_call(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void) kwargs;
    PyObject *internal_tuple = ((PyTupleObject *) args)->ob_item[0];
    PyObject **index = ((PyTupleObject *) internal_tuple)->ob_item;
    RandomSignal *signal_reference = (RandomSignal *) self;
    double value = 0.;
    for (const auto &freq_amp: signal_reference->frequencies) {
        if (bool_inner_product(freq_amp.first, index)) {
            value -= freq_amp.second;
        } else {
            value += freq_amp.second;
        }
    }
    signal_reference->history.push_back(value);
    return PyFloat_FromDouble(value);
}


/**
 * Output query results based on history.
 */
PyObject *fast_call(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void) args; (void) kwargs;
    double response = *(((RandomSignal *) self)->history_reader++);
    return PyFloat_FromDouble(response);
}


/**
 * Randomly generates a frequency-amplitude pair and adds it to the internal
 * mapping (makes sure that it is unique).
 */
void RandomSignal::add_random_coefficient() {

    // Loop while no new frequency found
    bool not_fresh = true;
    std::vector<unsigned long> indexes(n);
    std::iota(indexes.begin(), indexes.end(), 0ul);
    do {
    
        // Pick a degree
        unsigned long effective_degree = degree_picker(generator);

        // Generate random frequency
        std::list<unsigned long> selection;
#if __GNUC__ > 7
        std::sample(indexes.begin(), indexes.end(),
#else
        std::experimental::sample(indexes.begin(), indexes.end(),
#endif
            std::back_inserter(selection), effective_degree, generator);
        vbits frequency(n, 0u);
        for (const unsigned long &i: selection)
            frequency[i] = 1u;

        // If unique store with random amplitude
        auto match = frequencies.find(frequency);
        if (match == frequencies.end()) {
            frequencies[frequency] = amplitude_picker(generator);
            not_fresh = false;
        }
        
    } while (not_fresh);
}


/**
 * Generates a mapping of K random frequency-amplitude items and readies the
 * base query function.
 */
RandomSignal::RandomSignal(unsigned long n, unsigned long K, unsigned long degree):
n(n), degree(degree), frequencies({}), generator(std::random_device()()), amplitude_picker(1, 10) {

    // Set initial query method
    ob_type = new _typeobject();
    ob_type->tp_call = &init_call;

    // Define frequency degree random selector (binomial coefficients as likelihood)
    unsigned long n_degrees = degree + 1ul;
    std::vector<mpz_class> binomial_coefficients(n_degrees);
    mpz_class total_coefficients = 0;
    for (unsigned long d = 0ul; d <= degree; d++) {
        mpz_bin_uiui(binomial_coefficients[d].get_mpz_t(), n, d);
        total_coefficients += binomial_coefficients[d];
    }
    std::vector<double> degree_probabilities(n_degrees);
    for (unsigned long d = 0ul; d <= degree; d++)
        degree_probabilities[d] = mpq_class(binomial_coefficients[d], total_coefficients).get_d();
    degree_picker = std::discrete_distribution<unsigned long>(degree_probabilities.begin(), degree_probabilities.end());

    // Generate K random frequencies
    for (unsigned long i = 0ul; i < K; i++)
        add_random_coefficient();
}


/**
 * Cleans up the fake type object contained.
 */
RandomSignal::~RandomSignal() {
    delete ob_type;
}


/**
 * Readies the history-based querying.
 */
void RandomSignal::ready_fast_calls() {
    ob_type->tp_call = &fast_call;
    history_reader = history.begin();
}


/**
 * Compares the signal to an output mapping.
 */
bool RandomSignal::operator==(frequency_map other) {
    if (frequencies.size() != other.size()) return false;
    for (const auto &item: other) {
        auto match = frequencies.find(item.first);
        if (match == frequencies.end()) return false;
        if (std::abs(item.second - match->second) > 0.1) return false;
    }
    return true;
}
bool RandomSignal::operator==(PyObject *other) {
    if (frequencies.size() != (size_t) PyDict_Size(other)) return false;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(other, &pos, &key, &value)) {
        vbits frequency(n);
        for (size_t i = 0ul; i < n; i++) frequency[i] = (unsigned) PyLong_AsUnsignedLong(PyTuple_GET_ITEM(key, i));
        auto match = frequencies.find(frequency);
        if (match == frequencies.end()) return false;
        if (std::abs(PyFloat_AsDouble(value) - match->second) > 0.1) return false;
    }
    return true;
}


/**
 * Signal easy display for debugging.
 */
std::ostream &operator<<(std::ostream &output, const std::unordered_map<vbits, double, bit_hash> &x) {
    if (x.empty()) {
        output << "{}";
        return output;
    }
    auto iter = x.begin();
    output << '{' << iter->first << ": " << iter->second;
    iter++;
    for (; iter != x.end(); iter++) {
        output << ", " << iter->first << ": " << iter->second;
    }
    output << '}';
    return output;
}
std::ostream &operator<<(std::ostream &output, const RandomSignal &signal) {
    return operator<<(output, signal.frequencies);
}
std::ostream &operator<<(std::ostream &output, frequency_map &x) {
    if (x.empty()) {
        output << "{}";
        return output;
    }
    auto iter = x.begin();
    output << '{' << iter->first << ": " << iter->second;
    iter++;
    for (; iter != x.end(); iter++) {
        output << ", " << iter->first << ": " << iter->second;
    }
    output << '}';
    return output;
}
