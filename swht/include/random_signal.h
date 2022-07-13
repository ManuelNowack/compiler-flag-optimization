//=====================
// Random signal class
//=====================

#ifndef RANDOM_SIGNAL_H
#define RANDOM_SIGNAL_H


#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <random>
#include <list>

#include "global_constants.h"


/**
 * Base and fast query functions, respectively for real response and history
 * replay (switch to fast_call for replay mode).
 */
PyObject *init_call(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *fast_call(PyObject *self, PyObject *args, PyObject *kwargs);


/**
 * Simulation of a signal that randomly assigns itself frequencies and answers
 * queries in the time domain based on these frequencies. Allows for quick
 * verification of a SWHT output and history storage for quick query replay.
 */
struct RandomSignal: PyObject {

    unsigned long n, degree;
    frequency_map frequencies;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> amplitude_picker;
    std::discrete_distribution<unsigned long> degree_picker;
    std::list<double> history;
    std::list<double>::iterator history_reader;

    RandomSignal(unsigned long n, unsigned long K, unsigned long degree);
    ~RandomSignal();

    /**
     * Randomly generates a frequency-amplitude pair and adds it to the internal
     * mapping (makes sure that it is unique).
     */
    void add_random_coefficient();

    /**
     * Readies the history-based querying.
     */
    void ready_fast_calls();

    /**
     * Compares the signal to an output mapping.
     */
    bool operator==(frequency_map other);
    bool operator==(PyObject *other);
};


// Debug display
std::ostream &operator<<(std::ostream &output, const frequency_map &x);
std::ostream &operator<<(std::ostream &output, const RandomSignal &signal);

#endif
