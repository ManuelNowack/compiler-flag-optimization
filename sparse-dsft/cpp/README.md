# This is still work in progress. Things are expected to fail.

## Compilation and running

### To compile the C++ code, run from the cpp directory:
```
cmake .
make
```

### To run, from the cpp directory:
```
./estimator <data file name> <c> <steps>
```

### To build the Python binding:

After compiling the C++ code, run from the cpp directory:
```
python setup-cpp.py build_ext -i
```

### To run in Python:

In the main project directory, add the following imports:
```
import sys
sys.path.append('./cpp')
import _fit
```

And then replace the line
```
estimator.fit(X_train, Y_train)
```
with
```
cpp_est = _fit.cpp_fit(X_train, Y_train, <C>, <N_LAMBDAS>, <STEPS>, <IS_RECURSIVE>)
estimator.from_cpp(cpp_est, Y_train)
```

### To run only the correlation functions:

In the main project directory, add the following imports:
```
import sys
sys.path.append('./cpp')
import _fit
```

And then create object
```
executor = _fit.CorrExecutor(<X>, <N_THREADS>)
```
This creates the executor object with data `<X>` and specified number of threads.
If `N_THREADS` is set to zero, this produces the executor with the number of threads
equal to the number of threads available on the machine (hardware concurrency).

Then, correlation functions can be run as:
```
corr = executor.compute_max_correlation(<Y>, freq, fB)
```

#### Diagnostic information:

CorrExecutor stores some diagnostic information from the last function run:
- `last_max_queue` - maximum queue size (iterative) or maximum recursion depth (recursive);
- `last_total_processed` - total number of subsets processed.

You can access these elements as fields of CorrExecutor, e.g.:
```
executor = _fit.CorrExecutor(X_train, 0)
corr = executor.compute_max_correlation(Y_train, freq, fB)
print(executor.last_max_queue, executor.last_total_processed)
```

NOTE: Collecting this information can potentially harm execution speed.
If you don't want to collect this, you can edit the file `cpp/src/common.hpp`.
Change
```
#define GET_STATS true
```
to
```
#define GET_STATS false
```
and recompile to disable the collection of diagnostic information.

## Further comments

### Data format

The data files currently need a specific binary format:
- 32-bit int: #samples
- 32-bit int: #features
- 8-bit bool * #samples * #features: X
- 64-bit double * #samples: Y

Some data files can be converted to this format:
```
mkdir cpp-data
python readwrite.py
```
### Modes of operation

To change mode of operation, look up precompiler variables #defined in file src/common.hpp and modify according to needs. Note that the modes are incremental and not all combinations will work.
