import csv
import itertools
import operator

import benchmark


flags = [
    "-fno-inline-atomics",
    "-fno-inline-functions",
    "-fno-inline-functions-called-once",
    "-fno-inline-small-functions"]
powerset = itertools.chain.from_iterable(
    itertools.combinations(flags, i) for i in range(len(flags) + 1))
times = [benchmark.compile_and_run(" ".join(subset)) for subset in powerset]
with open("results/execution_times.csv", "a") as fh:
    writer = csv.writer(fh)
    writer.writerow(times)
indices = [i for i, _ in sorted(enumerate(times), key=operator.itemgetter(1))]
print(indices)

with open("results/execution_times.csv") as fh:
    reader = csv.reader(fh)
    all_times = [[float(val) for val in row] for row in reader]
    reference_times = all_times[0]
    for times in all_times[1:]:
        diff = [abs(a - b) for a, b in zip(times, reference_times)]
        noise = [a / b for a, b in zip(diff, reference_times)]
        print(noise)
