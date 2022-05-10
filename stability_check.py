import numpy as np
import os
import sys

import benchmark


def print_noise(run_times: np.ndarray):
    reference_time = np.median(run_times)
    noise = np.abs(1 - run_times / reference_time)
    for x in noise:
        print("{:.2f} %".format(x * 100), file=log)
    max_noise = 1.0 - run_times.min() / run_times.max()
    print("Max noise: {:.2f} %".format(max_noise * 100), file=log)
    std = np.sqrt(np.mean((np.abs(run_times - reference_time) ** 2)))
    print("Standard deviation (median):", std, file=log)


log = open("results/{:s}.log".format(os.path.basename(__file__)), "w")

try:
    repetitions = int(sys.argv[1])
except IndexError:
    repetitions = 10
run_times = np.array([benchmark.run("") for _ in range(repetitions)])
with open("results/stability.txt", "a") as fh:
    np.savetxt(fh, run_times)
print("=== Current benchmark ===", file=log)
print_noise(run_times)
print("=== All benchmarks ===", file=log)
print_noise(np.loadtxt("results/stability.txt"))
