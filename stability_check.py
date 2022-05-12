import datetime
import numpy as np
import os
import sys

import benchmark


try:
    repetitions = int(sys.argv[1])
except IndexError:
    repetitions = 10
run_times = np.array([benchmark.run("") for _ in range(repetitions)])
# Write runtimes to file for later usage
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
with open(f"results/stability_{timestamp}.txt", "x") as fh:
    np.savetxt(fh, run_times)
# Log noise for your information
with open(f"results/{os.path.basename(__file__)}.log", "w") as fh:
    reference_time = np.median(run_times)
    noise = np.abs(1 - run_times / reference_time)
    for x in noise:
        fh.write(f"{x * 100:.2f} %\n")
    max_noise = 1.0 - run_times.min() / run_times.max()
    fh.write(f"Max noise: {max_noise * 100:.2f} %\n")
    std = np.sqrt(np.mean((np.abs(run_times - reference_time) ** 2)))
    fh.write(f"Standard deviation (from median) {std}:\n")
