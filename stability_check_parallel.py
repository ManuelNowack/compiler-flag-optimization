import argparse
import benchmark
import datetime
import itertools
import multiprocessing
import numpy as np


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


default_programs = ["cbench-network-dijkstra"]
default_datasets = ["cdataset-dijkstra-0001"]

parser = argparse.ArgumentParser()
parser.add_argument("--repetitions", default=10, type=int)
parser.add_argument("--program", default=default_programs, action=SplitArgs)
parser.add_argument("--dataset", default=default_datasets, action=SplitArgs)
args = parser.parse_args()


def benchmark_process(program, dataset, repetitions):
    benchmark.compile(program, "-O3")
    run_times = [benchmark.run(program, dataset) for _ in range(repetitions)]
    return np.array(run_times)


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
with open(f"results/stability_check_{timestamp}.txt", "w") as fh:
    with multiprocessing.Pool(processes=len(args.program)) as pool:
        a = zip(args.program, args.dataset, itertools.repeat(args.repetitions))
        output = pool.starmap(benchmark_process, a)
    for program, dataset, run_times in zip(args.program, args.dataset, output):
        # Write runtimes to file for later usage
        np.savetxt(f"results/{program}_{timestamp}.txt", run_times)
        # Log noise for your information
        fh.write(f"{program}\n")
        max_noise = 1.0 - run_times.min() / run_times.max()
        fh.write(f"Max noise: {max_noise * 100:.2f} %\n")
        std = np.std(run_times) / np.mean(run_times)
        fh.write(f"Relative standard deviation {std}\n")
