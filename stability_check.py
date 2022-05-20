import argparse
import benchmark
import datetime
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

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
with open(f"results/stability_check_{timestamp}.txt", "w") as fh:
    for program, dataset in zip(args.program, args.dataset):
        run_times = np.array([benchmark.compile_and_run("", program, dataset)
                             for _ in range(args.repetitions)])
        # Write runtimes to file for later usage
        np.savetxt(f"results/{program}_{timestamp}.txt", run_times)
        # Log noise for your information
        fh.write(f"{program}\n")
        max_noise = 1.0 - run_times.min() / run_times.max()
        fh.write(f"Max noise: {max_noise * 100:.2f} %\n")
        std = np.std(run_times) / np.mean(run_times)
        fh.write(f"Relative standard deviation {std}\n")
