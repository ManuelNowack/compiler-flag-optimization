import argparse
import benchmark
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
parser.add_argument("--parallel", type=int)
args = parser.parse_args()


for i in range(100):
    try:
        with open(f"results/stability_check_{i:02d}.txt", "x"):
            nonce = i
            break
    except Exception:
        pass
with open(f"results/stability_check_{nonce:02d}.txt", "w") as fh:
    def benchmark_thread(program, dataset):
        benchmark.compile(program, "-O3")
        run_times = [benchmark.run(program, dataset)
                     for _ in range(args.repetitions)]
        return np.array(run_times)

    if args.parallel is not None:
        with multiprocessing.Pool(processes=args.parallel) as pool:
            o = pool.starmap(benchmark_thread, zip(args.program, args.dataset))
    else:
        o = map(benchmark_thread, args.program, args.dataset)
    for program, dataset, run_times in zip(args.program, args.dataset, o):
        # Write runtimes to file for later usage
        np.savetxt(f"results/{program}_{nonce:02d}.txt", run_times)
        # Log noise for your information
        noise = np.abs(1 - run_times / np.median(run_times))
        fraction_noisy = np.count_nonzero(noise > 0.01) / len(noise)
        percentile = np.percentile(noise, 95)
        max_noise = 1.0 - run_times.min() / run_times.max()
        std = np.std(run_times) / np.mean(run_times)
        fh.write(f"{program}\n")
        fh.write(f"Deviating more than 1%: {fraction_noisy:.2f}\n")
        fh.write(f"95-percentile: {percentile * 100:.2f} %\n")
        fh.write(f"Max noise: {max_noise * 100:.2f} %\n")
        fh.write(f"Relative standard deviation {std}\n")
        fh.write("\n")
