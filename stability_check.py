import argparse
import multiprocessing

import numpy as np

from compiler_opt import benchmark


class SplitArgsProgram(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        programs = [tuple(x.split(":")) for x in values]
        setattr(namespace, self.dest, programs)


parser = argparse.ArgumentParser()
parser.add_argument(
    "programs",
    default=["cbench-network-dijkstra:cdataset-dijkstra-0001:"],
    action=SplitArgsProgram,
    nargs="*")
parser.add_argument("--repetitions", default=10, type=int)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()


for i in range(100):
    try:
        with open(f"results/stability_{i:02d}.txt", "x"):
            nonce = i
            break
    except Exception:
        pass
with open(f"results/stability_{nonce:02d}.txt", "w") as fh:
    def benchmark_thread(program, dataset, command):
        dir = benchmark.compile(program, "-w -O3", generate_rnd_tmp_dir=True)
        run_times = [benchmark.run(program, dataset, command, dir)
                     for _ in range(args.repetitions)]
        return np.array(run_times)

    if args.parallel is not None:
        with multiprocessing.Pool(processes=args.parallel) as pool:
            results = pool.starmap(benchmark_thread, args.programs)
    else:
        results = [benchmark_thread(program, dataset, command)
                   for program, dataset, command in args.programs]
    for (program, _, command), run_times in zip(args.programs, results):
        if command == "":
            benchmark_name = program
        else:
            benchmark_name = f"{program}-{command}"
        # Write runtimes to file for later usage
        file_name = f"results/stability_{nonce:02d}_{benchmark_name}.txt"
        np.savetxt(file_name, run_times)
        # Log noise for your information
        noise = np.abs(1 - run_times / np.median(run_times))
        fraction_noisy = np.count_nonzero(noise > 0.01) / len(noise)
        percentile = np.percentile(noise, 95)
        max_noise = 1.0 - run_times.min() / run_times.max()
        std = np.std(run_times) / np.mean(run_times)
        fh.write(f"{benchmark_name}\n")
        fh.write(f"Mean runtime: {np.mean(run_times)} s\n")
        fh.write(f"Deviating more than 1%: {fraction_noisy:.2f}\n")
        fh.write(f"95-percentile: {percentile * 100:.2f} %\n")
        fh.write(f"Max noise: {max_noise * 100:.2f} %\n")
        fh.write(f"Relative standard deviation {std}\n")
        fh.write("\n")
