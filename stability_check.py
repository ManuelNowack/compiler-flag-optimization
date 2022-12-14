import argparse
import multiprocessing
import shutil

import numpy as np
import pandas as pd

from compiler_opt import benchmark


parser = argparse.ArgumentParser()
parser.add_argument(
    "modules",
    default=["cbench-network-dijkstra:cdataset-dijkstra-0001:"],
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


def all_benchmarks_finished():
    for module in args.modules:
        file_name = f"results/stability_{nonce:02d}_{module}.txt"
        with open(file_name) as fh:
            if len(fh.readlines()) < args.repetitions:
                return False
    return True


def benchmark_thread(module: str):
    program, dataset, command = module.split(":")
    tmp_dir = benchmark.compile(program, "-w -O3", True)
    repeat = benchmark.get_repeat(program, dataset, command, "-w -O3")
    run_times = []
    file_name = f"results/stability_{nonce:02d}_{module}.txt"
    with open(file_name, "x", buffering=1) as fh:
        for _ in range(args.repetitions):
            run_time = benchmark.run(
                program, dataset, command, tmp_dir, repeat)
            run_times.append(run_time)
            print(run_time, file=fh)
    if args.parallel is not None:
        while not all_benchmarks_finished():
            benchmark.run(program, dataset, command, tmp_dir, repeat)
    shutil.rmtree(tmp_dir)
    return np.array(run_times)


with open(f"results/stability_{nonce:02d}.txt", "w") as fh:
    if args.parallel is not None:
        with multiprocessing.Pool(processes=args.parallel) as pool:
            results = pool.map(benchmark_thread, args.modules)
    else:
        results = [benchmark_thread(module) for module in args.modules]
    df = pd.DataFrame()
    for module, run_times in zip(args.modules, results):
        df[module] = run_times
        noise = np.abs(1 - run_times / np.median(run_times))
        fraction_noisy = np.count_nonzero(noise > 0.01) / len(noise)
        percentile = np.percentile(noise, 95)
        max_noise = 1.0 - run_times.min() / run_times.max()
        std = np.std(run_times) / np.mean(run_times)
        fh.write(f"{module}\n")
        fh.write(f"Mean runtime: {np.mean(run_times)} s\n")
        fh.write(f"Deviating more than 1%: {fraction_noisy:.2f}\n")
        fh.write(f"95-percentile: {percentile * 100:.2f} %\n")
        fh.write(f"Max noise: {max_noise * 100:.2f} %\n")
        fh.write(f"Relative standard deviation {std}\n")
        fh.write("\n")
    df.to_csv(f"results/stability_{nonce:02d}.csv", index=False)
