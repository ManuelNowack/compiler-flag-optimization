import argparse

import compiler_opt


parser = argparse.ArgumentParser()
parser.add_argument(
    "modules",
    default=["cbench-network-dijkstra:cdataset-dijkstra-0001:"],
    nargs="*")
parser.add_argument("--samples", default=10, type=int)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

compiler_opt.Samples(args.modules, args.samples, args.parallel)
