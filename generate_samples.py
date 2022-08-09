import argparse

import compiler_opt


parser = argparse.ArgumentParser()
parser.add_argument(
    "modules",
    default=["cbench-network-dijkstra:cdataset-dijkstra-0001:"],
    nargs="*")
parser.add_argument(
    "--search_space",
    default="gcc_flags/search_space_98.txt",
    type=str)
parser.add_argument("--samples", default=10, type=int)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

compiler_opt.Samples(
    args.modules,
    compiler_opt.read_gcc_search_space(args.search_space),
    args.samples,
    args.parallel)
