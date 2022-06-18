import argparse

import compiler_opt


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
parser.add_argument("--samples", default=10, type=int)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

compiler_opt.Samples(args.programs, args.samples, args.parallel)
