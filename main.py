import argparse

import compiler_opt


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


default_budget = 10
default_programs = ["cbench-telecom-crc32"]
default_datasets = ["pcm-0001"]
default_commands = [""]

parser = argparse.ArgumentParser()
parser.add_argument("--budget", default=default_budget, type=int)
parser.add_argument("--program", default=default_programs, action=SplitArgs)
parser.add_argument("--dataset", default=default_datasets, action=SplitArgs)
parser.add_argument("--command", default=default_commands, action=SplitArgs)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

compiler_opt.Experiment(
    args.program,
    args.dataset,
    args.command,
    args.budget,
    args.parallel)
