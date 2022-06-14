import argparse

import compiler_opt


tuner_type_dict = {
    "bocs": compiler_opt.BOCSTuner,
    "fourier": compiler_opt.FourierTuner,
    "mono": compiler_opt.MonoTuner,
    "random": compiler_opt.RandomTuner,
    "sr": compiler_opt.SRTuner}


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


class SplitArgsTuner(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        tuner_types = [tuner_type_dict[tuner] for tuner in values.split(",")]
        setattr(namespace, self.dest, tuner_types)


default_budget = 10
default_programs = ["cbench-telecom-crc32"]
default_datasets = ["pcm-0001"]
default_commands = [""]
default_tuners = list(tuner_type_dict.values())

parser = argparse.ArgumentParser()
parser.add_argument("--budget", default=default_budget, type=int)
parser.add_argument("--program", default=default_programs, action=SplitArgs)
parser.add_argument("--dataset", default=default_datasets, action=SplitArgs)
parser.add_argument("--command", default=default_commands, action=SplitArgs)
parser.add_argument("--tuners", default=default_tuners, action=SplitArgsTuner)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

compiler_opt.Experiment(
    args.program,
    args.dataset,
    args.command,
    args.tuners,
    args.budget,
    args.parallel)
