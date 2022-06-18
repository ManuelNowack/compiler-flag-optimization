import argparse

import compiler_opt


tuner_type_dict = {
    "bocs": compiler_opt.BOCSTuner,
    "fourier": compiler_opt.FourierTuner,
    "mono": compiler_opt.MonoTuner,
    "random": compiler_opt.RandomTuner,
    "sr": compiler_opt.SRTuner}


class SplitArgsProgram(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        programs = [tuple(x.split(":")) for x in values]
        setattr(namespace, self.dest, programs)


class SplitArgsTuner(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        tuner_types = [tuner_type_dict[tuner] for tuner in values.split(",")]
        setattr(namespace, self.dest, tuner_types)


default_programs = ["cbench-telecom-crc32:pcm-0001:"]
default_tuners = list(tuner_type_dict.values())
default_budget = 10
default_reruns = 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "programs",
    default=default_programs,
    action=SplitArgsProgram,
    nargs="*")
parser.add_argument("--tuner", default=default_tuners, action=SplitArgsTuner)
parser.add_argument("--budget", default=default_budget, type=int)
parser.add_argument("--rerun", default=default_reruns, type=int)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

for _ in range(args.rerun):
    compiler_opt.Experiment(
        args.programs,
        args.tuner,
        args.budget,
        args.parallel)
