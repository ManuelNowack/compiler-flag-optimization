import argparse

import compiler_opt


tuner_type_dict = {
    "active-fourier": compiler_opt.ActiveFourierTuner,
    "active-fourier-low-degree": compiler_opt.ActiveLowDegreeTuner,
    "bocs-sa": compiler_opt.BOCSSATuner,
    "bocs-sdp": compiler_opt.BOCSSDPTuner,
    "fourier": compiler_opt.FourierTuner,
    "fourier-low-degree": compiler_opt.LowDegreeTuner,
    "hadamard": compiler_opt.HadamardTuner,
    "mono": compiler_opt.MonoTuner,
    "random": compiler_opt.RandomTuner,
    "sr": compiler_opt.SRTuner}


class SplitArgsTuner(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        tuner_types = [tuner_type_dict[tuner] for tuner in values.split(",")]
        setattr(namespace, self.dest, tuner_types)


default_modules = ["cbench-telecom-crc32:pcm-0001:"]
default_tuners = list(tuner_type_dict.values())
default_search_space = "gcc_flags/search_space_98.txt"
default_budget = 10
default_reruns = 1

parser = argparse.ArgumentParser()
parser.add_argument("modules", default=default_modules, nargs="*")
parser.add_argument("--tuner", default=default_tuners, action=SplitArgsTuner)
parser.add_argument("--search_space", default=default_search_space, type=str)
parser.add_argument("--budget", default=default_budget, type=int)
parser.add_argument("--rerun", default=default_reruns, type=int)
parser.add_argument("--simulation", action="store_true")
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

if args.simulation:
    search_space = compiler_opt.fake_gcc_search_space(int(args.search_space))
    evaluator_type = compiler_opt.Simulator
else:
    search_space = compiler_opt.read_gcc_search_space(args.search_space)
    evaluator_type = compiler_opt.Evaluator

for _ in range(args.rerun):
    compiler_opt.Experiment(
        args.modules,
        args.tuner,
        search_space,
        args.budget,
        evaluator_type,
        args.parallel)
