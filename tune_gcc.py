import argparse
import multiprocessing
from tuner import RandomTuner, SRTuner, BOCSTuner, FourierTuner
from tuner import Evaluator
from tuner import convert_to_str, read_gcc_opts


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


default_budget = 10
default_programs = ["cbench-network-dijkstra"]
default_datasets = ["cdataset-dijkstra-0001"]
default_commands = [""]

parser = argparse.ArgumentParser()
parser.add_argument("--budget", default=default_budget, type=int)
parser.add_argument("--program", default=default_programs, action=SplitArgs)
parser.add_argument("--dataset", default=default_datasets, action=SplitArgs)
parser.add_argument("--command", default=default_commands, action=SplitArgs)
parser.add_argument("--parallel", type=int)
args = parser.parse_args()

search_space = read_gcc_opts("gcc_opts.txt")
default_setting = {"stdOptLv": 3}

for i in range(100):
    try:
        with open(f"results/tuning_{i:02d}.txt", "x"):
            nonce = i
            break
    except Exception:
        pass


def tuning_thread(program, dataset, command):
    evaluator = Evaluator(program, 1, search_space, dataset, command)
    tuners = [
        RandomTuner(search_space, evaluator, default_setting),
        SRTuner(search_space, evaluator, default_setting),
        BOCSTuner(search_space, evaluator, default_setting),
        FourierTuner(search_space, evaluator, default_setting)
    ]
    # append suffix to ensure unique file name
    if command == "encode":
        program += "-e"
    elif command == "decode":
        program += "-d"
    elif command != "":
        raise ValueError("Unrecognized command " + command)
    for tuner in tuners:
        tuner_file = f"results/tuning_{nonce:02d}_{program}_{tuner.name}.txt"
        with open(tuner_file, "x") as fh:
            best_opt_setting, best_perf = tuner.tune(args.budget, file=fh)
        default_flags = convert_to_str(tuner.default_setting, search_space)
        best_flags = convert_to_str(best_opt_setting, search_space)
        with open(f"results/tuning_{nonce:02d}.txt", "a") as fh:
            fh.write("\n")
            fh.write(f"{program} with {tuner.name}\n")
            fh.write(f"speedup: {tuner.default_perf / best_perf:.3f}\n")
            fh.write(f"default runtime: {tuner.default_perf:.3e} s\n")
            fh.write(f"best runtime: {best_perf:.3e} s\n")
            fh.write(f"default flags: {default_flags}\n")
            fh.write(f"best flags: {best_flags}\n")


tuning_thread_args = zip(args.program, args.dataset, args.command)
if args.parallel is not None:
    with multiprocessing.Pool(processes=args.parallel) as pool:
        pool.starmap(tuning_thread, tuning_thread_args)
else:
    for program, dataset, command in tuning_thread_args:
        tuning_thread(program, dataset, command)
