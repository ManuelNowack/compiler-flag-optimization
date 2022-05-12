import benchmark

from tuner import FlagInfo, Evaluator
from tuner import RandomTuner, SRTuner


class GCCFlagInfo(FlagInfo):
    def __init__(self, name, configs, isParametric, stdOptLv):
        super().__init__(name, configs)
        self.isParametric = isParametric
        self.stdOptLv = stdOptLv


def read_gcc_opts(path):
    """Reads the list of gcc optimizations that follow a certain format.

    Due to a slight difference in GCC distributions, the supported flags are
    confirmed by using -fverbose-asm. Each chunk specifies flags supported
    under each standard optimization levels. Besides flags identified by
    -fverbose-asm, we also considered flags in online doc. They are placed as
    the last chunk and considered as last optimization level. (Any standard
    optimization level would not configure them.)
    """
    search_space = dict()  # pair: flag, configs
    # special case handling
    search_space["stdOptLv"] = GCCFlagInfo(
        name="stdOptLv", configs=[1, 2, 3], isParametric=True, stdOptLv=-1)
    with open(path, "r") as fp:
        stdOptLv = 0
        for raw_line in fp.read().split('\n'):
            # Process current chunk
            if(len(raw_line)):
                line = raw_line.replace(" ", "").strip()
                if line[0] != '#':
                    tokens = line.split("=")
                    flag_name = tokens[0]
                    # Binary flag
                    if len(tokens) == 1:
                        info = GCCFlagInfo(
                            name=flag_name,
                            configs=[False, True],
                            isParametric=False,
                            stdOptLv=stdOptLv)
                    # Parametric flag
                    else:
                        assert(len(tokens) == 2)
                        info = GCCFlagInfo(
                            name=flag_name,
                            configs=tokens[1].split(','),
                            isParametric=True,
                            stdOptLv=stdOptLv)
                    search_space[flag_name] = info
            # Move onto next chunk
            else:
                stdOptLv = stdOptLv + 1
    return search_space


def convert_to_str(opt_setting, search_space):
    str_opt_setting = " -O" + str(opt_setting["stdOptLv"])

    for flag_name, config in opt_setting.items():
        assert flag_name in search_space
        flag_info = search_space[flag_name]
        # Parametric flag
        if flag_info.isParametric:
            if flag_info.name != "stdOptLv" and len(config) > 0:
                str_opt_setting += f" {flag_name}={config}"
        # Binary flag
        else:
            assert(isinstance(config, bool))
            if config:
                str_opt_setting += f" {flag_name}"
            else:
                negated_flag_name = flag_name.replace("-f", "-fno-", 1)
                str_opt_setting += f" {negated_flag_name}"
    return str_opt_setting


# Define tuning task
class cBenchEvaluator(Evaluator):
    def __init__(self, path, num_repeats, search_space):
        super().__init__(path, num_repeats)
        if path == "cbench-consumer-jpeg-c":
            self.dataset = "image-pgm-0001"
        else:
            self.dataset = ""
        self.search_space = search_space

    def evaluate(self, opt_setting, num_repeats=None):
        flags = convert_to_str(opt_setting, self.search_space)
        benchmark.ck_cmd({"action": "compile",
                          "module_uoa": "program",
                          "data_uoa": self.path,
                          "flags": flags,
                          "lflags": "-fopenmp"})
        r = benchmark.ck_cmd({"action": "run",
                              "module_uoa": "program",
                              "data_uoa": self.path,
                              "dataset_uoa": self.dataset})
        return r["characteristics"]["execution_time"]


if __name__ == "__main__":
    # Assign the number of trials as the budget.
    budget = 1000
    # Benchmark info
    program_list = [
        "cbench-network-dijkstra",
        "cbench-consumer-jpeg-c",
        "cbench-telecom-adpcm-d"]
    gcc_optimization_info = "gcc_opts.txt"

    # Extract GCC search space
    search_space = read_gcc_opts(gcc_optimization_info)
    default_setting = {"stdOptLv": 3}

    with open("results/tuning_result.txt", "w") as ofp:
        ofp.write("=== Result ===\n")

    for program in program_list:
        evaluator = cBenchEvaluator(
            program, num_repeats=30, search_space=search_space)

        tuners = [
            RandomTuner(search_space, evaluator, default_setting),
            SRTuner(search_space, evaluator, default_setting)
        ]

        for tuner in tuners:
            best_opt_setting, best_perf = tuner.tune(budget)
            if best_opt_setting is not None:
                default_perf = tuner.default_perf
                best_perf = evaluator.evaluate(best_opt_setting)
                print(f"Tuning {program} w/ {tuner.name}: {default_perf:.3f}/{best_perf:.3f} = {default_perf/best_perf:.3f}x")
                with open("results/tuning_result.txt", "a") as ofp:
                    ofp.write(f"Tuning {program} w/ {tuner.name}: {default_perf:.3f}/{best_perf:.3f} = {default_perf/best_perf:.3f}x\n")
