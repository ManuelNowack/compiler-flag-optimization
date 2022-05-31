import datetime
import multiprocessing

from tuner import RandomTuner, SRTuner, BOCSTuner
from tuner import Evaluator
from tuner import convert_to_str, read_gcc_opts


if __name__ == "__main__":
    # Assign the number of trials as the budget.
    budget = 1000
    # Benchmark info
    program_list = [
        ("cbench-consumer-jpeg-c", "image-ppm-0002", ""),
        ("cbench-network-dijkstra", "cdataset-dijkstra-0002", ""),
        ("cbench-security-blowfish", "", "encode"),
        ("cbench-security-blowfish", "", "decode"),
        ("cbench-telecom-adpcm-c", "adpcm-0002", ""),
        ("cbench-telecom-adpcm-d", "adpcm-0002", ""),
        ("cbench-telecom-crc32", "adpcm-0002", ""),
        ("cbench-telecom-gsm", "", ""),
        ("cbench-bzip2", "", "decode")]
    gcc_optimization_info = "gcc_opts.txt"

    # Extract GCC search space
    search_space = read_gcc_opts(gcc_optimization_info)
    default_setting = {"stdOptLv": 3}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    result_file = f"results/tuning_{timestamp}.txt"
    with open(result_file, "x") as fh:
        fh.write("=== Result ===\n")

    def tuning_process(program, dataset, command):
        evaluator = Evaluator(program, 1, search_space, dataset, command)
        tuners = [
            RandomTuner(search_space, evaluator, default_setting),
            SRTuner(search_space, evaluator, default_setting),
            BOCSTuner(search_space, evaluator, default_setting)
        ]
        # append suffix to ensure unique file name
        if command == "encode":
            program += "-e"
        elif command == "decode":
            program += "-d"
        elif command != "":
            raise ValueError("Unrecognized command " + command)
        for tuner in tuners:
            tuner_file = f"results/tuning_{timestamp}_{program}_{tuner.name}.txt"
            with open(tuner_file, "x") as fh:
                best_opt_setting, best_perf = tuner.tune(budget, file=fh)
            default_flags = convert_to_str(tuner.default_setting, search_space)
            best_flags = convert_to_str(best_opt_setting, search_space)
            with open(result_file, "a") as fh:
                fh.write("\n")
                fh.write(f"{program} with {tuner.name}\n")
                fh.write(f"speedup: {tuner.default_perf / best_perf:.3f}\n")
                fh.write(f"default runtime: {tuner.default_perf:.3e} s\n")
                fh.write(f"best runtime: {best_perf:.3e} s\n")
                fh.write(f"default flags: {default_flags}\n")
                fh.write(f"best flags: {best_flags}\n")

    with multiprocessing.Pool(processes=len(program_list)) as pool:
        pool.starmap(tuning_process, program_list)
