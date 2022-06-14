import multiprocessing

import compiler_opt


class Experiment():
    def __init__(
            self,
            programs: list[str],
            datasets: list[str],
            commands: list[str],
            tuner_types: list[type],
            budget: int,
            parallel: int = None):
        self.programs = programs
        self.datasets = datasets
        self.commands = commands
        self.tuner_types = tuner_types
        self.budget = budget
        self.parallel = parallel
        self.search_space = compiler_opt.read_gcc_search_space("gcc_opts.txt")
        self.default_optimization = {"stdOptLv": 3}
        self.nonce_()
        self.run_()
        self.write_results_()

    def nonce_(self) -> None:
        for i in range(100):
            try:
                with open(f"results/tuning_{i:02d}.txt", "x"):
                    self.nonce = i
                    break
            except Exception:
                pass
        else:
            raise ValueError("No nonce available")

    def tuning_thread_(self, program: str, dataset: str,
                       command: str) -> list[compiler_opt.Tuner]:
        evaluator = compiler_opt.Evaluator(
            program, 1, self.search_space, dataset, command)
        tuners: list[compiler_opt.Tuner] = [
            tuner_type(
                self.search_space,
                evaluator,
                self.default_optimization) for tuner_type in self.tuner_types]
        # append suffix to ensure unique file name
        if command == "encode":
            program += "-e"
        elif command == "decode":
            program += "-d"
        elif command != "":
            raise ValueError("Unrecognized command " + command)
        for tuner in tuners:
            tuner_file = f"results/tuning_{self.nonce:02d}_{program}_{tuner.name}.txt"
            with open(tuner_file, "x", buffering=1) as fh:
                best_optimization, best_perf = tuner.tune(self.budget, file=fh)
            tuner.best_optimization = best_optimization
            tuner.best_perf = best_perf
        return tuners

    def run_(self) -> None:
        tuning_thread_args = zip(self.programs, self.datasets, self.commands)
        if self.parallel is not None:
            with multiprocessing.Pool(processes=self.parallel) as pool:
                self.results = pool.starmap(
                    self.tuning_thread_, tuning_thread_args)
        else:
            self.results = map(
                self.tuning_thread_,
                self.programs,
                self.datasets,
                self.commands)

    def write_results_(self) -> None:
        for program, command, tuners in zip(
                self.programs, self.commands, self.results):
            for tuner in tuners:
                if command == "encode":
                    benchmark_name = program + "-e"
                elif command == "decode":
                    benchmark_name = program + "-d"
                elif command == "":
                    benchmark_name = program
                else:
                    raise ValueError("Unrecognized command " + command)
                default_flags = compiler_opt.optimization_to_str(
                    tuner.default_optimization, tuner.search_space)
                best_flags = compiler_opt.optimization_to_str(
                    tuner.best_optimization, tuner.search_space)
                speedup = tuner.default_perf / tuner.best_perf
                with open(f"results/tuning_{self.nonce:02d}.txt", "a") as fh:
                    fh.write("\n")
                    fh.write(f"{benchmark_name} with {tuner.name}\n")
                    fh.write(f"speedup: {speedup:.3f}\n")
                    fh.write(f"default runtime: {tuner.default_perf:.3e} s\n")
                    fh.write(f"best runtime: {tuner.best_perf:.3e} s\n")
                    fh.write(f"default flags: {default_flags}\n")
                    fh.write(f"best flags: {best_flags}\n")
