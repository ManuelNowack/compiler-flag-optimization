import multiprocessing

import pandas as pd

import compiler_opt


class Experiment():
    def __init__(
            self,
            programs: tuple[str, str, str],
            tuner_types: list[type],
            budget: int,
            parallel: int = None):
        self.programs = programs
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
        if command == "":
            benchmark_name = program
        else:
            benchmark_name = f"{program}-{command}"
        for tuner in tuners:
            file_name = (f"results/tuning_{self.nonce:02d}_{benchmark_name}"
                         f"_{tuner.name}.txt")
            with open(file_name, "x", buffering=1) as fh:
                best_optimization, best_perf = tuner.tune(self.budget, file=fh)
            tuner.best_optimization = best_optimization
            tuner.best_perf = best_perf
        return tuners

    def run_(self) -> None:
        if self.parallel is not None:
            with multiprocessing.Pool(processes=self.parallel) as pool:
                self.results = pool.starmap(self.tuning_thread_, self.programs)
        else:
            self.results = [self.tuning_thread_(program, dataset, command)
                            for program, dataset, command in self.programs]

    def write_results_(self) -> None:
        row_names = [name for tuner in self.results[0]
                     for name in ("Default", tuner.name)]
        df = pd.DataFrame(index=row_names)
        for (program, _, command), tuners in zip(self.programs, self.results):
            if command == "":
                benchmark_name = program
            else:
                benchmark_name = f"{program}-{command}"
            df[benchmark_name] = [perf for t in tuners
                                  for perf in (t.default_perf, t.best_perf)]
            for tuner in tuners:
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
        df.to_csv(f"results/tuning_{self.nonce:02d}.csv")