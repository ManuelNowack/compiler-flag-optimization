import collections
import glob
import math
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable

def shorten_program_name(s: str):
    prefix = "cbench-"
    if not s.startswith(prefix):
        raise ValueError("Not a cbench program")
    return s[len(prefix):]


def shorten_tuner_name(s: str):
    if s == "BOCS-SDP-l1":
        return "BOCS"
    if s == "ActiveFourier":
        return "Fourier"
    if s == "ActiveLowDegree":
        return "Fourier (low-degree)"
    if s == "MonoTuner":
        return "Greedy"
    if s == "RandomTuner":
        return "Random"
    return s


def color_table(path: str, f: Callable[[str], str]):
    new_lines = []
    with open(path) as fh:
        read = False
        for line in fh:
            if read:
                if line == "\\end{longtable}\n":
                    read = False
                else:
                    values = line.removesuffix(" \\\\\n").split(" & ")
                    max_value = f(values[1:])
                    values = [f"\\color{{Green}}{{{val}}}" if val == max_value else val for val in values]
                    line = " & ".join(values) + " \\\\\n"
            elif line == "\\endlastfoot\n":
                read = True
            new_lines.append(line)
    with open(path, "w") as fh:
        fh.writelines(new_lines)


def validate_score_evaluation_20():
    alphas = {
        "1e-1": "fourier_1e-01",
        "1e-2": "fourier",
        "1e-3": "fourier_1e-03",
        "1e-4": "fourier_1e-04",
        "1e-5": "fourier_1e-05"}

    data_validate_scores = []
    for alpha, path in alphas.items():
        validate_scores = []
        for file in glob.glob(f"evaluation/{path}/n_020_budget_0500_??_validate_score.csv"):
            df = pd.read_csv(file, index_col=0).transpose()
            validate_scores.append(df["Fourier"])
        assert validate_scores
        df = pd.DataFrame(validate_scores)
        data_validate_scores.append(df.mean())
    validate_scores = pd.DataFrame(data_validate_scores, index=alphas.keys()).transpose().rename(shorten_program_name)
    s = validate_scores.style
    s.format(precision=2)
    s.to_latex(
        "analysis/table/evaluation_20_validate_scores.tex",
        hrules=True,
        label="table:validate-score",
        caption="$R^2$ validation score of the learned Fourier-sparse set function for different $\\alpha$; 500 queries and 20 flags in the search space",
        environment="longtable")
    color_table("analysis/table/evaluation_20_validate_scores.tex", max)

    data_speedup = []
    for alpha, path in alphas.items():
        speedup = []
        for file in glob.glob(f"evaluation/{path}/n_020_budget_0500_??.csv"):
            df = pd.read_csv(file, index_col=0).transpose()
            speedup.append(df["Train"] / df["Fourier"])
        assert speedup
        df = pd.DataFrame(speedup)
        data_speedup.append(df.mean())
    speedup = pd.DataFrame(data_speedup, index=alphas.keys()).transpose().rename(shorten_program_name)
    s = speedup.style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/evaluation_20_validate_scores_speedup.tex",
        hrules=True,
        label="table:validate-score-speedup",
        caption="Speedup of the minimum of the learned simulated Fourier-sparse set function over the best optimizating setting from the training data for different $\\alpha$; 500 queries and 20 flags in the search space",
        environment="longtable")
    color_table("analysis/table/evaluation_20_validate_scores_speedup.tex", max)


def simulation_validate_score():
    alphas = {
        "1e-1": "regularization_1e-1",
        "1e-2": "noise_5e-3",
        "1e-3": "regularization_1e-3",
        "1e-4": "regularization_1e-4",
        "1e-5": "regularization_1e-5"}
    data_validate_scores = collections.defaultdict(list)
    for alpha, directory in alphas.items():
        for search_space in range(20, 101, 10):
            file = f"simulation/{directory}/n_{search_space:03d}_budget_0500_00_validate_score.csv"
            df = pd.read_csv(file, index_col=0).transpose()
            data_validate_scores[search_space].append(df["Fourier"].mean())
    validate_scores = pd.DataFrame(data_validate_scores, index=alphas.keys())
    s = validate_scores.style
    s.format(precision=2)
    s.to_latex(
        "analysis/table/simulation_validate_scores.tex",
        hrules=True,
        label="table:simulation-validate-score",
        caption="$R^2$ validation score of the learned simulated Fourier-sparse set function for different $\\alpha$; 500 queries and 20-100 flags in the search space",
        environment="longtable")
    validate_scores

    data_speedup = collections.defaultdict(list)
    for alpha, directory in alphas.items():
        for search_space in range(20, 101, 10):
            file = f"simulation/{directory}/n_{search_space:03d}_budget_0500_00.csv"
            df = pd.read_csv(file, index_col=0).transpose()
            data_speedup[search_space].append((df["Train"] / df["Fourier"]).mean())
    speedup = pd.DataFrame(data_speedup, index=alphas.keys())
    s = speedup.style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/simulation_validate_scores_speedup.tex",
        hrules=True,
        label="table:simulation-validate-score-speedup",
        caption="Speedup of the minimum of the learned simulated Fourier-sparse set function over the best optimizating setting from the training data for different $\\alpha$; 500 queries and 20-100 flags in the search space",
        environment="longtable")


def offline_fourier_success_chance_(n: int):
    data_train = []
    data_fourier = []
    for file in glob.glob(f"evaluation/fourier/n_{n:03d}_budget_0500_??.csv"):
        df = pd.read_csv(file, index_col=0).transpose()
        data_train.append(df["Train"])
        data_fourier.append(df["Fourier"])
    df_train = pd.DataFrame(data_train).reset_index(drop=True)
    df_fourier = pd.DataFrame(data_fourier).reset_index(drop=True)
    return (df_fourier < df_train).sum() / len(df_train.index)


def offline_fourier_success_chance():
    data = {
        20: offline_fourier_success_chance_(20),
        98: offline_fourier_success_chance_(98)}
    s = pd.DataFrame(data).rename(index=shorten_program_name).style
    s.format(precision=2)
    s.to_latex(
        f"analysis/table/offline_success_chance.tex",
        hrules=True,
        label="table:offline-success-chance",
        caption="Probability that the minimum of the learned Fourier-sparse set function is better than the best optimization setting from the training data; 500 queries and 20/98 flags in the search space",
        environment="longtable")


def online_fourier_speedup():
    data_random = []
    data_fourier = []
    for file in glob.glob(f"evaluation/random/n_020_budget_0500_??.csv"):
        df = pd.read_csv(file, index_col=0).transpose()
        data_random.append(df["RandomTuner"])
    for file in glob.glob(f"evaluation/active_fourier/n_020_budget_0500_??.csv"):
        df = pd.read_csv(file, index_col=0).transpose()
        data_fourier.append(df["ActiveFourier"])
    random = pd.DataFrame(data_random)
    fourier = pd.DataFrame(data_fourier)
    speedup = random.mean() / fourier.mean()
    s = speedup.rename(index=shorten_program_name).to_frame("Speedup").style
    s.to_latex(
        f"analysis/table/online_speedup_learn.tex",
        hrules=True,
        label="table:online-speedup-learn",
        caption="Speedup of online Fourier-sparse set function learning over the best optimizating setting from randomly sampled training data; 500 queries and 20 flags in the search space",
        environment="longtable")


def evaluation_20_comparison():
    # the critical value of the z-distribution to obtain a 95% confidence interval
    z = {5: 2.571, 10: 2.228, 20: 2.086}
    tuners = {
        "RandomTuner": "random",
        "MonoTuner": "mono",
        "ActiveFourier": "active_fourier",
        "SRTuner": "srtuner"}
    data_runtimes_default = []
    data_runtimes_tuner = []
    data_margin_of_error = []
    for tuner, path in tuners.items():
        data = []
        for file in glob.glob(f"evaluation/{path}/n_020_budget_0500_??.csv"):
            df = pd.read_csv(file, index_col=0).transpose()
            data.append(df[tuner])
            data_runtimes_default.append(df["Default"])
        assert data
        df = pd.DataFrame(data)
        data_runtimes_tuner.append(df.mean())
        data_margin_of_error.append(z[len(df.index)] * df.std() / math.sqrt(len(df.index)))
    df = pd.DataFrame(data_runtimes_default)
    for module in df.columns:
        ax = df[module].plot(kind="density")
        plt.savefig(f"analysis/plots/default_runtime/{module}")
        plt.close(ax.figure)
    default_runtimes = pd.DataFrame(data_runtimes_default).mean()
    tuner_runtimes = pd.DataFrame(data_runtimes_tuner, index=tuners.keys()).transpose()
    margin_of_error = pd.DataFrame(data_margin_of_error, index=tuners.keys()).transpose()
    speedup = (1 / tuner_runtimes).multiply(default_runtimes, axis=0)
    s = speedup.rename(index=shorten_program_name, columns=shorten_tuner_name).style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/evaluation_20_speedup.tex",
        hrules=True,
        label="table:evaluation-20-speedup",
        caption="Speedup of the best optimization setting learned from different tuning methods over \\texttt{-O3}; 500 queries and 20 flags in the search space",
        environment="longtable")
    color_table("analysis/table/evaluation_20_speedup.tex", max)
    print()
    print("Evaluation winners (n=20, budget=500)")
    print(tuner_runtimes.transpose().idxmin().value_counts())


def evaluation_98_comparison():
    # the critical value of the z-distribution to obtain a 95% confidence interval
    z = {5: 2.571, 10: 2.228, 20: 2.086}
    tuners = {
        "RandomTuner": "random",
        "MonoTuner": "mono",
        "SRTuner": "srtuner"}
    data_runtimes_default = []
    data_runtimes_tuner = []
    data_margin_of_error = []
    for tuner, path in tuners.items():
        data = []
        for file in glob.glob(f"evaluation/{path}/n_098_budget_0500_??.csv"):
            df = pd.read_csv(file, index_col=0).transpose()
            data.append(df[tuner])
            data_runtimes_default.append(df["Default"])
        assert data
        df = pd.DataFrame(data)
        data_runtimes_tuner.append(df.mean())
        data_margin_of_error.append(z[len(df.index)] * df.std() / math.sqrt(len(df.index)))
    default_runtimes = pd.DataFrame(data_runtimes_default).mean()
    tuner_runtimes = pd.DataFrame(data_runtimes_tuner, index=tuners.keys()).transpose()
    margin_of_error = pd.DataFrame(data_margin_of_error, index=tuners.keys()).transpose()
    speedup = (1 / tuner_runtimes).multiply(default_runtimes, axis=0)
    s = speedup.rename(index=shorten_program_name, columns=shorten_tuner_name).style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/evaluation_98_speedup.tex",
        hrules=True,
        label="table:evaluation-98-speedup",
        caption="Speedup of the best optimization setting learned from different tuning methods over \\texttt{-O3}; 500 queries and 98 flags in the search space",
        environment="longtable")
    color_table("analysis/table/evaluation_98_speedup.tex", max)
    print()
    print("Evaluation winners (n=98, budget=500)")
    print(tuner_runtimes.transpose().idxmin().value_counts())


def evaluation_low_degree():
    # the critical value of the z-distribution to obtain a 95% confidence interval
    z = {5: 2.571, 10: 2.228, 20: 2.086}
    tuners = {
        "ActiveFourier": "active_fourier",
	    "ActiveLowDegree": "active_fourier_low_degree",
        "BOCS-SDP-l1": "bocs"}
    data_runtimes_default = []
    data_runtimes_tuner = []
    data_margin_of_error = []
    for tuner, path in tuners.items():
        data = []
        for file in glob.glob(f"evaluation/{path}/n_020_budget_0500_??.csv"):
            df = pd.read_csv(file, index_col=0).transpose()
            data.append(df[tuner])
            data_runtimes_default.append(df["Default"])
        assert data
        df = pd.DataFrame(data)
        data_runtimes_tuner.append(df.mean())
        data_margin_of_error.append(z[len(df.index)] * df.std() / math.sqrt(len(df.index)))
    df = pd.DataFrame(data_runtimes_default)
    for module in df.columns:
        ax = df[module].plot(kind="density")
        plt.savefig(f"analysis/plots/default_runtime/{module}")
        plt.close(ax.figure)
    default_runtimes = pd.DataFrame(data_runtimes_default).mean()
    tuner_runtimes = pd.DataFrame(data_runtimes_tuner, index=tuners.keys()).transpose()
    margin_of_error = pd.DataFrame(data_margin_of_error, index=tuners.keys()).transpose()
    speedup = (1 / tuner_runtimes).multiply(default_runtimes, axis=0)
    s = speedup.rename(index=shorten_program_name, columns=shorten_tuner_name).style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/evaluation_low_degree_speedup.tex",
        hrules=True,
        label="table:evaluation-low-degree-speedup",
        caption="Speedup of the best optimization setting learned from different Fourier-sparse set functions over \\texttt{-O3}; 500 queries and 20 flags in the search space",
        environment="longtable")
    color_table("analysis/table/evaluation_low_degree_speedup.tex", max)
    print()
    print("Evaluation winners (n=20, budget=500)")
    print(tuner_runtimes.transpose().idxmin().value_counts())


def simulation_offline_fourier():
    data_success_chance = collections.defaultdict(list)
    data_speedup_learn = collections.defaultdict(list)
    data_speedup_optimal = collections.defaultdict(list)
    for search_space in range(20, 121, 10):
        for budget in range(100, 1001, 100):
            path = f"simulation/noise_0/n_{search_space:03d}_budget_{budget:04d}_00.csv"
            df = pd.read_csv(path, index_col=0).transpose()
            assert len(df.columns) == 4
            assert df.columns[0] == "Default"
            assert df["Default"].min() == df["Default"].max() == 1.0
            assert df.columns[1] == "Train"
            assert df.columns[3] == "Optimal"
            try:
                assert repetitions == len(df.index)
            except NameError:
                repetitions = len(df.index)
            tuner = df.columns[2]
            success_count = (df[tuner] < df["Train"]).sum()
            data_success_chance[search_space].append(success_count / repetitions)
            data_speedup_learn[search_space].append((df["Train"] / df[tuner]).mean())
            data_speedup_optimal[search_space].append((df[tuner] / df["Optimal"]).mean())
    success_chance = pd.DataFrame(data_success_chance, index=range(100, 1001, 100))
    speedup_learn = pd.DataFrame(data_speedup_learn, index=range(100, 1001, 100))
    speedup_optimal = pd.DataFrame(data_speedup_optimal, index=range(100, 1001, 100))

    s = success_chance.style
    s.format(precision=2)
    s.to_latex(
        "analysis/table/simulation_success_chance.tex",
        hrules=True,
        label="table:simulation-success-chance",
        caption="Probability that the minimum of the learned simulated Fourier-sparse set function is better than the best optimization setting from the training data; 100-1000 queries and 20-120 flags in the search space",
        environment="longtable")

    s = speedup_learn.style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/simulation_speedup_learn.tex",
        hrules=True,
        label="table:simulation-speedup-learn",
        caption="Speedup of the minimum of the learned simulated Fourier-sparse set function over the best optimizating setting from the training data; 100-1000 queries and 20-120 flags in the search space",
        environment="longtable")

    s = speedup_optimal.style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/simulation_speedup_optimal.tex",
        hrules=True,
        label="table:simulation-speedup-optimal",
        caption="Speedup of the optimal optimization setting over the minimum of the learned simulated Fourier-sparse set function; 100-1000 queries and 20-120 flags in the search space",
        environment="longtable")


def simulation_noise():
    noise = {
        # 0: "noise_0",
        0.05: "noise_5e-2",
        0.01: "noise_1e-2",
        0.005: "noise_5e-3"}
    for noise, directory in noise.items():
        data_speedup_learn = collections.defaultdict(list)
        for search_space in range(20, 121, 10):
            for budget in range(100, 1001, 100):
                path = f"simulation/{directory}/n_{search_space:03d}_budget_{budget:04d}_00.csv"
                df = pd.read_csv(path, index_col=0).transpose()
                data_speedup_learn[search_space].append((df["Train"] / df["Fourier"]).mean())
        speedup_learn = pd.DataFrame(data_speedup_learn, index=range(100, 1001, 100))

        s = speedup_learn.style
        s.format(precision=3)
        s.to_latex(
            f"analysis/table/simulation_speedup_learn_noise_{noise:.0e}.tex",
            hrules=True,
            label=f"table:simulation-speedup-learn-{noise:.0e}",
            caption=f"Speedup of the minimum of the learned simulated Fourier-sparse set function over the best optimizating setting from the training data; 100-1000 queries with Gaussian noise $\sim \mathcal{{N}}(0,\\,{noise}^2)$ and 20-120 flags in the search space",
            environment="longtable")


def stability_latch():
    before = pd.read_csv(f"stability/latch/stability_00.csv")
    after = pd.read_csv(f"stability/latch/stability_01.csv")
    for module in before.columns:
        ax = before[module].plot()
        plt.savefig(f"analysis/plots/stability/latch_before_{module}")
        after[module].plot()
        plt.savefig(f"analysis/plots/stability/latch_after_{module}")
        plt.close(ax.figure)


def stability_min():
    before = pd.read_csv(f"stability/min/stability_00.csv")
    after = pd.read_csv(f"stability/min/stability_01.csv")
    for module in before.columns:
        ax = before[module].plot()
        plt.savefig(f"analysis/plots/stability/min_before_{module}")
        after[module].plot()
        plt.savefig(f"analysis/plots/stability/min_after_{module}")
        plt.close(ax.figure)


def stability_hopeless():
    df = pd.read_csv(f"stability/hopeless/stability_02.csv")
    for module in df.columns:
        ax = df[module].plot()
        plt.savefig(f"analysis/plots/stability/hopeless_{module}")
        plt.close(ax.figure)


def stability_default():
    default_runtimes = []
    for file in glob.glob(f"evaluation/*/n_???_budget_????_??.csv"):
        df = pd.read_csv(file, index_col=0).transpose()
        default_runtimes.append(df["Default"])
    df = pd.DataFrame(default_runtimes)
    for module in df.columns:
        ax = df[module].plot(kind="density")
        plt.savefig(f"analysis/plots/default_runtime/{module}")
        plt.close(ax.figure)
    data = {"$\\mu$": df.mean(), "$\\sigma$": df.std(), "$\\sigma / \\mu$": df.std() / df.mean()}
    df = pd.DataFrame(data).rename(shorten_program_name)
    s = df.style
    s.to_latex(
        "analysis/table/default_relative_standard_deviation.tex",
        hrules=True,
        label="table:default-relative-standard-deviation",
        caption="Mean, standard deviation, and relative standard deviation of \\texttt{-O3}",
        environment="longtable")
    print()
    print(f"Number of -O3 queries: {len(default_runtimes)}")

validate_score_evaluation_20()
offline_fourier_success_chance()
online_fourier_speedup()
evaluation_low_degree()
evaluation_20_comparison()
evaluation_98_comparison()
simulation_offline_fourier()
simulation_validate_score()
simulation_noise()
stability_latch()
stability_min()
stability_hopeless()
stability_default()
