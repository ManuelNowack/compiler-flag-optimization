import collections
import glob
import matplotlib.pyplot as plt
import pandas as pd

def shorten_program_name(s: str):
    prefix = "cbench-"
    if not s.startswith(prefix):
        raise ValueError("Not a cbench program")
    return s[len(prefix):]


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
        "analysis/table/n_20_validate_scores.tex",
        hrules=True,
        label="table:validate-score",
        caption="$R^2$ score of validation dataset",
        environment="longtable")
    validate_scores

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
    s.format(precision=2)
    s.to_latex(
        "analysis/table/n_20_validate_scores_speedup.tex",
        hrules=True,
        label="table:validate-score-speedup",
        caption="Speedup of the learned flags over the best flags from the training data",
        environment="longtable")
    speedup


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
        caption="Probability that the learned flags are better than the best flags from the training data",
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
        caption="Speedup of online Fourier-sparse function learning over the best flags from randomly sampled training data",
        environment="longtable")


def simulation_offline_fourier():
    data_success_chance = collections.defaultdict(list)
    data_speedup_learn = collections.defaultdict(list)
    data_speedup_optimal = collections.defaultdict(list)
    for search_space in range(20, 121, 10):
        for budget in range(100, 1001, 100):
            path = f"simulation/noise_5e-3/n_{search_space:03d}_budget_{budget:04d}_00.csv"
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
        caption="Probability that the learned flags are better than the best flags from the training data",
        environment="longtable")

    s = speedup_learn.style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/simulation_speedup_learn.tex",
        hrules=True,
        label="table:simulation-speedup-learn",
        caption="Speedup of the learned flags over the best flags from the training data",
        environment="longtable")

    s = speedup_optimal.style
    s.format(precision=3)
    s.to_latex(
        "analysis/table/simulation_speedup_optimal.tex",
        hrules=True,
        label="table:simulation-speedup-optimal",
        caption="Speedup of the optimal flags over the learned flags",
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
    directories = [
        "active_fourier",
    	"active_fourier_low_degree",
    	"bocs",
        "fourier",
        "fourier_low_degree",
        "mono",
        "random",
        "srtuner"]
    default_runtimes = []
    for directory in directories:
        for file in glob.glob(f"evaluation/{directory}/n_020_budget_0500_??.csv"):
            df = pd.read_csv(file, index_col=0).transpose()
            default_runtimes.append(df["Default"])
    df = pd.DataFrame(default_runtimes)
    for module in df.columns:
        ax = df[module].plot(kind="density")
        plt.savefig(f"analysis/plots/default_runtime/{module}")
        plt.close(ax.figure)
    noise = df.std() / df.mean()
    s = noise.to_frame("$\\sigma$").style
    s.to_latex(
        "analysis/table/default_relative_standard_deviation.tex",
        hrules=True,
        label="table:default-relative-standard-deviation",
        caption="Relative standard deviation of \\texttt{-O3}",
        environment="longtable")

validate_score_evaluation_20()
offline_fourier_success_chance()
online_fourier_speedup()
simulation_offline_fourier()
stability_latch()
stability_min()
stability_hopeless()
stability_default()
