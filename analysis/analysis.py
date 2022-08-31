import glob
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


validate_score_evaluation_20()
