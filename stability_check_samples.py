import random

import pandas as pd

import compiler_opt


search_space_path = "gcc_flags/search_space_98.txt"
search_space = compiler_opt.read_gcc_search_space(search_space_path)

df = pd.read_csv("samples/10000_98.csv", index_col=0)
num_rows, num_cols = df.shape

rng = random.Random(42)

samples = df.sample(n=5, random_state=42)


def f(x: pd.Series):
    program, dataset, command = x.name.split(":")
    evaluator = compiler_opt.Evaluator(program, dataset, command, search_space)
    transformation = []
    for flags, runtime in x.iteritems():
        optimization = compiler_opt.str_to_optimization(flags, search_space)
        runtime_rerun = evaluator.evaluate(optimization)
        noise = 1.0 - runtime / runtime_rerun
        transformation.append(noise)
    return transformation


samples.apply(f)
for i in range(100):
    try:
        with open(f"results/samples_{i:02d}.csv", "x"):
            nonce = i
            break
    except Exception:
        pass
samples.to_csv(f"results/samples_{nonce:02d}.csv")
with open(f"results/samples_{nonce:02d}.txt", "w") as fh:
    fh.write(samples.to_string(index=False))
