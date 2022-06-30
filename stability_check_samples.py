import random

import pandas as pd

import compiler_opt


search_space = compiler_opt.read_gcc_search_space("gcc_opts.txt")

df = pd.read_csv("samples/10000.csv", index_col=0)
num_rows, num_cols = df.shape

rng = random.Random(42)

noise = []
for _ in range(100):
    row_index = rng.randrange(num_rows)
    col_index = rng.randrange(num_cols)
    program, dataset, command = df.columns[col_index].split(":")
    evaluator = compiler_opt.Evaluator(
        program, 1, search_space, dataset, command)
    flags = df.index[row_index]
    optimization = compiler_opt.str_to_optimization(flags, search_space)
    runtime = evaluator.evaluate(optimization)
    noise.append(abs(1.0 - df.iloc[row_index, col_index] / runtime))
print(noise)
