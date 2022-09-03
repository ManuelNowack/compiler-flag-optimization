import pandas as pd

import compiler_opt

search_space_path = "gcc_flags/search_space_98.txt"
num_samples = 100

search_space = compiler_opt.read_gcc_search_space(search_space_path)
samples_path = f"samples/10000_{len(search_space) - 1}.csv"
samples_verify_path = f"samples/{num_samples}_{len(search_space) - 1}.csv"

samples = pd.read_csv(samples_path, index_col=0).sample(num_samples)
compiler_opt.Samples(
    list(samples.columns),
    search_space,
    list(samples.index),
    len(samples.columns))
samples_verify = pd.read_csv(samples_verify_path, index_col=0)
noise = (1.0 - samples / samples_verify).abs()
max_noise = noise.max()

for i in range(100):
    try:
        with open(f"results/stability_samples_{i:02d}.csv", "x"):
            nonce = i
            break
    except Exception:
        pass
noise.to_csv(f"results/stability_samples_{nonce:02d}.csv")
with open(f"results/samples_{nonce:02d}.txt", "w") as fh:
    fh.write(max_noise.to_string(index=False))
