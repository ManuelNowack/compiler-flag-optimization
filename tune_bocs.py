import benchmark
from BOCS.BOCS import BOCS
from BOCS.sample_models import sample_models
import numpy as np


all_flags = [
    "-fno-gcse-after-reload",
    "-fno-inline-functions",
    "-fno-ipa-cp-clone",
    "-fno-peel-loops",
    "-fno-predictive-commoning",
    "-fno-split-loops",
    "-fno-split-paths",
    "-fno-tree-loop-distribute-patterns",
    "-fno-tree-loop-vectorize",
    "-fno-tree-partial-pre",
    "-fno-tree-slp-vectorize",
    "-fno-unswitch-loops"]

program = "cbench-network-dijkstra"
dataset = "cdataset-dijkstra-0001"


def subset_to_flags(subset: np.ndarray):
    all_flags_np = np.array(all_flags, dtype=object)
    return "-O3 " + " ".join(all_flags_np[subset.astype(bool)])


def f(x):
    y = []
    for subset in x:
        benchmark.compile(program, subset_to_flags(subset))
        y.append(benchmark.run(program, dataset))
    return np.array(y)


# Save inputs in dictionary
inputs = {}
inputs["n_vars"] = len(all_flags)
inputs["evalBudget"] = 50
inputs["n_init"] = 10
inputs["lambda"] = 1e-4

inputs["model"] = f
inputs["penalty"] = lambda x: inputs["lambda"] * np.sum(x, axis=1)

# Generate initial samples for statistical models
inputs["x_vals"] = sample_models(inputs["n_init"], inputs["n_vars"])
inputs["y_vals"] = inputs["model"](inputs["x_vals"])

# Run BOCS-SA and BOCS-SDP (order 2)
(BOCS_SA_model, BOCS_SA_obj) = BOCS(inputs.copy(), 2, "SA")
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, "SDP-l1")
print(BOCS_SA_model)
print(BOCS_SA_obj)

# Compute optimal value found by BOCS
BOCS_SA_flags = subset_to_flags(BOCS_SA_model[BOCS_SA_obj.argmin()])
BOCS_SDP_flags = subset_to_flags(BOCS_SDP_model[BOCS_SDP_obj.argmin()])
BOCS_SA_opt = BOCS_SA_obj.min()
BOCS_SDP_opt = BOCS_SDP_obj.min()

baseline = f(np.zeros((1, inputs["n_vars"])))[0]

for i in range(100):
    try:
        with open(f"results/BOCS_{i:02d}.txt", "x"):
            nonce = i
            break
    except Exception:
        pass
with open(f"results/BOCS_{nonce:02d}.txt", "w") as fh:
    fh.write(f"baseline {baseline}\n")
    fh.write(f"BOCS_SA best {BOCS_SA_opt}\n")
    fh.write(f"BOCS_SDP best {BOCS_SDP_opt}\n")
    fh.write(f"BOCS_SA speedup {baseline / BOCS_SA_opt}\n")
    fh.write(f"BOCS_SDP speedup {baseline / BOCS_SDP_opt}\n")
