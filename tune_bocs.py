import benchmark
from BOCS.BOCS import BOCS
import numpy as np


all_flags = [
    # O1
    "-fno-branch-count-reg",
    "-fno-combine-stack-adjustments",
    "-fno-compare-elim",
    "-fno-cprop-registers",
    "-fno-defer-pop",
    "-fno-forward-propagate",
    "-fno-guess-branch-probability",
    "-fno-if-conversion",
    "-fno-if-conversion2",
    "-fno-inline",
    "-fno-inline-functions-called-once",
    "-fno-ipa-profile",
    "-fno-ipa-pure-const",
    "-fno-ipa-reference",
    "-fno-merge-constants",
    "-fno-move-loop-invariants",
    "-fno-omit-frame-pointer",
    "-fno-reorder-blocks",
    "-fno-shrink-wrap",
    "-fno-split-wide-types",
    "-fno-ssa-phiopt",
    "-fno-toplevel-reorder",
    "-fno-tree-bit-ccp",
    "-fno-tree-builtin-call-dce",
    "-fno-tree-ccp",
    "-fno-tree-ch",
    "-fno-tree-coalesce-vars",
    "-fno-tree-copy-prop",
    "-fno-tree-dce",
    "-fno-tree-dominator-opts",
    "-fno-tree-dse",
    "-fno-tree-fre",
    "-fno-tree-pta",
    "-fno-tree-sink",
    "-fno-tree-slsr",
    "-fno-tree-sra",
    "-fno-tree-ter",
    #O2
    "-fno-align-labels",
    "-fno-caller-saves",
    "-fno-code-hoisting",
    "-fno-crossjumping",
    "-fno-cse-follow-jumps",
    "-fno-devirtualize",
    "-fno-devirtualize-speculatively",
    "-fno-expensive-optimizations",
    "-fno-gcse",
    "-fno-hoist-adjacent-loads",
    "-fno-indirect-inlining",
    "-fno-inline-small-functions",
    "-fno-ipa-bit-cp",
    "-fno-ipa-cp",
    "-fno-ipa-icf",
    "-fno-ipa-icf-functions",
    "-fno-ipa-icf-variables",
    "-fno-ipa-ra",
    "-fno-ipa-sra",
    "-fno-ipa-vrp",
    "-fno-isolate-erroneous-paths-dereference",
    "-fno-lra-remat",
    "-fno-optimize-sibling-calls",
    "-fno-optimize-strlen",
    "-fno-partial-inlining",
    "-fno-peephole2",
    "-fno-ree",
    "-fno-reorder-functions",
    "-fno-rerun-cse-after-loop",
    "-fno-schedule-insns2",
    "-fno-store-merging",
    "-fno-strict-aliasing",
    "-fno-strict-overflow",
    "-fno-thread-jumps",
    "-fno-tree-pre",
    "-fno-tree-switch-conversion",
    "-fno-tree-tail-merge",
    "-fno-tree-vrp",
    # O3
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
rng = np.random.default_rng()
inputs["x_vals"] = rng.random((inputs["n_init"], inputs["n_vars"])).round()
inputs["y_vals"] = inputs["model"](inputs["x_vals"])

# Run BOCS-SA and BOCS-SDP (order 2)
(BOCS_SA_model, BOCS_SA_obj) = BOCS(inputs.copy(), 2, "SA")
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, "SDP-l1")

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
