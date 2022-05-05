import ck.kernel as ck
import glob
import itertools
import operator


def ck_cmd(cmd: dict):
    r = ck.access(cmd)
    if r["return"] > 0:
        ck.err(r)
    return r


def install_dependencies() -> None:
    ck_cmd({"action": "pull",
            "module_uoa": "repo",
            "data_uoa": "mlcommons@ck-mlops"})
    ck_cmd({"action": "pull",
            "module_uoa": "repo",
            "data_uoa": "ctuning-programs"})
    ck_cmd({"action": "pull",
            "module_uoa": "repo",
            "data_uoa": "ctuning-datasets-min"})
    # ck_cmd({"action": "detect",
    #         "module_uoa": "soft",
    #         "data_uoa": "compiler.gcc"})


def extract_flags(path: str):
    """Extracts enabled optimization flags from a GCC assemble code file.

    Args:
        path: Path to the assemble code file with file name suffix ".s".

    Returns:
        A list containing all enabled optimization flags.

    Raises:
        ValueError: Flags are not found in the assembler code file.
    """
    with open(path) as fh:
        flags = []
        extracting = False
        for line in fh:
            if not extracting and line.startswith("# options enabled:  "):
                extracting = True
                flags += line[20:].rstrip().split(" ")
            elif extracting:
                if line.startswith("# "):
                    flags += line[2:].rstrip().split(" ")
                else:
                    for flag in flags:
                        assert flag.startswith("-f") or flag.startswith("-m")
                    return [f for f in flags if f.startswith("-f")]
    raise ValueError("Flags not found in assembler code file " + path)


def negate_flags(flags: list):
    return [flag.replace("-f", "-f-no-") for flag in flags]


def benchmark(flags: list) -> float:
    r = ck_cmd({"action": "compile",
                "module_uoa": "program",
                "data_uoa": "cbench-automotive-susan",
                "speed": "yes",
                "flags": "-w -save-temps -fverbose-asm " + " ".join(flags)})
    # print(r.keys())
    # print(r["return"])
    # print(r["tmp_dir"])
    # print(r["misc"].keys())
    # print(r["characteristics"].keys())
    # print(r["deps"].keys())
    # for file in glob.glob("*.s"):
    #     print(file, extract_flags(file))
    r = ck_cmd({"action": "run",
                "module_uoa": "program",
                "data_uoa": "cbench-automotive-susan",
                "cmd_key": "corners",
                "dataset_uoa": "image-pgm-0001"})
    return r["characteristics"]["execution_time"]


flags = [
    "-fno-inline-atomics",
    "-fno-inline-functions",
    "-fno-inline-functions-called-once",
    "-fno-inline-small-functions"]
powerset = itertools.chain.from_iterable(
    itertools.combinations(flags, i) for i in range(len(flags) + 1))
times = [benchmark(subset) for subset in powerset]
indices = [i for i, _ in sorted(enumerate(times), key=operator.itemgetter(1))]
print(indices)