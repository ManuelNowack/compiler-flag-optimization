import ck.kernel as ck
import glob
import os


def ck_cmd(cmd: dict):
    current_working_directory = os.getcwd()
    r = ck.access(cmd)
    if r["return"] > 0:
        ck.err(r)
    os.chdir(current_working_directory)
    return r


def extract_flags(path: str):
    """Extracts enabled optimization flags from a GCC assembler code file.

    You can generate the GCC assembler code file by passing the flags
    -save-temps -fverbose-asm.

    Args:
        path: Path to the assembler code file with file name suffix ".s".

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


def run(flags: str, program: str, dataset: str = "") -> float:
    r = ck_cmd({"action": "compile",
                "module_uoa": "program",
                "data_uoa": program,
                "speed": "yes",
                "flags": "-w -save-temps -fverbose-asm " + flags})
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
                "data_uoa": program,
                "dataset_uoa": dataset})
    return r["characteristics"]["execution_time"]
