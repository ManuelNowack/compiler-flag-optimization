import glob
import os
import re
import requests
import shutil
import subprocess
from typing import Callable

from . import benchmark
from .typing import Optimization, SearchSpace


def read_gcc_search_space(path: str) -> SearchSpace:
    """Reads the list of gcc optimizations that follow a certain format.

    Due to a slight difference in GCC distributions, the supported flags are
    confirmed by using -fverbose-asm. Each chunk specifies flags supported
    under each standard optimization levels. Besides flags identified by
    -fverbose-asm, we also considered flags in online doc. They are placed as
    the last chunk and considered as last optimization level. (Any standard
    optimization level would not configure them.)
    """
    search_space = {"stdOptLv": (1, 2, 3)}
    with open(path) as fp:
        for raw_line in fp.read().split("\n"):
            if raw_line != "":
                line = raw_line.replace(" ", "").strip()
                if line[0] != "#":
                    tokens = line.split("=")
                    flag_name = tokens[0]
                    # Binary flag
                    if len(tokens) == 1:
                        search_space[flag_name] = (False, True)
                    # Parametric flag
                    else:
                        assert(len(tokens) == 2)
                        search_space[flag_name] = tuple(tokens[1].split(","))
    return search_space


def extract_gcc_flags(path: str) -> list[str]:
    """Extracts enabled optimization flags from a GCC assembler code file.

    You can generate the GCC assembler code file by passing the flags
    -save-temps -fverbose-asm during compilation.

    The file is parsed according to the observed output of GCC 9.4.0.

    Args:
        path: Path to the assembler code file with file name suffix ".s".

    Returns:
        A list containing all enabled optimization flags.

    Raises:
        ValueError: Flags are not found in the assembler code file. Typically
        indicates the read file is not a proper GCC assembler code file.
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
    raise ValueError(f"Flags not found in assembler code file {path}")


def read_gcc_flags(program: str, flags: str) -> list[str]:
    flags += " -w -save-temps -fverbose-asm"
    tmp_dir = benchmark.compile(program, flags, True)
    actual_flags_prev = None
    for file in glob.glob(os.path.join(tmp_dir, "*.s")):
        actual_flags = set(extract_gcc_flags(file))
        if actual_flags_prev is not None:
            assert actual_flags == actual_flags_prev
        actual_flags_prev = actual_flags
    shutil.rmtree(tmp_dir)
    return sorted(actual_flags)


def request_gcc_flags_online() -> list[str]:
    url = "https://gcc.gnu.org/onlinedocs/gcc-9.5.0/gcc/Optimize-Options.html"
    resp = requests.get(url)
    r = re.compile(r"<dt><code>(-f[^=]+)(=.+)?</code></dt>")
    flags = set()
    for line in resp.text.split("\n"):
        m = r.match(line)
        if m:
            if m.group(2) and not m.group(1) in flags:
                flags.add(m.group(1) + "=")
            else:
                assert m.group(1) + "=" not in flags
                flags.add(m.group(1))
    return sorted(flags)


def request_gcc_flags(opt_level: int):
    if opt_level < 0 or opt_level > 3:
        raise ValueError("Invalid optimization level")
    r = subprocess.run(["gcc", f"-O{opt_level}", "-Q", "--help=optimizers"],
                       capture_output=True, text=True)
    r.check_returncode()
    lines = [line.strip() for line in r.stdout.split("\n") if line != ""]
    lines.remove("The following options control optimizations:")
    re_opt_level = re.compile(r"-O.+")
    re_binary = re.compile(r"(-f\S+)\s+\[(enabled|disabled)\]")
    re_binary_no_default = re.compile(r"(-f[^=\s]+)$")
    re_parametric = re.compile(r"(-f[^=]+)=\[([^\]]+)\]\s+(.+)")
    re_numeric = re.compile(r"(-f[^=]+)=\s+(.+)")
    enabled_flags = []
    disabled_flags = []
    ignored_flags = []
    unknown_options = []
    for line in lines:
        match_opt_level = re_opt_level.match(line)
        match_binary = re_binary.match(line)
        match_binary_no_default = re_binary_no_default.match(line)
        match_parametric = re_parametric.match(line)
        match_numeric = re_numeric.match(line)
        if match_opt_level is not None:
            pass
        elif match_binary is not None:
            flag_name = match_binary.group(1)
            value = match_binary.group(2)
            if value == "enabled":
                enabled_flags.append(flag_name)
            else:
                disabled_flags.append(flag_name)
        elif match_binary_no_default:
            flag_name = match_binary_no_default.group(1)
            ignored_flags.append(flag_name)
        elif match_parametric is not None:
            flag_name = match_parametric.group(1)
            domain = match_parametric.group(2).split("|")
            value = match_parametric.group(3)
            if value in domain:
                enabled_flags.append(flag_name + "=" + value)
            else:
                unknown_options.append(line)
        elif match_numeric is not None:
            flag_name = match_numeric.group(1)
            value = match_numeric.group(2)
            if flag_name not in enabled_flags:
                enabled_flags.append(flag_name + "=" + value)
        else:
            unknown_options.append(line)
    return enabled_flags, disabled_flags, ignored_flags, unknown_options


def request_gcc_search_space() -> SearchSpace:
    r = subprocess.run(["gcc", "-Q", "--help=optimizers"],
                       capture_output=True, text=True)
    r.check_returncode()
    lines = [line.strip() for line in r.stdout.split("\n") if line != ""]
    lines.remove("The following options control optimizations:")
    re_opt_level = re.compile(r"-O.+")
    re_binary = re.compile(r"(-f\S+)\s+\[(enabled|disabled)\]")
    re_binary_no_default = re.compile(r"(-f[^=\s]+)$")
    re_parametric = re.compile(r"(-f[^=]+)=\[([^\]]+)\]\s+(.+)")
    re_numeric = re.compile(r"(-f[^=]+)=\s+(.+)")
    search_space = {"stdOptLv": (1, 2, 3)}
    ignored_flags = []
    unknown_options = []
    for line in lines:
        match_opt_level = re_opt_level.match(line)
        match_binary = re_binary.match(line)
        match_binary_no_default = re_binary_no_default.match(line)
        match_parametric = re_parametric.match(line)
        match_numeric = re_numeric.match(line)
        if match_opt_level is not None:
            pass
        elif match_binary is not None:
            flag_name = match_binary.group(1)
            search_space[flag_name] = (False, True)
        elif match_binary_no_default:
            flag_name = match_binary_no_default.group(1)
            ignored_flags.append(flag_name)
        elif match_parametric is not None:
            flag_name = match_parametric.group(1)
            domain = match_parametric.group(2).split("|")
            search_space[flag_name] = tuple(domain)
        elif match_numeric is not None:
            if flag_name not in search_space:
                ignored_flags.append(flag_name)
        else:
            unknown_options.append(line)
    ignored_flags.remove("-ffast-math")
    ignored_flags.remove("-fhandle-exceptions")
    ignored_flags.remove("-flive-patching")
    ignored_flags.remove("-ftree-vectorize")
    if ignored_flags:
        raise ValueError("Unexpected flags " + ignored_flags)
    # TODO unknown_options
    return search_space


def flatten_search_space(search_space: SearchSpace) -> list[str]:
    flat_search_space = []
    for flag_name, domain in search_space.items():
        if flag_name == "stdOptLv":
            continue
        if domain == (False, True):
            flat_search_space.append(flag_name)
        else:
            for value in domain:
                flat_search_space.append(f"{flag_name}={value}")
    return flat_search_space


def optimization_to_str(optimization: Optimization,
                        search_space: SearchSpace) -> str:
    flags_str = f"-O{optimization['stdOptLv']}"

    for flag_name, value in optimization.items():
        assert value in search_space[flag_name]
        if flag_name == "stdOptLv":
            continue
        if search_space[flag_name] != (False, True):
            if value != "":
                flags_str += f" {flag_name}={value}"
        else:
            if value:
                flags_str += f" {flag_name}"
            else:
                negated_flag_name = flag_name.replace("-f", "-fno-", 1)
                flags_str += f" {negated_flag_name}"
    return flags_str

def str_to_optimization(flags_str: str, search_space: SearchSpace) -> str:
    flags = [flag for flag in flags_str.split(" ") if flag != ""]
    optimization = {}
    for flag in flags:
        try:
            flag_name, value = flag.split("=")
        except ValueError:
            if flag.startswith("-O"):
                flag_name = "stdOptLv"
                value = int(flag[2:])
            elif flag.startswith("-fno-"):
                flag_name = flag.replace("-fno-", "-f", 1)
                value = False
            else:
                flag_name = flag
                value = True
        assert value in search_space[flag_name]
        optimization[flag_name] = value
    return optimization


def write_gcc_search_space(path: str, search_space: SearchSpace,
                           get_gcc_flags: Callable[[int], list[str]]) -> None:
    all_flags_0 = get_gcc_flags(0)
    all_flags_1 = get_gcc_flags(1)
    all_flags_2 = get_gcc_flags(2)
    all_flags_3 = get_gcc_flags(3)
    active_flags_0 = sorted(set(all_flags_0))
    active_flags_1 = sorted(set(all_flags_1) - set(all_flags_0))
    active_flags_2 = sorted(set(all_flags_2) - set(all_flags_1))
    active_flags_3 = sorted(set(all_flags_3) - set(all_flags_2))
    active_flags_all = (active_flags_0 + active_flags_1 + active_flags_2
                        + active_flags_3)
    flat_search_space = flatten_search_space(search_space)
    extra_flags = sorted(set(flat_search_space) - set(active_flags_all))
    with open(path, "w") as fh:
        fh.write("# O0\n")
        fh.writelines(map(lambda x: x + "\n", active_flags_0))
        fh.write("\n# O1\n")
        fh.writelines(map(lambda x: x + "\n", active_flags_1))
        fh.write("\n# O2\n")
        fh.writelines(map(lambda x: x + "\n", active_flags_2))
        fh.write("\n# O3\n")
        fh.writelines(map(lambda x: x + "\n", active_flags_3))
        fh.write("\n# Additional optimizations\n")
        fh.writelines(map(lambda x: x + "\n", extra_flags))


if __name__ == "__main__":
    search_space = request_gcc_search_space()
    write_gcc_search_space(
        "gcc_search_space_help.txt",
        search_space,
        lambda x: request_gcc_flags(x)[0])
    write_gcc_search_space(
        "gcc_search_space_asm.txt",
        search_space,
        lambda x: read_gcc_flags("cbench-network-dijkstra", f"-O{x}"))
