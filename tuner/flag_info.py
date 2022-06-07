import re
import subprocess
from typing import Union

OptSetting = dict[str, Union[bool, int, str]]
SearchSpace = dict[str, list[Union[bool, int, str]]]


def read_gcc_opts(path: str) -> SearchSpace:
    """Reads the list of gcc optimizations that follow a certain format.

    Due to a slight difference in GCC distributions, the supported flags are
    confirmed by using -fverbose-asm. Each chunk specifies flags supported
    under each standard optimization levels. Besides flags identified by
    -fverbose-asm, we also considered flags in online doc. They are placed as
    the last chunk and considered as last optimization level. (Any standard
    optimization level would not configure them.)
    """
    search_space = {"stdOptLv", [1, 2, 3]}
    with open(path) as fp:
        for raw_line in fp.read().split("\n"):
            if raw_line != "":
                line = raw_line.replace(" ", "").strip()
                if line[0] != "#":
                    tokens = line.split("=")
                    flag_name = tokens[0]
                    # Binary flag
                    if len(tokens) == 1:
                        search_space[flag_name] = [False, True]
                    # Parametric flag
                    else:
                        assert(len(tokens) == 2)
                        search_space[flag_name] = tokens[1].split(",")
    return search_space


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
            configs = match_parametric.group(2).split("|")
            value = match_parametric.group(3)
            if value in configs:
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


def convert_to_str(opt_setting: OptSetting, search_space: SearchSpace) -> str:
    str_opt_setting = f"-O{opt_setting['stdOptLv']}"

    for flag_name, config in opt_setting.items():
        assert config in search_space[flag_name]
        if flag_name == "stdOptLv":
            continue
        if search_space[flag_name] != [False, True]:
            if config != "":
                str_opt_setting += f" {flag_name}={config}"
        else:
            if config:
                str_opt_setting += f" {flag_name}"
            else:
                negated_flag_name = flag_name.replace("-f", "-fno-", 1)
                str_opt_setting += f" {negated_flag_name}"
    return str_opt_setting


if __name__ == "__main__":
    for flag in request_gcc_flags(3)[0]:
        print(flag)
