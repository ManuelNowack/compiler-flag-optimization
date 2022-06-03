import re
import subprocess


class FlagInfo:
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs


class GCCFlagInfo(FlagInfo):
    def __init__(self, name, configs, isParametric, stdOptLv):
        super().__init__(name, configs)
        self.isParametric = isParametric
        self.stdOptLv = stdOptLv


def read_gcc_opts(path):
    """Reads the list of gcc optimizations that follow a certain format.

    Due to a slight difference in GCC distributions, the supported flags are
    confirmed by using -fverbose-asm. Each chunk specifies flags supported
    under each standard optimization levels. Besides flags identified by
    -fverbose-asm, we also considered flags in online doc. They are placed as
    the last chunk and considered as last optimization level. (Any standard
    optimization level would not configure them.)
    """
    search_space = dict()  # pair: flag, configs
    # special case handling
    search_space["stdOptLv"] = GCCFlagInfo(
        name="stdOptLv", configs=[1, 2, 3], isParametric=True, stdOptLv=-1)
    with open(path, "r") as fp:
        stdOptLv = 0
        for raw_line in fp.read().split("\n"):
            # Process current chunk
            if(len(raw_line)):
                line = raw_line.replace(" ", "").strip()
                if line[0] != "#":
                    tokens = line.split("=")
                    flag_name = tokens[0]
                    # Binary flag
                    if len(tokens) == 1:
                        info = GCCFlagInfo(
                            name=flag_name,
                            configs=[False, True],
                            isParametric=False,
                            stdOptLv=stdOptLv)
                    # Parametric flag
                    else:
                        assert(len(tokens) == 2)
                        info = GCCFlagInfo(
                            name=flag_name,
                            configs=tokens[1].split(","),
                            isParametric=True,
                            stdOptLv=stdOptLv)
                    search_space[flag_name] = info
            # Move onto next chunk
            else:
                stdOptLv = stdOptLv + 1
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


def convert_to_str(opt_setting, search_space):
    str_opt_setting = "-O" + str(opt_setting["stdOptLv"])

    for flag_name, config in opt_setting.items():
        assert flag_name in search_space
        flag_info = search_space[flag_name]
        # Parametric flag
        if flag_info.isParametric:
            if flag_info.name != "stdOptLv" and len(config) > 0:
                str_opt_setting += f" {flag_name}={config}"
        # Binary flag
        else:
            assert(isinstance(config, bool))
            if config:
                str_opt_setting += f" {flag_name}"
            else:
                negated_flag_name = flag_name.replace("-f", "-fno-", 1)
                str_opt_setting += f" {negated_flag_name}"
    return str_opt_setting


if __name__ == "__main__":
    for flag in request_gcc_flags(3)[0]:
        print(flag)
