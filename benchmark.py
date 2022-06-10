import ck.kernel as ck
import glob
import os
from tuner.flag_info import extract_gcc_flags


def ck_cmd(cmd: dict):
    current_working_directory = os.getcwd()
    r = ck.access(cmd)
    if r["return"] > 0:
        ck.err(r)
    os.chdir(current_working_directory)
    return r


def compile(program: str, flags: str, lflags: str = "",
            generate_rnd_tmp_dir: bool = False) -> str:
    r = ck_cmd({"action": "compile",
                "module_uoa": "program",
                "data_uoa": program,
                "generate_rnd_tmp_dir": "yes" if generate_rnd_tmp_dir else "",
                "flags": flags + " -save-temps -fverbose-asm",
                "lflags": lflags})
    expected_flags = {f for f in flags.split()
                      if f.startswith("-f") and not f.startswith("-fno-")}
    for file in glob.glob(os.path.join(r["tmp_dir"], "*.s")):
        actual_flags = set(extract_gcc_flags(file))
        missing_flags = expected_flags - actual_flags
        unexpected_flags = actual_flags - expected_flags
        if missing_flags:
            print("Missing flags", sorted(missing_flags))
        if unexpected_flags:
            print("Unexpected flags", sorted(unexpected_flags))
    return r["tmp_dir"]


def run(program: str, dataset: str = "", command: str = "", tmp_dir: str = "") -> float:
    r = ck_cmd({"action": "run",
                "module_uoa": "program",
                "data_uoa": program,
                "tmp_dir": tmp_dir,
                "cmd_key": command,
                "dataset_uoa": dataset})
    if not r["misc"]["run_success_bool"]:
        raise RuntimeError(
            f"{r['misc']['fail_reason']} at {program},{dataset},{command}")
    return r["characteristics"]["execution_time"]
