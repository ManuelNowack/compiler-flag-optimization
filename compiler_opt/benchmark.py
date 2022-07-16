import math
import os
import shutil
import sys

import ck.kernel as ck


def ck_cmd(cmd: dict) -> dict:
    current_working_directory = os.getcwd()
    r = ck.access(cmd)
    if r["return"] > 0:
        ck.err(r)
    os.chdir(current_working_directory)
    return r


def compile(
        program: str,
        flags: str,
        generate_rnd_tmp_dir: bool = False) -> str:
    r = ck_cmd({"action": "compile",
                "module_uoa": "program",
                "data_uoa": program,
                "generate_rnd_tmp_dir": "yes" if generate_rnd_tmp_dir else "",
                "flags": flags})
    return r["tmp_dir"]


def get_repeat(program: str, dataset: str, command: str, flags: str) -> str:
    r = ck_cmd({"action": "compile",
                "module_uoa": "program",
                "data_uoa": program,
                "generate_rnd_tmp_dir": "yes",
                "flags": flags})
    tmp_dir = r["tmp_dir"]
    r = ck_cmd({"action": "run",
                "module_uoa": "program",
                "data_uoa": program,
                "tmp_dir": tmp_dir,
                "cmd_key": command,
                "dataset_uoa": dataset,
                "dataset_file": "data.txt" if dataset == "txt-0001" else ""})
    shutil.rmtree(tmp_dir)
    # The repeat value targets an execution time of 4 seconds, we want 1 second
    return str(math.ceil(r["characteristics"]["repeat"] / 4))


def run(
        program: str,
        dataset: str,
        command: str,
        tmp_dir: str = "",
        repeat: str = "") -> float:
    runtimes = []
    for _ in range(5):
        r = ck_cmd({"action": "run",
                    "module_uoa": "program",
                    "data_uoa": program,
                    "tmp_dir": tmp_dir,
                    "cmd_key": command,
                    "dataset_uoa": dataset,
                    "dataset_file": "data.txt" if dataset == "txt-0001" else "",
                    "repeat": repeat})
        try:
            assert r["misc"]["calibration_success"]
            assert r["misc"]["run_success_bool"]
            assert not r["misc"]["output_check_failed_bool"]
            runtimes.append(r["characteristics"]["execution_time"])
        except AssertionError:
            print(
                f"{r['misc']['fail_reason']} at {program}:{dataset}:{command}",
                file=sys.stderr)
    return min(runtimes)
