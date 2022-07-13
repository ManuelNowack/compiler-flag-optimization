import os
import sys

import ck.kernel as ck


def ck_cmd(cmd: dict) -> dict:
    current_working_directory = os.getcwd()
    r = ck.access(cmd)
    if r["return"] > 0:
        ck.err(r)
    os.chdir(current_working_directory)
    return r


def compile(program: str,
            flags: str,
            generate_rnd_tmp_dir: bool = False) -> str:
    r = ck_cmd({"action": "compile",
                "module_uoa": "program",
                "data_uoa": program,
                "generate_rnd_tmp_dir": "yes" if generate_rnd_tmp_dir else "",
                "flags": flags})
    return r["tmp_dir"]


def run(program: str, dataset: str = "", command: str = "",
        tmp_dir: str = "") -> float:
    for _ in range(10):
        r = ck_cmd({"action": "run",
                    "module_uoa": "program",
                    "data_uoa": program,
                    "tmp_dir": tmp_dir,
                    "cmd_key": command,
                    "dataset_uoa": dataset,
                    "dataset_file": "data.txt" if dataset == "txt-0001" else ""})
        if r["misc"]["run_success_bool"]:
            return r["characteristics"]["execution_time"]
        print(f"{r['misc']['fail_reason']} at {program}:{dataset}:{command}",
              file=sys.stderr)
    raise RuntimeError(
        f"{r['misc']['fail_reason']} at {program}:{dataset}:{command}")
