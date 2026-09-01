"""Run the CSF-enabled FastSPM recipe on a worklist, saving only the mwc3
(CSF) output per subject. GM (mwc1) is not saved here -- it's already been
extracted and consolidated in all_patient_maps/.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--worklist", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--recipe", type=Path, required=True)
parser.add_argument("--image", default="fastbrainage:local")
parser.add_argument("--workers", type=int, default=3)
parser.add_argument("--ids-file", type=Path, default=None)
parser.add_argument("--status-log", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=1800)
parser.add_argument("--sudo-docker", action="store_true", help="prefix docker invocations with sudo (fresh boxes where the docker group hasn't taken effect yet)")
parser.add_argument("--compress-output", action="store_true", help="gzip each CSF map as *_mwc3.nii.gz to reduce batch storage")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

with args.worklist.open(newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if args.ids_file:
    wanted = {l.strip() for l in args.ids_file.read_text().splitlines() if l.strip()}
    rows = [r for r in rows if r["test_id"] in wanted]

RUNNER_SH = """#!/usr/bin/env bash
set -euo pipefail
input_path=$1
out_dir=/data/out
tid=$2
work_dir=$(mktemp -d)
mkdir -p "$out_dir"
if [[ "$input_path" == *.nii.gz ]]; then
    gzip -dc "$input_path" > "$work_dir/input.nii"
else
    cp "$input_path" "$work_dir/input.nii"
fi
octave --quiet --no-gui --eval \
    "addpath('/opt/fastbrainage/matlab'); run_fastspm_batch('$work_dir/input.nii','$work_dir/out','/opt/spm12','/data/recipe/fastspm_v1_gmcsf.m')"
csf="$work_dir/mwc3input.nii"
if [[ ! -f "$csf" ]]; then
    echo "NO mwc3 OUTPUT" >&2
    exit 3
fi
if [[ "__COMPRESS_OUTPUT__" == "1" ]]; then
    gzip -c "$csf" > "$out_dir/${tid}_mwc3.nii.gz"
else
    cp "$csf" "$out_dir/${tid}_mwc3.nii"
fi
rm -rf "$work_dir"
echo "saved ${tid}_mwc3__OUTPUT_SUFFIX__"
"""
# Keep generated runner code with the resumable batch workspace rather than
# writing an untracked helper into the source-tree recipe directory.
runner_path = args.worklist.parent / "_runner.sh"
RUNNER_SH = RUNNER_SH.replace(
    "__COMPRESS_OUTPUT__", "1" if args.compress_output else "0"
).replace(
    "__OUTPUT_SUFFIX__", ".nii.gz" if args.compress_output else ".nii"
)
runner_path.write_text(RUNNER_SH)
runner_path.chmod(0o755)


def process_one(row: dict) -> dict:
    tid = row["test_id"]
    output_suffix = ".nii.gz" if args.compress_output else ".nii"
    out_map = args.out_dir / f"{tid}_mwc3{output_suffix}"
    if out_map.exists() and out_map.stat().st_size:
        return {"test_id": tid, "status": "exists", "seconds": "0"}
    nifti = Path(row["nifti_path"])
    if not nifti.exists():
        return {"test_id": tid, "status": "missing_input", "seconds": "0"}
    started = time.monotonic()
    log_path = args.out_dir / f"{tid}.log"
    try:
        with log_path.open("w") as log:
            docker_cmd = (["sudo", "docker"] if args.sudo_docker else ["docker"])
            result = subprocess.run(
                [
                    *docker_cmd, "run", "--rm", "--cpus=2",
                    "-v", f"{nifti}:/data/input{''.join(nifti.suffixes)}:ro",
                    "-v", f"{args.recipe.parent}:/data/recipe:ro",
                    "-v", f"{args.out_dir}:/data/out",
                    "-v", f"{args.worklist.parent}:/data/work",
                    "--entrypoint", "/bin/bash",
                    args.image,
                    "/data/work/_runner.sh", f"/data/input{''.join(nifti.suffixes)}", tid,
                ],
                stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
            )
    except subprocess.TimeoutExpired:
        return {"test_id": tid, "status": "error_timeout", "seconds": f"{time.monotonic()-started:.1f}"}
    except Exception as exc:
        return {"test_id": tid, "status": f"error_{type(exc).__name__}", "seconds": f"{time.monotonic()-started:.1f}"}
    status = "converted" if result.returncode == 0 and out_map.exists() and out_map.stat().st_size else f"error_rc_{result.returncode}"
    return {"test_id": tid, "status": status, "seconds": f"{time.monotonic()-started:.1f}"}


done = 0
mode = "a" if args.status_log.exists() else "w"
with args.status_log.open(mode) as sh, ThreadPoolExecutor(max_workers=args.workers) as pool:
    if mode == "w":
        sh.write("test_id\tstatus\tseconds\n")
    futures = {pool.submit(process_one, r): r for r in rows}
    for fut in as_completed(futures):
        res = fut.result()
        done += 1
        sh.write(f"{res['test_id']}\t{res['status']}\t{res['seconds']}\n")
        sh.flush()
        print(f"{done}/{len(rows)} {res['test_id']} {res['status']} {res['seconds']}s", flush=True)
