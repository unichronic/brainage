#!/usr/bin/env python3
"""Download and process the public portion of the exp2 CSF manifest in batches.

The controller is intentionally resumable: completed ``*_mwc3.nii.gz`` files
are never rerun, completed raw batches can be discarded, and an interrupted
batch can be restarted without changing the fixed manifest order. When the
stage manifest contains a valid ``raw_path`` from the main Exp2 download
stage, that file is reused instead of downloading a second copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


PUBLIC_STATUSES = {"available", "available_via_mirror", "candidate_openneuro_path"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_filename(test_id: str, url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_id).strip("._") or "subject"
    digest = hashlib.sha1(test_id.encode()).hexdigest()[:10]
    remote_name = Path(urlparse(url).path).name
    suffix = ".nii.gz" if remote_name.endswith(".nii.gz") else ".nii"
    return f"{stem}_{digest}{suffix}"


def output_path(output_dir: Path, test_id: str) -> Path:
    return output_dir / f"{test_id}_mwc3.nii.gz"


def download_one(row: dict[str, str], batch_dir: Path) -> tuple[str, Path, str]:
    test_id = row["participant_id"]
    destination = batch_dir / safe_filename(test_id, row["download_url"])
    if destination.exists() and destination.stat().st_size > 0:
        return test_id, destination, "exists"

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "1200",
        "-o",
        str(partial),
        row["download_url"],
    ]
    last_detail = ""
    for attempt in range(3):
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and partial.exists() and partial.stat().st_size > 0:
            os.replace(partial, destination)
            return test_id, destination, "downloaded"
        partial.unlink(missing_ok=True)
        last_detail = result.stderr.strip().replace("\n", " ")[-300:]
        if "404" in last_detail or attempt == 2:
            break
        time.sleep(5 * (attempt + 1))
    return test_id, destination, f"error:{last_detail or result.returncode}"


def acquire_one(row: dict[str, str], batch_dir: Path) -> tuple[str, Path, str]:
    existing = row.get("raw_path", "").strip()
    if existing:
        path = Path(existing)
        if path.is_file() and path.stat().st_size > 0:
            return row["participant_id"], path, "existing"
    return download_one(row, batch_dir)


def write_worklist(path: Path, rows: list[dict[str, str]], downloaded: dict[str, Path], batch_name: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["test_id", "batch", "nifti_path", "age"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "test_id": row["participant_id"],
                    "batch": batch_name,
                    "nifti_path": str(downloaded[row["participant_id"]]),
                    "age": row["age"],
                }
            )


def append_batch_status(path: Path, values: list[str]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        if new_file:
            handle.write("batch\tstatus\tn_rows\tdownload_ok\textracted_ok\tseconds\tmessage\n")
        handle.write("\t".join(values) + "\n")


def process_batch(
    batch_index: int,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    controller_log,
) -> bool:
    started = time.monotonic()
    batch_name = f"exp2_{batch_index:04d}"
    batch_dir = args.input_root / f"batch_{batch_index:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not output_path(args.output_dir, row["participant_id"]).exists()]
    if not pending:
        append_batch_status(args.status_log, [batch_name, "exists", str(len(rows)), "0", str(len(rows)), "0.0", "all outputs already present"])
        controller_log.write(f"{batch_name}: all {len(rows)} outputs already present\n")
        controller_log.flush()
        return True

    downloaded: dict[str, Path] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(acquire_one, row, batch_dir): row for row in pending}
        for future in as_completed(futures):
            test_id, path, status = future.result()
            if status.startswith("error:"):
                failures.append(f"{test_id}={status}")
            else:
                downloaded[test_id] = path
    if failures:
        message = "download failures: " + "; ".join(failures[:5])
        append_batch_status(args.status_log, [batch_name, "download_failed", str(len(rows)), str(len(downloaded)), "0", f"{time.monotonic() - started:.1f}", message])
        controller_log.write(f"{batch_name}: {message}\n")
        controller_log.flush()
        return False

    worklist = args.work_root / f"worklist_{batch_index:04d}.tsv"
    write_worklist(worklist, pending, downloaded, batch_name)
    batch_log = args.work_root / f"extract_{batch_index:04d}.log"
    command = [
        sys.executable,
        str(args.extractor),
        "--worklist",
        str(worklist),
        "--out-dir",
        str(args.output_dir),
        "--recipe",
        str(args.recipe),
        "--image",
        args.image,
        "--workers",
        str(args.workers),
        "--status-log",
        str(args.extract_status_log),
        "--timeout",
        str(args.timeout),
        "--compress-output",
    ]
    if args.sudo_docker:
        command.append("--sudo-docker")
    with batch_log.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)

    missing = [
        row["participant_id"]
        for row in pending
        if not output_path(args.output_dir, row["participant_id"]).exists()
    ]
    if result.returncode != 0 or missing:
        message = f"extract_rc={result.returncode}; missing={','.join(missing[:5])}"
        append_batch_status(args.status_log, [batch_name, "extract_failed", str(len(rows)), str(len(downloaded)), str(len(pending) - len(missing)), f"{time.monotonic() - started:.1f}", message])
        controller_log.write(f"{batch_name}: {message}; see {batch_log}\n")
        controller_log.flush()
        return False

    shutil.rmtree(batch_dir)
    append_batch_status(args.status_log, [batch_name, "completed", str(len(rows)), str(len(downloaded)), str(len(pending)), f"{time.monotonic() - started:.1f}", "raw batch removed after output validation"])
    controller_log.write(f"{batch_name}: completed {len(pending)}; raw batch removed; output_count={len(list(args.output_dir.glob('*_mwc3.nii.gz')))}\n")
    controller_log.flush()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--image", default="fastbrainage:gcp-amd64")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--status-log", type=Path, required=True)
    parser.add_argument("--extract-status-log", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all remaining batches")
    parser.add_argument("--source-split", default="", help="process only rows with this manifest source_split")
    parser.add_argument("--exclude-source-split", default="", help="skip rows with this manifest source_split")
    parser.add_argument("--row-start", type=int, default=None, help="after source filtering, process only rows in [start, end)")
    parser.add_argument("--row-end", type=int, default=None, help="exclusive end for --row-start")
    parser.add_argument("--exclude-row-start", type=int, default=None, help="after source filtering, exclude rows in [start, end)")
    parser.add_argument("--exclude-row-end", type=int, default=None, help="exclusive end for --exclude-row-start")
    parser.add_argument(
        "--exclude-row-range",
        action="append",
        nargs=2,
        type=int,
        default=[],
        metavar=("START", "END"),
        help="after source filtering, exclude an additional row range; may be repeated",
    )
    parser.add_argument("--sudo-docker", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.input_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in read_tsv(args.manifest)
        if row.get("download_url") and row.get("download_status") in PUBLIC_STATUSES
    ]
    if args.source_split:
        rows = [row for row in rows if row.get("source_split") == args.source_split]
    if args.exclude_source_split:
        rows = [row for row in rows if row.get("source_split") != args.exclude_source_split]
    exclude_ranges = list(args.exclude_row_range)
    if (args.exclude_row_start is None) != (args.exclude_row_end is None):
        raise SystemExit("--exclude-row-start and --exclude-row-end must be provided together")
    if args.exclude_row_start is not None:
        exclude_ranges.append([args.exclude_row_start, args.exclude_row_end])
    normalized_ranges = []
    for start, end in sorted(exclude_ranges):
        if not 0 <= start <= end <= len(rows) or start == end:
            raise SystemExit("invalid exclusion row range")
        if normalized_ranges and start < normalized_ranges[-1][1]:
            raise SystemExit("exclusion row ranges must not overlap")
        normalized_ranges.append([start, end])
    if normalized_ranges:
        rows = [
            row
            for index, row in enumerate(rows)
            if not any(start <= index < end for start, end in normalized_ranges)
        ]
    if (args.row_start is None) != (args.row_end is None):
        raise SystemExit("--row-start and --row-end must be provided together")
    if args.row_start is not None:
        if not 0 <= args.row_start <= args.row_end <= len(rows):
            raise SystemExit("invalid --row-start/--row-end range")
        rows = rows[args.row_start : args.row_end]
    ids = [row["participant_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("public exp2 rows contain duplicate participant_id values")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    total_batches = math.ceil(len(rows) / args.batch_size)
    args.status_log.parent.mkdir(parents=True, exist_ok=True)
    args.extract_status_log.parent.mkdir(parents=True, exist_ok=True)
    controller_log_path = args.work_root / "controller.log"
    with controller_log_path.open("a") as controller_log:
        controller_log.write(f"starting public_rows={len(rows)} total_batches={total_batches} start={args.start_batch}\n")
        controller_log.flush()
        end = total_batches if args.max_batches <= 0 else min(total_batches, args.start_batch + args.max_batches)
        for batch_index in range(args.start_batch, end):
            batch_rows = rows[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
            if not process_batch(batch_index, batch_rows, args, controller_log):
                raise SystemExit(2)
        controller_log.write(f"finished through batch={end - 1}; output_count={len(list(args.output_dir.glob('*_mwc3.nii.gz')))}\n")


if __name__ == "__main__":
    main()
