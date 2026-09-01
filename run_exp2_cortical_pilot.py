#!/usr/bin/env python3
"""Resumable, label-blind cortical-thickness pilot for the exp2 splits.

Default methods are:
  1. CortexODE on an ADNI-style 1 mm conformed/intensity-adapted T1;
  2. DL+DiReCT segmentation-only front end;
  3. CortexMorph regional thickness on that segmentation.

Only the resolved acquisition manifest is read.  Ages and base-model
predictions are deliberately not loaded.  Per-subject raw/intermediate files
live below this script's own ``work/`` directory and are removed after all
requested methods succeed, leaving compact TSV results and logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DL_DIRECT_DIR = Path(os.environ.get("BRAINAGE_DL_DIRECT_DIR", str(PROJECT / "DL-DiReCT")))
PILOT_DIR = ROOT / "cortical_thickness_pilot"
DEFAULT_MANIFEST = PILOT_DIR / "exp2_resolved_raw_manifest.tsv"
RESULTS_DIR = Path(os.environ.get("BRAINAGE_CORTICAL_RESULTS_DIR", str(PILOT_DIR / "results")))
LOG_DIR = Path(os.environ.get("BRAINAGE_CORTICAL_LOG_DIR", str(PILOT_DIR / "logs")))
WORK_DIR = Path(os.environ.get("BRAINAGE_CORTICAL_WORK_DIR", str(PILOT_DIR / "work")))
ODE_SCRIPT = ROOT / "run_cortexode_subject.py"
MORPH_SCRIPT = ROOT / "run_cortexmorph_subject.py"
USER_AGENT = "brainage-exp2-cortical-pilot/1.0"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def upsert_tsv(path: Path, key: str, record: dict[str, object]) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        existing = read_tsv(path)
    key_value = str(record[key])
    replaced = False
    for index, row in enumerate(existing):
        if row.get(key) == key_value:
            existing[index] = {**row, **{k: str(v) for k, v in record.items()}}
            replaced = True
            break
    if not replaced:
        existing.append({k: str(v) for k, v in record.items()})

    fields: list[str] = []
    for row in existing:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(existing)
    os.replace(temp, path)


def slugify(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    return result.strip("_")[:180]


def raw_name(row: dict[str, str]) -> str:
    declared = row.get("raw_filename", "") or ""
    if declared:
        return unquote(declared)
    return Path(unquote(urlparse(row["raw_url"]).path)).name or "input.nii.gz"


def download_raw(row: dict[str, str], destination: Path) -> None:
    url = row["raw_url"]
    expected = row.get("expected_bytes", "") or ""
    expected_int = int(float(expected)) if expected else None
    if destination.is_file() and destination.stat().st_size > 0:
        if expected_int is None or destination.stat().st_size == expected_int:
            return
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            size = partial.stat().st_size
            if size == 0:
                raise RuntimeError("zero-byte download")
            if expected_int is not None and size != expected_int:
                raise RuntimeError(f"size mismatch: downloaded={size}, expected={expected_int}")
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.HTTPError, RuntimeError) as error:
            last_error = error
            if partial.exists():
                partial.unlink()
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last_error}")


def run_command(command: list[str], cwd: Path, log_path: Path, timeout: int = 3600) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
            return completed.returncode, time.monotonic() - started
        except subprocess.TimeoutExpired:
            log.write(f"TIMEOUT after {timeout} seconds\n")
            return 124, time.monotonic() - started


def prepare_ode_input(conformed_path: Path, ode_dir: Path) -> None:
    image = nib.load(str(conformed_path))
    if tuple(image.shape[:3]) != (256, 256, 256):
        image = nib_processing.conform(
            image,
            out_shape=(256, 256, 256),
            voxel_size=(1.0, 1.0, 1.0),
            orientation="LIA",
        )
    values = np.asarray(image.get_fdata(), dtype=np.float32)
    values[~np.isfinite(values)] = 0
    values[values < 1] = 0
    positive = values[values > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    values = np.clip(values / scale * 255.0, 0, 255).astype(np.float32)
    destination = ode_dir / "mri" / "orig.mgz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(values, np.eye(4)), str(destination))


def flatten_ode_json(path: Path, row: dict[str, str]) -> dict[str, object]:
    payload = json.loads(path.read_text())
    record: dict[str, object] = {
        "split": row["split"],
        "participant_id": row["participant_id"],
        "source_cohort": row["source_cohort"],
        "source_dataset_id": row["source_dataset_id"],
        "raw_url": row["raw_url"],
        "method_status": payload.get("status", "error"),
    }
    for hemisphere in ("lh", "rh"):
        for name, value in payload.get(hemisphere, {}).items():
            record[f"{hemisphere}_{name}"] = value
    record["elapsed_sec"] = payload.get("elapsed_sec", "")
    return record


def flatten_morph_csv(path: Path, row: dict[str, str]) -> dict[str, object]:
    import pandas as pd

    stats = pd.read_csv(path, index_col=0)
    if len(stats) != 1:
        raise RuntimeError(f"expected one CortexMorph row, found {len(stats)}")
    values = stats.iloc[0].to_dict()
    record: dict[str, object] = {
        "split": row["split"],
        "participant_id": row["participant_id"],
        "source_cohort": row["source_cohort"],
        "source_dataset_id": row["source_dataset_id"],
        "raw_url": row["raw_url"],
        "method_status": "ok",
    }
    for key, value in values.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            record[key] = ""
        else:
            record[key] = value
    return record


def status_record(row: dict[str, str]) -> dict[str, object]:
    return {
        "split": row["split"],
        "participant_id": row["participant_id"],
        "source_cohort": row["source_cohort"],
        "source_dataset_id": row["source_dataset_id"],
        "source_subject_id": row["source_subject_id"],
        "raw_url": row["raw_url"],
        "raw_filename": row.get("raw_filename", ""),
        "download_status": "pending",
        "cortexode_status": "not_run",
        "dldirect_status": "not_run",
        "cortexmorph_status": "not_run",
        "error": "",
    }


def existing_status(path: Path, participant_id: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    for row in read_tsv(path):
        if row.get("participant_id") == participant_id:
            return row
    return None


def main() -> None:
    global DL_DIRECT_DIR, RESULTS_DIR, LOG_DIR, WORK_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only-participant")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("cortexode", "dldirect", "cortexmorph"),
        default=["cortexode", "dldirect", "cortexmorph"],
    )
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--dl-direct-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.dl_direct_dir is not None:
        DL_DIRECT_DIR = args.dl_direct_dir.resolve()
    if args.results_dir is not None:
        RESULTS_DIR = args.results_dir.resolve()
    if args.log_dir is not None:
        LOG_DIR = args.log_dir.resolve()
    if args.work_dir is not None:
        WORK_DIR = args.work_dir.resolve()

    rows = read_tsv(args.manifest)
    required = {"split", "participant_id", "raw_url", "resolver_status"}
    missing = required.difference(rows[0] if rows else set())
    if missing:
        raise SystemExit(f"manifest missing fields: {sorted(missing)}")
    forbidden = {field for field in rows[0] if "age" in field.lower() or "pred" in field.lower()}
    if forbidden:
        raise SystemExit(f"label/prediction fields present in acquisition manifest: {sorted(forbidden)}")

    selected = rows[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.only_participant:
        selected = [row for row in rows if row["participant_id"] == args.only_participant]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS_DIR / "pipeline_status.tsv"
    ode_results = RESULTS_DIR / "cortexode_features.tsv"
    morph_results = RESULTS_DIR / "cortexmorph_features.tsv"

    requested = set(args.methods)
    if "cortexmorph" in requested:
        requested.add("dldirect")

    for number, row in enumerate(selected, start=1):
        participant = row["participant_id"]
        if row.get("resolver_status") != "resolved":
            print(f"[{number}/{len(selected)}] SKIP unresolved {participant}", flush=True)
            continue
        slug = slugify(participant)
        subject_work = WORK_DIR / slug
        shared_raw_path = Path(row["raw_path"]) if row.get("raw_path") else None
        raw_path = (
            shared_raw_path
            if shared_raw_path is not None and shared_raw_path.is_file()
            else subject_work / raw_name(row)
        )
        conformed_path = subject_work / "T1w_conformed.nii.gz"
        ode_dir = subject_work / "ode_input"
        dldir = subject_work / "dldirect"
        ode_json = subject_work / "cortexode.json"
        morph_csv = subject_work / "cortexmorph.csv"
        dld_log = LOG_DIR / f"{slug}.dldirect.log"
        ode_log = LOG_DIR / f"{slug}.cortexode.log"
        morph_log = LOG_DIR / f"{slug}.cortexmorph.log"

        state = status_record(row)
        previous = existing_status(status_path, participant)
        if previous:
            state.update(previous)
        state["error"] = ""
        started = time.monotonic()
        try:
            print(f"[{number}/{len(selected)}] {participant} ({row['split']})", flush=True)
            if raw_path == shared_raw_path:
                state["download_status"] = "existing_main_pipeline_raw"
            else:
                download_raw(row, raw_path)
                state["download_status"] = "downloaded"
            upsert_tsv(status_path, "participant_id", state)

            if "cortexode" in requested and state.get("cortexode_status") != "ok":
                if not conformed_path.is_file():
                    code, seconds = run_command(
                        [sys.executable, str(DL_DIRECT_DIR / "src" / "conform.py"), str(raw_path), str(conformed_path)],
                        DL_DIRECT_DIR,
                        ode_log,
                        timeout=900,
                    )
                    if code != 0:
                        raise RuntimeError(f"conform failed with code {code}")
                if not (ode_dir / "mri" / "orig.mgz").is_file():
                    prepare_ode_input(conformed_path, ode_dir)
                code, seconds = run_command(
                    [
                        sys.executable,
                        str(ODE_SCRIPT),
                        "--input-dir",
                        str(ode_dir),
                        "--output-json",
                        str(ode_json),
                        "--device",
                        "cuda",
                    ],
                    ROOT,
                    ode_log,
                    timeout=900,
                )
                if code != 0 or not ode_json.is_file():
                    raise RuntimeError(f"CortexODE failed with code {code}")
                ode_record = flatten_ode_json(ode_json, row)
                if ode_record.get("method_status") != "ok":
                    raise RuntimeError("CortexODE returned non-ok status")
                upsert_tsv(ode_results, "participant_id", ode_record)
                state["cortexode_status"] = "ok"
                upsert_tsv(status_path, "participant_id", state)

            required_dld = [dldir / name for name in ("softmax_seg.nii.gz", "gmprobT.nii.gz", "wmprobT.nii.gz")]
            if "dldirect" in requested:
                segmentation_file = dldir / "softmax_seg.nii.gz"
                if not segmentation_file.is_file():
                    code, seconds = run_command(
                        [
                            "bash",
                            str(DL_DIRECT_DIR / "dl+direct.sh"),
                            "--no-cth",
                            "--bet",
                            "--keep",
                            "--subject",
                            slug,
                            str(raw_path),
                            str(dldir),
                        ],
                        DL_DIRECT_DIR,
                        dld_log,
                        timeout=3600,
                    )
                    if code != 0:
                        raise RuntimeError(f"DL+DiReCT segmentation failed with code {code}")
                if not segmentation_file.is_file():
                    raise RuntimeError("DL+DiReCT completed without softmax segmentation")
                if "cortexmorph" in requested and not all(path.is_file() for path in required_dld[1:]):
                    code, seconds = run_command(
                        [
                            sys.executable,
                            str(DL_DIRECT_DIR / "src" / "DiReCT.py"),
                            "--prepare-only",
                            "True",
                            str(dldir),
                            str(dldir),
                        ],
                        DL_DIRECT_DIR,
                        dld_log,
                        timeout=900,
                    )
                    if code != 0:
                        raise RuntimeError(f"DiReCT probability-map preparation failed with code {code}")
                if "cortexmorph" in requested and not all(path.is_file() for path in required_dld[1:]):
                    raise RuntimeError("DiReCT preparation completed without required CortexMorph maps")
                state["dldirect_status"] = "ok"
                upsert_tsv(status_path, "participant_id", state)

            if "cortexmorph" in requested and state.get("cortexmorph_status") != "ok":
                if not all(path.is_file() for path in required_dld):
                    raise RuntimeError("CortexMorph prerequisites are absent after DL+DiReCT")
                code, seconds = run_command(
                    [
                        sys.executable,
                        str(MORPH_SCRIPT),
                        "--input-dir",
                        str(dldir),
                        "--case-id",
                        participant,
                        "--output-csv",
                        str(morph_csv),
                    ],
                    ROOT,
                    morph_log,
                    timeout=900,
                )
                if code != 0 or not morph_csv.is_file():
                    raise RuntimeError(f"CortexMorph failed with code {code}")
                morph_record = flatten_morph_csv(morph_csv, row)
                upsert_tsv(morph_results, "participant_id", morph_record)
                state["cortexmorph_status"] = "ok"
                upsert_tsv(status_path, "participant_id", state)

            state["elapsed_sec"] = round(time.monotonic() - started, 2)
            upsert_tsv(status_path, "participant_id", state)
            complete = all(state.get(f"{name}_status") == "ok" for name in requested)
            if complete and not args.keep_work:
                shutil.rmtree(subject_work, ignore_errors=False)
            print(
                f"  done ode={state.get('cortexode_status')} dldirect={state.get('dldirect_status')} "
                f"morph={state.get('cortexmorph_status')} elapsed={state['elapsed_sec']}s",
                flush=True,
            )
        except Exception as error:
            state["error"] = f"{type(error).__name__}: {error}"
            state["elapsed_sec"] = round(time.monotonic() - started, 2)
            for name in requested:
                if state.get(f"{name}_status") not in {"ok", "not_run"}:
                    state[f"{name}_status"] = "error"
            upsert_tsv(status_path, "participant_id", state)
            print(f"  ERROR {state['error']}", flush=True)


if __name__ == "__main__":
    main()
