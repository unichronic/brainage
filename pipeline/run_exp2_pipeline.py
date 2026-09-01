#!/usr/bin/env python3
"""Orchestrate the clone-portable Exp2 workflow.

This driver owns the data layout and split boundaries. It deliberately keeps
the heavier neuroimaging and research implementations in their existing
modules, but gives them one ordered, reproducible entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
DEFAULT_MANIFEST = PIPELINE / "data" / "exp2_manifest.tsv"
DEFAULT_DATA_ROOT = PIPELINE / "data" / "exp2"
PACKAGE_ROOT = ROOT / "FastBrainAge" / "FastBrainAge"
if not PACKAGE_ROOT.is_dir():
    PACKAGE_ROOT = ROOT
PACKAGED_MODEL = PACKAGE_ROOT / "models" / "exp2.joblib"
PACKAGE_MASK = PACKAGE_ROOT / "assets" / "brainmask_12.8.nii"
EXPECTED_SPLITS = {"train": 2742, "calibration": 554, "test": 561}
DOWNLOADABLE_STATUSES = {"available", "available_via_mirror", "candidate_openneuro_path"}
USER_AGENT = "brainage-exp2-pipeline/1.0"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in fields})


def load_json_config() -> dict:
    return json.loads((PIPELINE / "config" / "exp2.json").read_text())


def validate_rows(rows: list[dict[str, str]], source: Path, require_urls: bool = True) -> None:
    required = {
        "split",
        "role",
        "sample_key",
        "participant_id",
        "age",
        "source_cohort",
        "source_dataset_id",
        "source_subject_id",
        "session_id",
        "source_url",
        "download_url",
        "download_status",
    }
    if not rows:
        raise SystemExit(f"manifest is empty: {source}")
    missing = required.difference(rows[0])
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    ids = set()
    keys = set()
    for line, row in enumerate(rows, start=2):
        participant_id = row["participant_id"]
        if not participant_id or participant_id in ids or "/" in participant_id or "\\" in participant_id:
            raise SystemExit(f"line {line}: invalid or duplicate participant_id")
        if not row["sample_key"] or row["sample_key"] in keys:
            raise SystemExit(f"line {line}: invalid or duplicate sample_key")
        if row["split"] not in EXPECTED_SPLITS:
            raise SystemExit(f"line {line}: unexpected split {row['split']!r}")
        try:
            float(row["age"])
        except ValueError as error:
            raise SystemExit(f"line {line}: invalid age") from error
        if require_urls:
            if not row["download_url"].startswith(("http://", "https://")):
                raise SystemExit(f"line {line}: download_url is not HTTP(S)")
            if row["download_status"] not in DOWNLOADABLE_STATUSES:
                raise SystemExit(f"line {line}: unsupported download status {row['download_status']!r}")
        ids.add(participant_id)
        keys.add(row["sample_key"])


def slug(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "subject"
    digest = hashlib.sha1(value.encode()).hexdigest()[:10]
    return f"{readable[:100]}_{digest}"


def safe_name(value: str, fallback: str) -> str:
    name = Path(unquote(urlparse(value).path)).name if value else ""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if not name or name in {".", ".."}:
        name = f"{fallback}_T1w.nii.gz"
    if not name.endswith((".nii", ".nii.gz")):
        name += ".nii.gz"
    return name


def stage_paths(data_root: Path, row: dict[str, str]) -> dict[str, str]:
    participant = row["participant_id"]
    subject_slug = slug(participant)
    raw_name = row.get("raw_filename", "") or row["download_url"]
    raw_rel = Path("raw") / subject_slug / safe_name(raw_name, subject_slug)
    gm_rel = Path("gm_maps") / f"{subject_slug}_mwc1.nii"
    # The CSF scripts index maps by the exact participant_id stem.
    csf_rel = Path("csf_maps") / f"{participant}_mwc3.nii.gz"
    return {
        "raw_relpath": raw_rel.as_posix(),
        "map_relpath": gm_rel.as_posix(),
        "csf_map_relpath": csf_rel.as_posix(),
        "raw_path": str(data_root / raw_rel),
        "map_path": str(data_root / gm_rel),
        "csf_map_path": str(data_root / csf_rel),
    }


def prepare(args: argparse.Namespace) -> None:
    rows = read_tsv(args.manifest)
    validate_rows(rows, args.manifest)
    data_root = args.data_root.resolve()
    enriched = []
    for row in rows:
        output = dict(row)
        output.update(stage_paths(data_root, row))
        enriched.append(output)

    fields = list(enriched[0])
    manifest_dir = data_root / "manifests"
    write_tsv(manifest_dir / "all.tsv", enriched, fields)
    for split in EXPECTED_SPLITS:
        write_tsv(manifest_dir / f"{split}.tsv", [row for row in enriched if row["split"] == split], fields)

    cortical_fields = [
        "split",
        "participant_id",
        "source_cohort",
        "source_dataset_id",
        "source_subject_id",
        "manifest_session",
        "source_url",
        "declared_raw_path",
        "raw_path",
        "raw_url",
        "raw_filename",
        "resolver",
        "resolver_status",
        "candidate_count",
        "resolver_note",
    ]
    cortical_rows = []
    for row in enriched:
        cortical_rows.append(
            {
                "split": "locked_test" if row["split"] == "test" else row["split"],
                "participant_id": row["participant_id"],
                "source_cohort": row["source_cohort"],
                "source_dataset_id": row["source_dataset_id"],
                "source_subject_id": row["source_subject_id"],
                "manifest_session": row["session_id"],
                "source_url": row["source_url"],
                "declared_raw_path": "",
                "raw_path": row["raw_path"],
                "raw_url": row["download_url"],
                "raw_filename": Path(row["raw_relpath"]).name,
                "resolver": "exp2-pipeline-manifest",
                "resolver_status": "resolved",
                "candidate_count": "1",
                "resolver_note": "label-free acquisition row from the Exp2 manifest",
            }
        )
    write_tsv(data_root / "manifests" / "cortical_all.tsv", cortical_rows, cortical_fields)
    write_tsv(
        data_root / "manifests" / "cortical_calibration.tsv",
        [row for row in cortical_rows if row["split"] == "calibration"],
        cortical_fields,
    )
    print(f"prepared {len(enriched)} rows under {data_root}")
    print("stage manifests: all.tsv, train.tsv, calibration.tsv, test.tsv")
    print("cortical manifests: cortical_calibration.tsv, cortical_all.tsv")


def load_stage_rows(data_root: Path) -> list[dict[str, str]]:
    path = data_root.resolve() / "manifests" / "all.tsv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run prepare first")
    rows = read_tsv(path)
    validate_rows(rows, path)
    return rows


def select_rows(rows: list[dict[str, str]], split: str, limit: Optional[int]) -> list[dict[str, str]]:
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    if limit is not None:
        selected = selected[:limit]
    return selected


def download_one(row: dict[str, str], data_root: Path) -> dict[str, str]:
    destination = Path(row["raw_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "participant_id": row["participant_id"],
        "split": row["split"],
        "raw_path": str(destination),
        "download_url": row["download_url"],
        "status": "",
        "error": "",
    }
    if destination.is_file() and destination.stat().st_size > 0:
        base["status"] = "exists"
        return base

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(row["download_url"], headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=1800) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if partial.stat().st_size == 0:
            raise RuntimeError("zero-byte download")
        os.replace(partial, destination)
        base["status"] = "downloaded"
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as error:
        if partial.exists():
            partial.unlink()
        base["status"] = "error"
        base["error"] = f"{type(error).__name__}: {error}"[:500]
    return base


def download(args: argparse.Namespace) -> None:
    rows = select_rows(load_stage_rows(args.data_root), args.split, args.limit)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, args.data_root) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']:10s} {result['participant_id']}", flush=True)
    order = {row["participant_id"]: index for index, row in enumerate(load_stage_rows(args.data_root))}
    results.sort(key=lambda row: order[row["participant_id"]])
    status_path = args.data_root / "status" / "download.tsv"
    fields = ["participant_id", "split", "raw_path", "download_url", "status", "error"]
    write_tsv(status_path, results, fields)
    failures = [row for row in results if row["status"] == "error"]
    print(f"downloaded/existing={len(results) - len(failures)} errors={len(failures)}")
    if failures:
        raise SystemExit(2)


def docker_map_path(data_root: Path, path: str) -> str:
    root = data_root.resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"generated path escapes data root: {candidate}") from error
    return "/data/" + relative.as_posix()


def preprocess_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    output = Path(row["map_path"])
    result = {"participant_id": row["participant_id"], "split": row["split"], "status": "", "error": ""}
    if output.is_file() and output.stat().st_size > 0:
        result["status"] = "exists"
        return result
    if not Path(row["raw_path"]).is_file():
        result["status"] = "missing_raw"
        result["error"] = row["raw_path"]
        return result
    log_path = args.data_root / "status" / "preprocess" / f"{slug(row['participant_id'])}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{args.data_root.resolve()}:/data",
        args.image,
        "preprocess",
        docker_map_path(args.data_root, row["raw_path"]),
        docker_map_path(args.data_root, row["map_path"]),
    ]
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    result["status"] = "ok" if completed.returncode == 0 and output.is_file() else "error"
    result["error"] = "" if result["status"] == "ok" else f"docker exit {completed.returncode}; see {log_path}"
    return result


def preprocess(args: argparse.Namespace) -> None:
    rows = select_rows(load_stage_rows(args.data_root), args.split, args.limit)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(preprocess_one, row, args) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']:10s} {result['participant_id']}", flush=True)
    fields = ["participant_id", "split", "status", "error"]
    write_tsv(args.data_root / "status" / "preprocess.tsv", results, fields)
    failures = [row for row in results if row["status"] not in {"ok", "exists"}]
    print(f"processed/existing={len(results) - len(failures)} errors={len(failures)}")
    if failures:
        raise SystemExit(2)


def package_env() -> dict[str, str]:
    environment = os.environ.copy()
    source = str(PACKAGE_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = source + (os.pathsep + existing if existing else "")
    return environment


def run_command(command: list[str], cwd: Path = ROOT, env: Optional[dict[str, str]] = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_package(command: str, arguments: list[str]) -> None:
    run_command([sys.executable, "-m", "fastbrainage", command, *arguments], env=package_env())


def extract_gm(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    manifest = data_root / "manifests" / "all.tsv"
    output = data_root / "results" / "gm_features.npz"
    run_package(
        "extract-features",
        ["--manifest", str(manifest), "--mask", str(PACKAGE_MASK), "--output", str(output)],
    )


def train(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output = args.output_model or (data_root / "models" / "exp2_retrained.joblib")
    run_package(
        "train",
        [
            "--manifest",
            str(data_root / "manifests" / "train.tsv"),
            "--features",
            str(data_root / "results" / "gm_features.npz"),
            "--output-model",
            str(output),
            "--pca-components",
            "128",
            "--gpr-kernel",
            "matern_white",
            "--gpr-restarts",
            "1",
            "--random-state",
            "20260806",
            "--age-expansion-factor",
            "1.010",
        ],
    )


def predict(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    model = (args.model or PACKAGED_MODEL).resolve()
    output = args.output or (data_root / "results" / "predictions.tsv")
    command = [
        "predict",
        "--model",
        str(model),
        "--features",
        str(data_root / "results" / "gm_features.npz"),
        "--manifest",
        str(data_root / "manifests" / "all.tsv"),
        "--output",
        str(output),
    ]
    if args.bias_correction:
        command.extend(["--bias-correction", str(args.bias_correction.resolve())])
    run_package(command[0], command[1:])


def fit_bias(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    predictions = args.predictions or (data_root / "results" / "predictions.tsv")
    rows = read_tsv(predictions)
    calibration = [
        {field: row[field] for field in ("participant_id", "age", "predicted_age")}
        for row in rows
        if row.get("split") == "calibration"
    ]
    if len(calibration) != EXPECTED_SPLITS["calibration"]:
        raise SystemExit(f"expected 554 calibration predictions, found {len(calibration)}")
    calibration_path = data_root / "manifests" / "calibration_predictions.tsv"
    write_tsv(calibration_path, calibration, ["participant_id", "age", "predicted_age"])
    output = args.output or (data_root / "results" / "age_bias_correction.json")
    run_package(
        "fit-bias-correction",
        ["--calibration-predictions", str(calibration_path), "--output", str(output)],
    )


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    return numerator / (x_var * y_var) ** 0.5 if x_var and y_var else float("nan")


def score(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    predictions = args.predictions or (data_root / "results" / "predictions.tsv")
    rows = read_tsv(predictions)
    report = {"prediction_file": str(predictions), "splits": {}}
    for split in ("train", "calibration", "test"):
        selected = [row for row in rows if row.get("split") == split]
        if not selected:
            continue
        age = [float(row["age"]) for row in selected]
        predicted = [float(row["predicted_age"]) for row in selected]
        gap = [prediction - actual for prediction, actual in zip(predicted, age)]
        result = {
            "n": len(selected),
            "mae": sum(abs(value) for value in gap) / len(gap),
            "rmse": (sum(value * value for value in gap) / len(gap)) ** 0.5,
            "bias": sum(gap) / len(gap),
            "pearson_age_prediction": correlation(age, predicted),
            "pearson_age_gap": correlation(age, gap),
        }
        corrected = [row.get("corrected_predicted_age", "") for row in selected]
        if all(corrected):
            corrected_values = [float(value) for value in corrected]
            corrected_gap = [value - actual for value, actual in zip(corrected_values, age)]
            result["corrected_mae"] = sum(abs(value) for value in corrected_gap) / len(corrected_gap)
        report["splits"][split] = result
    output = args.output or (data_root / "results" / "scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def csf_preprocess(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    command = [
        sys.executable,
        str(ROOT / "csf_experiment" / "run_exp2_csf_batches.py"),
        "--manifest",
        str(data_root / "manifests" / "all.tsv"),
        "--extractor",
        str(ROOT / "csf_extract_run.py"),
        "--recipe",
        str(ROOT / "csf_feasibility_test" / "fastspm_v1_gmcsf.m"),
        "--image",
        args.image,
        "--output-dir",
        str(data_root / "csf_maps"),
        "--input-root",
        str(data_root / "csf_raw_batches"),
        "--work-root",
        str(data_root / "csf_work"),
        "--status-log",
        str(data_root / "status" / "csf_batches.tsv"),
        "--extract-status-log",
        str(data_root / "status" / "csf_extract.tsv"),
        "--batch-size",
        str(args.batch_size),
        "--download-workers",
        str(args.download_workers),
        "--workers",
        str(args.workers),
        "--start-batch",
        str(args.start_batch),
        "--max-batches",
        str(args.max_batches),
    ]
    if args.source_split:
        command.extend(["--source-split", args.source_split])
    run_command(command)


def csf_features(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    environment = package_env()
    if args.atlas:
        environment["HARVARD_OXFORD_ATLAS"] = str(args.atlas.resolve())
    run_command(
        [
            sys.executable,
            str(ROOT / "extract_exp2_csf_features.py"),
            "--manifest",
            str(data_root / "manifests" / "all.tsv"),
            "--maps-dir",
            str(data_root / "csf_maps"),
            "--output",
            str(data_root / "results" / "csf_features.tsv"),
            "--workers",
            str(args.workers),
        ],
        env=environment,
    )


def csf_feature_age(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    run_command(
        [
            sys.executable,
            str(ROOT / "analyze_exp2_csf_feature_age_signal.py"),
            "--features",
            str(data_root / "results" / "csf_features.tsv"),
            "--correlations",
            str(data_root / "results" / "csf_feature_correlations.tsv"),
            "--report",
            str(data_root / "results" / "csf_feature_age_report.json"),
        ],
        env=package_env(),
    )


def csf_screen(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    run_command(
        [
            sys.executable,
            str(ROOT / "screen_exp2_csf_age_features.py"),
            "--features",
            str(data_root / "results" / "csf_features.tsv"),
            "--gm-qc",
            str(data_root / "results" / "gm_features.qc.tsv"),
            "--derived-features",
            str(data_root / "results" / "csf_age_features.tsv"),
            "--correlations",
            str(data_root / "results" / "csf_age_correlations.tsv"),
            "--report",
            str(data_root / "results" / "csf_age_screen_report.json"),
        ],
        env=package_env(),
    )


def csf_fusion(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    run_command(
        [
            sys.executable,
            str(ROOT / "analyze_exp2_csf_fusion.py"),
            "--manifest",
            str(data_root / "manifests" / "all.tsv"),
            "--raw-pred",
            str(data_root / "results" / "predictions.tsv"),
            "--features",
            str(data_root / "results" / "csf_features.tsv"),
            "--report",
            str(data_root / "results" / "csf_fusion_report.json"),
            "--predictions",
            str(data_root / "results" / "csf_fusion_predictions.tsv"),
            "--model-label",
            "Exp2 Matern+White production model",
        ],
        env=package_env(),
    )


def cortical(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    manifest = data_root / "manifests" / (
        "cortical_all.tsv" if args.include_locked_test else "cortical_calibration.tsv"
    )
    command = [
        sys.executable,
        str(ROOT / "run_exp2_cortical_pilot.py"),
        "--manifest",
        str(manifest),
        "--methods",
        *args.methods,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.offset:
        command.extend(["--offset", str(args.offset)])
    command.extend(
        [
            "--results-dir",
            str(data_root / "results" / "cortical"),
            "--log-dir",
            str(data_root / "status" / "cortical_logs"),
            "--work-dir",
            str(data_root / "work" / "cortical"),
        ]
    )
    if args.dl_direct_dir:
        command.extend(["--dl-direct-dir", str(args.dl_direct_dir.resolve())])
    run_command(command, env=package_env())


def age_aware(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    command = [
        sys.executable,
        str(ROOT / "exp2_age_aware_benchmark" / "run_linear_benchmark.py"),
        "--repeats",
        str(args.repeats),
        "--splits",
        str(args.splits),
        "--features",
        str(args.features or (data_root / "results" / "gm_features.npz")),
        "--train-manifest",
        str(args.train_manifest or (data_root / "manifests" / "train.tsv")),
        "--calibration-manifest",
        str(args.calibration_manifest or (data_root / "manifests" / "calibration.tsv")),
        "--output-dir",
        str(args.output_dir or (data_root / "results" / "age_aware")),
    ]
    if args.skip_calibration:
        command.append("--skip-calibration")
    if args.skip_pc_diagnostic:
        command.append("--skip-pc-diagnostic")
    run_command(command, env=package_env())


def run_core(args: argparse.Namespace) -> None:
    """Run the ordered GM/model/correction path in one command.

    The optional branches remain separate because CSF and cortical processing
    have different dependencies and do not belong in the production GM head.
    """
    data_root = args.data_root.resolve()
    prepare(argparse.Namespace(manifest=args.manifest, data_root=data_root))

    if not args.skip_download:
        download(
            argparse.Namespace(
                data_root=data_root,
                split="all",
                limit=None,
                workers=args.download_workers,
            )
        )
    if not args.skip_preprocess:
        preprocess(
            argparse.Namespace(
                data_root=data_root,
                split="all",
                limit=None,
                workers=args.workers,
                image=args.image,
            )
        )
    if not args.skip_extract:
        extract_gm(argparse.Namespace(data_root=data_root))

    model = None
    if args.retrain:
        model = data_root / "models" / "exp2_retrained.joblib"
        train(argparse.Namespace(data_root=data_root, output_model=model))

    uncorrected = data_root / "results" / "predictions.tsv"
    predict(argparse.Namespace(data_root=data_root, model=model, output=uncorrected, bias_correction=None))
    score(
        argparse.Namespace(
            data_root=data_root,
            predictions=uncorrected,
            output=data_root / "results" / "scores_uncorrected.json",
        )
    )

    correction = data_root / "results" / "age_bias_correction.json"
    fit_bias(argparse.Namespace(data_root=data_root, predictions=uncorrected, output=correction))
    corrected = data_root / "results" / "predictions_corrected.tsv"
    predict(
        argparse.Namespace(
            data_root=data_root,
            model=model,
            output=corrected,
            bias_correction=correction,
        )
    )
    score(
        argparse.Namespace(
            data_root=data_root,
            predictions=corrected,
            output=data_root / "results" / "scores_corrected.json",
        )
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_exp2_pipeline.py")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate the committed public manifest")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.set_defaults(function=validate_command)

    prepare_parser = sub.add_parser("prepare", help="make clone-specific stage manifests")
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    add_data_args(prepare_parser)
    prepare_parser.set_defaults(function=prepare)

    download_parser = sub.add_parser("download", help="download raw T1 images")
    add_data_args(download_parser)
    download_parser.add_argument("--split", choices=("all", "train", "calibration", "test"), default="all")
    download_parser.add_argument("--limit", type=int)
    download_parser.add_argument("--workers", type=int, default=8)
    download_parser.set_defaults(function=download)

    preprocess_parser = sub.add_parser("preprocess", help="run GM FastSPM preprocessing in Docker")
    add_data_args(preprocess_parser)
    preprocess_parser.add_argument("--image", default="fastbrainage:exp2")
    preprocess_parser.add_argument("--split", choices=("all", "train", "calibration", "test"), default="all")
    preprocess_parser.add_argument("--limit", type=int)
    preprocess_parser.add_argument("--workers", type=int, default=1)
    preprocess_parser.set_defaults(function=preprocess)

    extract_parser = sub.add_parser("extract-gm", help="extract S4_R4 GM features")
    add_data_args(extract_parser)
    extract_parser.set_defaults(function=extract_gm)

    train_parser = sub.add_parser("train", help="retrain the current Exp2 head on train rows")
    add_data_args(train_parser)
    train_parser.add_argument("--output-model", type=Path)
    train_parser.set_defaults(function=train)

    predict_parser = sub.add_parser("predict", help="predict all Exp2 rows with the current model")
    add_data_args(predict_parser)
    predict_parser.add_argument("--model", type=Path)
    predict_parser.add_argument("--output", type=Path)
    predict_parser.add_argument("--bias-correction", type=Path)
    predict_parser.set_defaults(function=predict)

    bias_parser = sub.add_parser("fit-bias", help="fit correction on calibration predictions only")
    add_data_args(bias_parser)
    bias_parser.add_argument("--predictions", type=Path)
    bias_parser.add_argument("--output", type=Path)
    bias_parser.set_defaults(function=fit_bias)

    score_parser = sub.add_parser("score", help="score predictions by split")
    add_data_args(score_parser)
    score_parser.add_argument("--predictions", type=Path)
    score_parser.add_argument("--output", type=Path)
    score_parser.set_defaults(function=score)

    csf_pre_parser = sub.add_parser("csf-preprocess", help="download and generate CSF mwc3 maps")
    add_data_args(csf_pre_parser)
    csf_pre_parser.add_argument("--image", default="fastbrainage:exp2")
    csf_pre_parser.add_argument("--batch-size", type=int, default=48)
    csf_pre_parser.add_argument("--download-workers", type=int, default=8)
    csf_pre_parser.add_argument("--workers", type=int, default=3)
    csf_pre_parser.add_argument("--start-batch", type=int, default=0)
    csf_pre_parser.add_argument("--max-batches", type=int, default=0)
    csf_pre_parser.add_argument("--source-split", default="")
    csf_pre_parser.set_defaults(function=csf_preprocess)

    csf_features_parser = sub.add_parser("csf-features", help="extract scalar CSF/ventricle features")
    add_data_args(csf_features_parser)
    csf_features_parser.add_argument("--workers", type=int, default=4)
    csf_features_parser.add_argument("--atlas", type=Path, help="Harvard-Oxford atlas NIfTI")
    csf_features_parser.set_defaults(function=csf_features)

    csf_age_parser = sub.add_parser("csf-feature-age", help="measure direct CSF feature-age signal")
    add_data_args(csf_age_parser)
    csf_age_parser.set_defaults(function=csf_feature_age)

    csf_screen_parser = sub.add_parser("csf-screen", help="run GM-normalized CSF feature screen")
    add_data_args(csf_screen_parser)
    csf_screen_parser.set_defaults(function=csf_screen)

    csf_fusion_parser = sub.add_parser("csf-fusion", help="run calibration-only CSF fusion diagnostic")
    add_data_args(csf_fusion_parser)
    csf_fusion_parser.set_defaults(function=csf_fusion)

    cortical_parser = sub.add_parser("cortical", help="run the label-blind cortical-thickness pilot")
    add_data_args(cortical_parser)
    cortical_parser.add_argument("--include-locked-test", action="store_true")
    cortical_parser.add_argument("--limit", type=int)
    cortical_parser.add_argument("--offset", type=int, default=0)
    cortical_parser.add_argument("--dl-direct-dir", type=Path)
    cortical_parser.add_argument(
        "--methods",
        nargs="+",
        choices=("cortexode", "dldirect", "cortexmorph"),
        default=["cortexode"],
    )
    cortical_parser.set_defaults(function=cortical)

    age_parser = sub.add_parser("age-aware", help="run the group-safe age-aware benchmark")
    add_data_args(age_parser)
    age_parser.add_argument("--features", type=Path)
    age_parser.add_argument("--train-manifest", type=Path)
    age_parser.add_argument("--calibration-manifest", type=Path)
    age_parser.add_argument("--output-dir", type=Path)
    age_parser.add_argument("--repeats", type=int, default=1)
    age_parser.add_argument("--splits", type=int, default=5)
    age_parser.add_argument("--skip-calibration", action="store_true")
    age_parser.add_argument("--skip-pc-diagnostic", action="store_true")
    age_parser.set_defaults(function=age_aware)

    core_parser = sub.add_parser(
        "run-core",
        help="run prepare -> download -> GM preprocess -> features -> predict -> bias correction -> score",
    )
    core_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    add_data_args(core_parser)
    core_parser.add_argument("--image", default="fastbrainage:exp2")
    core_parser.add_argument("--workers", type=int, default=1)
    core_parser.add_argument("--download-workers", type=int, default=8)
    core_parser.add_argument("--skip-download", action="store_true")
    core_parser.add_argument("--skip-preprocess", action="store_true")
    core_parser.add_argument("--skip-extract", action="store_true")
    core_parser.add_argument("--retrain", action="store_true", help="fit a new Exp2 head before prediction")
    core_parser.set_defaults(function=run_core)
    return parser


def validate_command(args: argparse.Namespace) -> None:
    rows = read_tsv(args.manifest)
    validate_rows(rows, args.manifest)
    counts = {split: sum(row["split"] == split for row in rows) for split in EXPECTED_SPLITS}
    expected = load_json_config()["dataset"]["splits"]
    if counts != expected:
        raise SystemExit(f"unexpected split counts: {counts}; expected {expected}")
    print(json.dumps({"manifest": str(args.manifest), "rows": len(rows), "splits": counts}, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
