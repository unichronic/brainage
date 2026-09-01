#!/usr/bin/env python3
"""Extract CSF and lateral-ventricle features for the exact exp2 manifest.

The output preserves the manifest's train/calibration/locked-test labels.  It
is deliberately separate from ``csf_fusion_fit.py``, whose historical 742-
subject hash split is not valid for exp2.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OUTPUT_FIELDS = [
    "split",
    "role",
    "sample_key",
    "participant_id",
    "age",
    "source_cohort",
    "source_dataset_id",
    "source_subject_id",
    "session_id",
    "map_file",
    "status",
    "total_csf_mL",
    "left_ventricle_mL",
    "right_ventricle_mL",
    "total_ventricle_mL",
    "error",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def map_id(path: Path) -> str | None:
    for suffix in ("_mwc3.nii.gz", "_mwc3.nii"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def index_maps(maps_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in maps_dir.iterdir():
        if not path.is_file():
            continue
        participant_id = map_id(path)
        if participant_id is None:
            continue
        # Prefer compressed maps if both formats happen to be present.
        if participant_id not in indexed or path.name.endswith(".nii.gz"):
            indexed[participant_id] = path
    return indexed


def row_base(row: dict[str, str], map_path: Path | None) -> dict[str, str]:
    return {
        "split": row.get("split", ""),
        "role": row.get("role", ""),
        "sample_key": row.get("sample_key", ""),
        "participant_id": row["participant_id"],
        "age": row.get("age", ""),
        "source_cohort": row.get("source_cohort", ""),
        "source_dataset_id": row.get("source_dataset_id", ""),
        "source_subject_id": row.get("source_subject_id", ""),
        "session_id": row.get("session_id", ""),
        "map_file": str(map_path) if map_path is not None else "",
        "status": "pending",
        "total_csf_mL": "",
        "left_ventricle_mL": "",
        "right_ventricle_mL": "",
        "total_ventricle_mL": "",
        "error": "",
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_one(row: dict[str, str], map_path: Path | None) -> dict[str, str]:
    result = row_base(row, map_path)
    if map_path is None:
        result["status"] = "missing_map"
        result["error"] = "no *_mwc3.nii[.gz] file matched participant_id"
        return result
    try:
        # Imported here so the script can validate/index a manifest without
        # requiring neuroimaging dependencies on the host that launches it.
        from csf_feature_extract import extract

        features = extract(str(map_path))
        result.update(
            {
                "status": "ok",
                "total_csf_mL": f"{features['total_csf_mL']:.9g}",
                "left_ventricle_mL": f"{features['left_ventricle_mL']:.9g}",
                "right_ventricle_mL": f"{features['right_ventricle_mL']:.9g}",
                "total_ventricle_mL": f"{features['total_ventricle_mL']:.9g}",
            }
        )
    except Exception as exc:  # preserve the exact subject-level failure
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    manifest = read_tsv(args.manifest)
    ids = [row["participant_id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise SystemExit("manifest contains duplicate participant_id values")
    maps = index_maps(args.maps_dir)
    print(f"manifest_rows={len(manifest)} indexed_maps={len(maps)}", flush=True)

    partial = args.output.with_suffix(args.output.suffix + ".partial")
    existing: dict[str, dict[str, str]] = {}
    if partial.exists():
        for row in read_tsv(partial):
            if row.get("status") == "ok":
                existing[row["participant_id"]] = row
        print(f"resuming_ok={len(existing)}", flush=True)

    results: dict[str, dict[str, str]] = dict(existing)
    pending = [row for row in manifest if row["participant_id"] not in results]
    print(f"pending={len(pending)} workers={args.workers}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, row, maps.get(row["participant_id"])): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[result["participant_id"]] = result
            if index % args.checkpoint_every == 0 or index == len(pending):
                ordered = [results[row["participant_id"]] for row in manifest if row["participant_id"] in results]
                write_rows(partial, ordered)
                ok = sum(row["status"] == "ok" for row in ordered)
                errors = len(ordered) - ok
                print(f"processed={index}/{len(pending)} ok={ok} errors={errors}", flush=True)

    ordered = [results.get(row["participant_id"], row_base(row, maps.get(row["participant_id"]))) for row in manifest]
    missing_results = [row["participant_id"] for row in ordered if row["status"] == "pending"]
    failed = [row for row in ordered if row["status"] != "ok"]
    write_rows(partial, ordered)
    if missing_results or failed:
        print(f"incomplete missing={len(missing_results)} failed={len(failed)}; kept {partial}", flush=True)
        raise SystemExit(2)
    os.replace(partial, args.output)
    print(f"wrote {args.output} rows={len(ordered)} ok={len(ordered)}", flush=True)


if __name__ == "__main__":
    main()
