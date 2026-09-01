#!/usr/bin/env python3
"""Create the clone-portable Exp2 acquisition manifest.

The research ledger contains local paths and intermediate-map bookkeeping.
This generator retains only the fields needed to download the same raw-T1
inputs and reproduce the fixed split, so a public checkout never inherits a
machine-specific ``/mnt`` path.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "pipeline" / "data" / "exp2_manifest.tsv"
DEFAULT_INPUT = DEFAULT_OUTPUT
SPLITS = {"train", "calibration", "test"}
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
    "source_url",
    "raw_url",
    "license_or_access_rule",
    "download_url",
    "download_status",
    "source_split",
    "raw_filename",
]


def raw_filename(url: str, participant_id: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if not name or name in {".", ".."}:
        name = f"{participant_id}_T1w.nii.gz"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if not name.endswith((".nii", ".nii.gz")):
        name += ".nii.gz"
    return name


def build(input_path: Path, output_path: Path) -> tuple[int, dict[str, int]]:
    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise SystemExit(f"source manifest is empty: {input_path}")

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
        "raw_url",
        "license_or_access_rule",
        "download_url",
        "download_status",
        "source_split",
    }
    missing = required.difference(rows[0])
    if missing:
        raise SystemExit(f"source manifest missing columns: {sorted(missing)}")

    output_rows = []
    seen_ids = set()
    seen_keys = set()
    for line_number, row in enumerate(rows, start=2):
        participant_id = row["participant_id"].strip()
        sample_key = row["sample_key"].strip()
        if not participant_id or not sample_key:
            raise SystemExit(f"line {line_number}: participant_id/sample_key is empty")
        if participant_id in seen_ids or sample_key in seen_keys:
            raise SystemExit(f"line {line_number}: duplicate subject identifier")
        if row["split"] not in SPLITS:
            raise SystemExit(f"line {line_number}: unexpected split {row['split']!r}")
        try:
            float(row["age"])
        except ValueError as error:
            raise SystemExit(f"line {line_number}: invalid age") from error
        url = row["download_url"].strip()
        if not url.startswith(("http://", "https://")):
            raise SystemExit(f"line {line_number}: download_url is not HTTP(S)")
        if "/" in participant_id or "\\" in participant_id:
            raise SystemExit(f"line {line_number}: unsafe participant_id for filenames")
        seen_ids.add(participant_id)
        seen_keys.add(sample_key)
        output_rows.append(
            {
                field: row.get(field, "").strip()
                for field in OUTPUT_FIELDS
                if field != "raw_filename"
            }
        )
        output_rows[-1]["raw_filename"] = raw_filename(url, participant_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    counts = {split: sum(row["split"] == split for row in output_rows) for split in sorted(SPLITS)}
    return len(output_rows), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    total, counts = build(args.input, args.output)
    print(f"wrote {args.output}: rows={total} splits={counts}")


if __name__ == "__main__":
    main()
