#!/usr/bin/env python3
"""Quantify chronological-age signal in the exp2 CSF features.

This is the primary CSF analysis.  It does not rank features by prediction
MAE or by age-gap bias; it reports their direct Pearson and Spearman
association with chronological age on development, calibration, locked-test,
and descriptive pooled groups.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


EXPECTED = {"train": 2742, "calibration": 554, "test": 561}
FEATURES = [
    "total_csf_mL",
    "left_ventricle_mL",
    "right_ventricle_mL",
    "total_ventricle_mL",
]
CORRELATION_FIELDS = [
    "group_type",
    "group",
    "n",
    "feature",
    "pearson_r",
    "pearson_p",
    "spearman_rho",
    "spearman_p",
    "feature_mean",
    "feature_sd",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def correlation_rows(group_type: str, group: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    age = np.array([float(row["age"]) for row in rows])
    result = []
    for feature in FEATURES:
        values = np.array([float(row[feature]) for row in rows])
        pearson = pearsonr(age, values)
        spearman = spearmanr(age, values)
        result.append(
            {
                "group_type": group_type,
                "group": group,
                "n": str(len(rows)),
                "feature": feature,
                "pearson_r": f"{pearson.statistic:.12g}",
                "pearson_p": f"{pearson.pvalue:.12g}",
                "spearman_rho": f"{spearman.statistic:.12g}",
                "spearman_p": f"{spearman.pvalue:.12g}",
                "feature_mean": f"{values.mean():.12g}",
                "feature_sd": f"{values.std():.12g}",
            }
        )
    return result


def to_json(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for row in rows:
        result.setdefault(row["group_type"], {})
        result[row["group_type"]].setdefault(row["group"], {})
        result[row["group_type"]][row["group"]][row["feature"]] = {
            "n": int(row["n"]),
            "pearson_r": float(row["pearson_r"]),
            "pearson_p": float(row["pearson_p"]),
            "spearman_rho": float(row["spearman_rho"]),
            "spearman_p": float(row["spearman_p"]),
            "mean": float(row["feature_mean"]),
            "sd": float(row["feature_sd"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--correlations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-cohort-n", type=int, default=10)
    args = parser.parse_args()

    rows = read_tsv(args.features)
    ids = [row["participant_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("feature table has duplicate participant_id values")
    if any(row.get("status") != "ok" for row in rows):
        raise SystemExit("feature table contains non-ok rows")

    split_counts = {split: sum(row["split"] == split for row in rows) for split in EXPECTED}
    if split_counts != EXPECTED:
        raise SystemExit(f"unexpected split counts: {split_counts}")

    groups: list[tuple[str, str, list[dict[str, str]]]] = [("split", "all", rows)]
    for split in ("train", "calibration", "test"):
        groups.append(("split", split, [row for row in rows if row["split"] == split]))
    for cohort in sorted({row["source_cohort"] for row in rows}):
        cohort_rows = [row for row in rows if row["source_cohort"] == cohort]
        if len(cohort_rows) >= args.min_cohort_n:
            groups.append(("source_cohort", cohort, cohort_rows))

    correlation_rows_out: list[dict[str, str]] = []
    for group_type, group, group_rows in groups:
        correlation_rows_out.extend(correlation_rows(group_type, group, group_rows))

    args.correlations.parent.mkdir(parents=True, exist_ok=True)
    with args.correlations.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CORRELATION_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(correlation_rows_out)

    report = {
        "protocol": {
            "rows": len(rows),
            "split_counts": split_counts,
            "primary_question": "How strongly do CSF/ventricle features correlate with chronological age?",
            "primary_statistics": ["Pearson r", "Spearman rho"],
            "locked_test_used_for_confirmation_only": True,
        },
        "correlations": to_json(correlation_rows_out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    for row in correlation_rows_out:
        if row["group_type"] == "split" and row["group"] in {"train", "calibration", "test", "all"}:
            print(
                f"{row['group']:11s} {row['feature']:22s} "
                f"Pearson={float(row['pearson_r']):+.4f} "
                f"Spearman={float(row['spearman_rho']):+.4f} n={row['n']}"
            )


if __name__ == "__main__":
    main()
