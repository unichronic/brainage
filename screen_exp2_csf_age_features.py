#!/usr/bin/env python3
"""Screen feature-only CSF/ventricle representations for chronological-age signal.

This driver deliberately does not load model predictions.  Its primary endpoint
is the direct association between a derived feature and chronological age.  A
small, pre-specified linear composite is reported as a secondary feature-only
summary; its train score is out-of-fold and its calibration/test scores come
from a fit on train only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


EXPECTED_SPLITS = {"train": 2742, "calibration": 554, "test": 561}
BASE_FEATURES = (
    "total_csf_mL",
    "left_ventricle_mL",
    "right_ventricle_mL",
    "total_ventricle_mL",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_correlation(age: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(age) & np.isfinite(values)
    age = age[valid]
    values = values[valid]
    if len(age) < 3:
        return {"n": int(len(age)), "pearson_r": math.nan, "spearman_rho": math.nan}
    return {
        "n": int(len(age)),
        "pearson_r": float(pearsonr(age, values).statistic),
        "spearman_rho": float(spearmanr(age, values).statistic),
    }


def candidate_values(row: dict[str, str]) -> dict[str, float]:
    csf = float(row["total_csf_mL"])
    left = float(row["left_ventricle_mL"])
    right = float(row["right_ventricle_mL"])
    vent = float(row["total_ventricle_mL"])
    gm = float(row["gm_integral_ml"])
    return {
        "gm_integral_ml": gm,
        "total_csf_mL": csf,
        "left_ventricle_mL": left,
        "right_ventricle_mL": right,
        "total_ventricle_mL": vent,
        "log1p_total_csf_mL": math.log1p(csf),
        "log1p_left_ventricle_mL": math.log1p(left),
        "log1p_right_ventricle_mL": math.log1p(right),
        "log1p_total_ventricle_mL": math.log1p(vent),
        "total_csf_over_gm": csf / gm,
        "left_ventricle_over_gm": left / gm,
        "right_ventricle_over_gm": right / gm,
        "total_ventricle_over_gm": vent / gm,
        # This ratio is retained as a descriptive candidate only.  One subject
        # has zero CSF in the extracted map, so its value is recorded as NaN.
        "total_ventricle_over_csf": vent / csf if csf > 0 else math.nan,
    }


def correlation_rows(
    rows: list[dict[str, str]], candidates: list[str], min_cohort_n: int
) -> list[dict[str, object]]:
    age = np.array([float(row["age"]) for row in rows], dtype=float)
    result: list[dict[str, object]] = []
    groups: list[tuple[str, str, list[dict[str, str]]]] = [("split", "all", rows)]
    groups.extend(
        ("split", split, [row for row in rows if row["split"] == split])
        for split in ("train", "calibration", "test")
    )
    groups.extend(
        ("source_cohort", cohort, [row for row in rows if row["source_cohort"] == cohort])
        for cohort in sorted({row["source_cohort"] for row in rows})
        if sum(row["source_cohort"] == cohort for row in rows) >= min_cohort_n
    )
    for group_type, group, group_rows in groups:
        group_age = np.array([float(row["age"]) for row in group_rows], dtype=float)
        for candidate in candidates:
            values = np.array([candidate_values(row)[candidate] for row in group_rows], dtype=float)
            stats = finite_correlation(group_age, values)
            result.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "n": stats["n"],
                    "feature": candidate,
                    "pearson_r": stats["pearson_r"],
                    "spearman_rho": stats["spearman_rho"],
                    "feature_mean": float(np.nanmean(values)),
                    "feature_sd": float(np.nanstd(values)),
                }
            )
    return result


def within_cohort_correlation_rows(
    rows: list[dict[str, str]], candidates: list[str]
) -> list[dict[str, object]]:
    """Correlate age and feature after source-cohort mean removal."""
    result: list[dict[str, object]] = []
    groups = [("all", rows)] + [
        (split, [row for row in rows if row["split"] == split])
        for split in ("train", "calibration", "test")
    ]
    for group, group_rows in groups:
        age = np.array([float(row["age"]) for row in group_rows], dtype=float)
        cohorts = np.array([row["source_cohort"] for row in group_rows])
        for candidate in candidates:
            values = np.array(
                [candidate_values(row)[candidate] for row in group_rows], dtype=float
            )
            valid = np.isfinite(age) & np.isfinite(values)
            centered_age = np.full(len(group_rows), np.nan, dtype=float)
            centered_values = np.full(len(group_rows), np.nan, dtype=float)
            for cohort in np.unique(cohorts[valid]):
                cohort_valid = valid & (cohorts == cohort)
                centered_age[cohort_valid] = age[cohort_valid] - age[cohort_valid].mean()
                centered_values[cohort_valid] = values[cohort_valid] - values[cohort_valid].mean()
            stats = finite_correlation(centered_age, centered_values)
            result.append(
                {
                    "group_type": "within_cohort",
                    "group": group,
                    "n": stats["n"],
                    "feature": candidate,
                    "pearson_r": stats["pearson_r"],
                    "spearman_rho": stats["spearman_rho"],
                    "feature_mean": float(np.nanmean(values)),
                    "feature_sd": float(np.nanstd(values)),
                }
            )
    return result


def design_matrix(rows: list[dict[str, str]], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[candidate_values(row)[name] for name in names] for row in rows], dtype=float)


def feature_composite(
    train: list[dict[str, str]],
    calibration: list[dict[str, str]],
    test: list[dict[str, str]],
    name: str,
    features: tuple[str, ...],
    folds: int,
    seed: int,
) -> dict[str, object]:
    train_age = np.array([float(row["age"]) for row in train], dtype=float)
    calibration_age = np.array([float(row["age"]) for row in calibration], dtype=float)
    test_age = np.array([float(row["age"]) for row in test], dtype=float)
    train_x = design_matrix(train, features)
    calibration_x = design_matrix(calibration, features)
    test_x = design_matrix(test, features)

    # Standardization and fitting happen independently inside every fold.
    oof = np.full(len(train), np.nan, dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit_idx, valid_idx in splitter.split(train_x):
        scaler = StandardScaler().fit(train_x[fit_idx])
        composite_model = LinearRegression().fit(
            scaler.transform(train_x[fit_idx]), train_age[fit_idx]
        )
        oof[valid_idx] = composite_model.predict(scaler.transform(train_x[valid_idx]))

    scaler = StandardScaler().fit(train_x)
    composite_model = LinearRegression().fit(scaler.transform(train_x), train_age)
    full_scores = {
        "train": composite_model.predict(scaler.transform(train_x)),
        "calibration": composite_model.predict(scaler.transform(calibration_x)),
        "test": composite_model.predict(scaler.transform(test_x)),
    }

    def score(age: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
        stats = finite_correlation(age, values)
        return {
            "n": stats["n"],
            "pearson_r": stats["pearson_r"],
            "spearman_rho": stats["spearman_rho"],
        }

    return {
        "name": name,
        "features": list(features),
        "train_oof": score(train_age, oof),
        "train_fit": score(train_age, full_scores["train"]),
        "calibration": score(calibration_age, full_scores["calibration"]),
        "test": score(test_age, full_scores["test"]),
        "standardized_train_coefficients": {
            feature: float(coef) for feature, coef in zip(features, composite_model.coef_)
        },
    }


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--gm-qc", type=Path, required=True)
    parser.add_argument("--derived-features", type=Path, required=True)
    parser.add_argument("--correlations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--min-cohort-n", type=int, default=10)
    args = parser.parse_args()

    csf_rows = read_tsv(args.features)
    if any(row.get("status") != "ok" for row in csf_rows):
        raise SystemExit("CSF feature table contains a non-ok row")
    ids = [row["participant_id"] for row in csf_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("CSF feature table has duplicate participant_id values")

    gm_rows = read_tsv(args.gm_qc)
    gm = {row["participant_id"]: float(row["gm_integral_ml"]) for row in gm_rows}
    if len(gm) != len(gm_rows):
        raise SystemExit("GM QC table has duplicate participant_id values")
    missing_gm = [participant_id for participant_id in ids if participant_id not in gm]
    if missing_gm:
        raise SystemExit(f"missing GM integrals for {len(missing_gm)} CSF rows")
    for row in csf_rows:
        row["gm_integral_ml"] = str(gm[row["participant_id"]])

    split_counts = {split: sum(row["split"] == split for row in csf_rows) for split in EXPECTED_SPLITS}
    if split_counts != EXPECTED_SPLITS:
        raise SystemExit(f"unexpected split counts: {split_counts}")
    if args.folds < 2 or args.folds > len([row for row in csf_rows if row["split"] == "train"]):
        raise SystemExit("--folds must be at least 2 and no larger than train n")

    candidates = list(candidate_values(csf_rows[0]))
    correlations = correlation_rows(csf_rows, candidates, args.min_cohort_n)
    correlations.extend(within_cohort_correlation_rows(csf_rows, candidates))
    correlation_fields = [
        "group_type",
        "group",
        "n",
        "feature",
        "pearson_r",
        "spearman_rho",
        "feature_mean",
        "feature_sd",
    ]
    args.correlations.parent.mkdir(parents=True, exist_ok=True)
    with args.correlations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=correlation_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(correlations)

    train = [row for row in csf_rows if row["split"] == "train"]
    calibration = [row for row in csf_rows if row["split"] == "calibration"]
    test = [row for row in csf_rows if row["split"] == "test"]
    composite_specs = [
        ("raw_csf_ventricle_composite", ("total_csf_mL", "total_ventricle_mL")),
        ("gm_normalized_csf_ventricle_composite", ("total_csf_over_gm", "total_ventricle_over_gm")),
        ("csf_gm_additive_composite", ("total_csf_mL", "gm_integral_ml")),
        ("ventricle_gm_additive_composite", ("total_ventricle_mL", "gm_integral_ml")),
        (
            "csf_ventricle_gm_additive_composite",
            ("total_csf_mL", "total_ventricle_mL", "gm_integral_ml"),
        ),
        ("left_right_ventricle_composite", ("left_ventricle_mL", "right_ventricle_mL")),
    ]
    composites = [
        feature_composite(train, calibration, test, name, features, args.folds, args.seed)
        for name, features in composite_specs
    ]

    report = {
        "protocol": {
            "rows": len(csf_rows),
            "split_counts": split_counts,
            "objective": "direct correlation between feature variation and chronological age",
            "production_predicted_age_or_age_gap_used": False,
            "feature_only_composite_is_secondary": True,
            "primary_statistics": ["Pearson r", "Spearman rho"],
            "within_cohort_diagnostic": "source-cohort mean removed from age and feature; descriptive only",
            "train_selection_rule": "descriptive direct correlations; composite train score is 5-fold out-of-fold",
            "calibration_and_test_role": "confirmation only; no test-based feature selection",
            "gm_qc_source": str(args.gm_qc),
            "min_cohort_n": args.min_cohort_n,
            "folds": args.folds,
            "seed": args.seed,
            "derived_features_output": str(args.derived_features),
        },
        "direct_correlations": correlations,
        "feature_only_composites": composites,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(json_safe(report), indent=2) + "\n")

    derived_fields = [
        "participant_id",
        "split",
        "role",
        "sample_key",
        "age",
        "source_cohort",
        "gm_integral_ml",
        *[candidate for candidate in candidates if candidate != "gm_integral_ml"],
    ]
    args.derived_features.parent.mkdir(parents=True, exist_ok=True)
    with args.derived_features.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=derived_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in csf_rows:
            values = candidate_values(row)
            output_row = {field: row.get(field, "") for field in derived_fields}
            output_row.update(
                {
                    candidate: f"{values[candidate]:.12g}"
                    for candidate in candidates
                    if candidate in derived_fields
                }
            )
            writer.writerow(output_row)

    for row in correlations:
        if row["group_type"] == "split" and row["group"] in {"train", "calibration", "test"}:
            print(
                f"{row['group']:11s} {row['feature']:30s} "
                f"Pearson={row['pearson_r']:+.4f} "
                f"Spearman={row['spearman_rho']:+.4f} n={row['n']}"
            )
    print("\nFeature-only composites (train OOF; full-train fit on calibration/test):")
    for composite in composites:
        print(
            f"{composite['name']:38s} "
            f"train_oof r={composite['train_oof']['pearson_r']:+.4f}/rho={composite['train_oof']['spearman_rho']:+.4f}; "
            f"cal r={composite['calibration']['pearson_r']:+.4f}/rho={composite['calibration']['spearman_rho']:+.4f}; "
            f"test r={composite['test']['pearson_r']:+.4f}/rho={composite['test']['spearman_rho']:+.4f}"
        )


if __name__ == "__main__":
    main()
