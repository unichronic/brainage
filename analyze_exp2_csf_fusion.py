#!/usr/bin/env python3
"""Run the pre-specified, leakage-safe exp2 CSF fusion evaluation.

The frozen exp2 prediction is augmented with total CSF and total lateral-
ventricle volume.  Coefficients are fit only on the 554-row calibration
split; the 561-row locked test is evaluated once and never used for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


EXPECTED = {"train": 2742, "calibration": 554, "test": 561}
BANDS = [
    ("under18", 0.0, 18.0),
    ("18-24", 18.0, 25.0),
    ("25-34", 25.0, 35.0),
    ("35-44", 35.0, 45.0),
    ("45-54", 45.0, 55.0),
    ("55-64", 55.0, 65.0),
    ("65-74", 65.0, 75.0),
    ("75+", 75.0, float("inf")),
]


def read_table(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def age_band(age: float) -> str:
    for name, low, high in BANDS:
        if low <= age < high:
            return name
    raise ValueError(f"age outside configured bands: {age}")


def metrics(age: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    gap = prediction - age
    gap_age_corr = float(np.corrcoef(age, gap)[0, 1])
    return {
        "mae": float(np.mean(np.abs(gap))),
        "rmse": float(np.sqrt(np.mean(gap**2))),
        "bias": float(np.mean(gap)),
        "age_prediction_corr": float(np.corrcoef(age, prediction)[0, 1]),
        "gap_age_corr": gap_age_corr,
        "abs_gap_age_corr": abs(gap_age_corr),
    }


def band_metrics(age: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float | int]]:
    gap = prediction - age
    result: dict[str, dict[str, float | int]] = {}
    for name, low, high in BANDS:
        mask = (age >= low) & (age < high)
        if not np.any(mask):
            continue
        result[name] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(gap[mask]))),
            "bias": float(np.mean(gap[mask])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-pred", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--model-label",
        default="models/exp2.joblib (Matern+White)",
    )
    args = parser.parse_args()

    manifest_rows = read_table(args.manifest, "\t")
    prediction_rows = read_table(args.raw_pred, "\t")
    feature_rows = read_table(args.features, "\t")
    manifest = {row["participant_id"]: row for row in manifest_rows}
    predictions = {row["participant_id"]: row for row in prediction_rows}
    features = {row["participant_id"]: row for row in feature_rows}
    if len(manifest) != len(manifest_rows):
        raise SystemExit("manifest has duplicate participant_id values")
    if len(predictions) != len(prediction_rows):
        raise SystemExit("prediction table has duplicate participant_id values")
    if len(features) != len(feature_rows):
        raise SystemExit("feature table has duplicate participant_id values")
    expected_ids = set(manifest)
    if set(predictions) != expected_ids:
        raise SystemExit("prediction IDs do not exactly match the exp2 manifest")
    if set(features) != expected_ids:
        raise SystemExit("CSF feature IDs do not exactly match the exp2 manifest")

    split_counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in EXPECTED
    }
    if split_counts != EXPECTED:
        raise SystemExit(f"unexpected split counts: {split_counts}")

    ordered = manifest_rows
    ages = np.array([float(row["age"]) for row in ordered])
    raw = np.array([float(predictions[row["participant_id"]]["predicted_age"]) for row in ordered])
    csf = np.array([float(features[row["participant_id"]]["total_csf_mL"]) for row in ordered])
    vent = np.array([float(features[row["participant_id"]]["total_ventricle_mL"]) for row in ordered])
    if not np.isfinite(np.c_[ages, raw, csf, vent]).all():
        raise SystemExit("joined data contain non-finite values")

    split = np.array([row["split"] for row in ordered])
    calibration = split == "calibration"
    locked_test = split == "test"
    if calibration.sum() != EXPECTED["calibration"] or locked_test.sum() != EXPECTED["test"]:
        raise SystemExit("calibration/test sizes do not match exp2")

    X_cal = np.c_[raw[calibration], csf[calibration], vent[calibration]]
    X_test = np.c_[raw[locked_test], csf[locked_test], vent[locked_test]]
    y_cal = ages[calibration]
    y_test = ages[locked_test]

    affine = LinearRegression().fit(raw[calibration, None], y_cal)
    fusion = LinearRegression().fit(X_cal, y_cal)
    raw_test = raw[locked_test]
    affine_test = affine.predict(raw_test[:, None])
    fusion_test = fusion.predict(X_test)

    models = {
        "raw_exp2": raw_test,
        "baseline_affine_calibration_only": affine_test,
        "csf_ventricle_fusion_calibration_only": fusion_test,
    }
    report = {
        "protocol": {
            "manifest_rows": len(ordered),
            "split_counts": split_counts,
            "calibration_used_for_fit": int(calibration.sum()),
            "locked_test_evaluated_once": int(locked_test.sum()),
            "model": args.model_label,
            "features": ["raw_exp2_prediction", "total_csf_mL", "total_ventricle_mL"],
            "primary_analysis": "direct CSF/ventricle feature correlation with chronological age",
            "feature_age_report": "csf_experiment/csf_feature_age_signal_report_3857.json",
            "fusion_analysis_role": "secondary diagnostic; not used to rank the age signal in the features",
        },
        "models": {},
        "coefficients": {
            "baseline_affine": {
                "coef_raw_pred": float(affine.coef_[0]),
                "intercept": float(affine.intercept_),
            },
            "csf_ventricle_fusion": {
                "feature_order": ["raw_exp2_prediction", "total_csf_mL", "total_ventricle_mL"],
                "coef": [float(x) for x in fusion.coef_],
                "intercept": float(fusion.intercept_),
            },
        },
    }
    for name, prediction in models.items():
        report["models"][name] = {
            **metrics(y_test, prediction),
            "age_bands": band_metrics(y_test, prediction),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    prediction_rows_out = []
    test_index = 0
    for index, (row, age_value, raw_value) in enumerate(zip(ordered, ages, raw)):
        out = {
            "participant_id": row["participant_id"],
            "split": row["split"],
            "role": row.get("role", ""),
            "age": f"{age_value:.9g}",
            "age_band": age_band(age_value),
            "raw_exp2_predicted_age": f"{raw_value:.9g}",
            "total_csf_mL": f"{csf[index]:.9g}",
            "total_ventricle_mL": f"{vent[index]:.9g}",
            "baseline_affine_predicted_age": "",
            "csf_ventricle_fusion_predicted_age": "",
        }
        if row["split"] == "test":
            out["baseline_affine_predicted_age"] = f"{affine_test[test_index]:.9g}"
            out["csf_ventricle_fusion_predicted_age"] = f"{fusion_test[test_index]:.9g}"
            test_index += 1
        prediction_rows_out.append(out)
    with args.predictions.open("w", newline="") as handle:
        fields = list(prediction_rows_out[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows_out)

    print(json.dumps({name: report["models"][name] for name in models}, indent=2))


if __name__ == "__main__":
    main()
