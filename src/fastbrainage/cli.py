"""Command-line interface for the FastBrainAge container."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .bias_correction import AgeBiasCorrection, fit_age_bias_correction
from .config import FastBrainAgeConfig
from .features import (
    FastSPMFeatureExtractor,
    FeatureExtractionConfig,
    resolve_map_paths,
)
from .io import load_features, read_manifest, save_features
from .model import FastBrainAgeModel


def _default_mask() -> Path:
    candidates = [
        Path(os.environ["FASTBRAINAGE_MASK"])
        if os.environ.get("FASTBRAINAGE_MASK")
        else None,
        Path("/opt/fastbrainage/assets/brainmask_12.8.nii"),
        Path(__file__).resolve().parents[2] / "assets" / "brainmask_12.8.nii",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    # Keep a useful path in argparse help/error messages when the source tree
    # is incomplete; the extractor will raise the concrete file error.
    return Path("/opt/fastbrainage/assets/brainmask_12.8.nii")


def extract_features_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.manifest)
    map_paths = resolve_map_paths(manifest, args.maps_dir, args.manifest)
    extractor = FastSPMFeatureExtractor(
        FeatureExtractionConfig(
            mask_path=args.mask,
            fwhm_mm=args.fwhm_mm,
            resample_mm=args.resample_mm,
            mask_threshold=args.mask_threshold,
        )
    )
    features, qc = extractor.extract_paths(manifest.participant_id.to_numpy(str), map_paths)
    metadata = {}
    if "age" in manifest:
        metadata["age"] = manifest.age.to_numpy(np.float32)
    for column in ("role", "study", "site", "sex"):
        if column in manifest:
            metadata[column] = manifest[column].astype(str).to_numpy()
    save_features(args.output, features, manifest.participant_id.to_numpy(str), **metadata)
    qc.to_csv(args.output.with_suffix(".qc.tsv"), sep="\t", index=False)
    args.output.with_suffix(".geometry.json").write_text(
        json.dumps(extractor.geometry(), indent=2) + "\n"
    )
    print(f"saved {args.output} {features.shape}")


def train_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.manifest)
    if "age" not in manifest:
        raise ValueError("training manifest must contain age")
    features, feature_ids, _ = load_features(
        args.features, manifest.participant_id.to_numpy(str)
    )
    config = FastBrainAgeConfig(
        pca_components=args.pca_components,
        gpr_kernel=args.gpr_kernel,
        gpr_restarts=args.gpr_restarts,
        random_state=args.random_state,
        age_expansion_factor=args.age_expansion_factor,
    )
    model = FastBrainAgeModel(config).fit(features, manifest.age.to_numpy(float))
    model.save(args.output_model)
    metadata = model.metadata()
    metadata["training_manifest_name"] = args.manifest.name
    metadata["training_feature_archive_name"] = args.features.name
    args.output_model.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    model = FastBrainAgeModel.load(args.model)
    manifest = read_manifest(args.manifest) if args.manifest else None
    ids = manifest.participant_id.to_numpy(str) if manifest is not None else None
    features, feature_ids, _ = load_features(args.features, ids)
    prediction, std = model.predict(features, return_std=True)
    output = manifest.copy() if manifest is not None else None
    if output is None:
        import pandas as pd

        output = pd.DataFrame({"participant_id": feature_ids})
    output["predicted_age"] = prediction
    output["prediction_std"] = std
    if "age" in output:
        output["gap"] = output.predicted_age - output.age.to_numpy(float)
        if args.bias_correction:
            correction = AgeBiasCorrection.load(args.bias_correction)
            output["corrected_predicted_age"] = correction.apply(
                output.predicted_age.to_numpy(float), output.age.to_numpy(float)
            )
            output["corrected_gap"] = output.corrected_predicted_age - output.age.to_numpy(float)
    elif args.bias_correction:
        raise ValueError("--bias-correction requires a manifest with a known age column")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    print(f"saved {args.output} n={len(output)}")


def fit_bias_correction_command(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.calibration_predictions)
    if "age" not in manifest or "predicted_age" not in manifest:
        raise ValueError("calibration predictions must contain age and predicted_age columns")
    correction = fit_age_bias_correction(
        manifest.age.to_numpy(float), manifest.predicted_age.to_numpy(float)
    )
    correction.save(args.output)
    print(json.dumps(correction.__dict__, indent=2))


def info_command(args: argparse.Namespace) -> None:
    print(json.dumps(FastBrainAgeModel.load(args.model).metadata(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastbrainage")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-features", help="extract S4_R4 features from FastSPM mwc1 maps")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--maps-dir", type=Path)
    extract.add_argument("--mask", type=Path, default=_default_mask())
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--fwhm-mm", type=float, default=4.0)
    extract.add_argument("--resample-mm", type=float, default=4.0)
    extract.add_argument("--mask-threshold", type=float, default=0.5)
    extract.set_defaults(function=extract_features_command)

    train = sub.add_parser("train", help="fit and serialize a FastBrainAge model")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--features", type=Path, required=True)
    train.add_argument("--output-model", type=Path, required=True)
    train.add_argument("--pca-components", type=int, default=128)
    train.add_argument("--gpr-kernel", choices=("rbf", "matern_white"), default="rbf")
    train.add_argument("--gpr-restarts", type=int, default=3)
    train.add_argument("--random-state", type=int, default=20260806)
    train.add_argument("--age-expansion-factor", type=float, default=1.010)
    train.set_defaults(function=train_command)

    predict = sub.add_parser("predict", help="predict ages from a feature archive")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--features", type=Path, required=True)
    predict.add_argument("--manifest", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--bias-correction", type=Path,
        help="AgeBiasCorrection JSON (see fit-bias-correction); requires a manifest with age",
    )
    predict.set_defaults(function=predict_command)

    fit_bias = sub.add_parser(
        "fit-bias-correction",
        help="fit a calibration-split affine age-bias correction (predicted = intercept + slope*age)",
    )
    fit_bias.add_argument(
        "--calibration-predictions", type=Path, required=True,
        help="TSV with age and predicted_age columns for the model's calibration split",
    )
    fit_bias.add_argument("--output", type=Path, required=True)
    fit_bias.set_defaults(function=fit_bias_correction_command)

    info = sub.add_parser("model-info", help="show serialized model metadata")
    info.add_argument("--model", type=Path, required=True)
    info.set_defaults(function=info_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)
