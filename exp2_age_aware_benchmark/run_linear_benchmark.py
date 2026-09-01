"""Leakage-safe benchmark of age-aware representations for exp2.

The first tier is deliberately dominated by fast linear models so that it can
screen a broad set of representations before expensive GPR refits. Every
fold-local operation is fit using the training part of that fold only:

* variance filtering and StandardScaler;
* ordinary PCA;
* PLS, which uses age labels;
* marginal age-correlation feature screening;
* supervised PCA (age screening followed by PCA);
* ElasticNet regularization.

The exp2 training manifest is split with StratifiedGroupKFold. `group_key`
keeps repeated scans of one participant together. The exp2 calibration set is
never used in CV and is evaluated only after each candidate is fit on the
entire exp2 training manifest. No locked-test file is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from fastbrainage.features import apply_feature_variant


REPO = Path(__file__).resolve().parents[1]
FEATURES_NPZ = REPO / "pipeline/data/exp2/results/gm_features.npz"
TRAIN_TSV = REPO / "pipeline/data/exp2/manifests/train.tsv"
CAL_TSV = REPO / "pipeline/data/exp2/manifests/calibration.tsv"
OUT = REPO / "exp2_age_aware_benchmark/results"
BASE_SEED = 20260806


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    params: dict


def candidate_grid() -> list[Candidate]:
    """Return the first-tier candidates.

    The grids are intentionally broad enough to distinguish representation
    effects from a single arbitrary component count, while remaining feasible
    on the cached 29,852-dimensional matrix.
    """

    candidates: list[Candidate] = []

    for alpha in (1.0, 10.0, 100.0, 1000.0):
        candidates.append(Candidate(
            name=f"raw_ridge_a{alpha:g}", family="raw_ridge",
            params={"alpha": alpha},
        ))

    for n_comp in (32, 64, 128, 256, 435):
        candidates.append(Candidate(
            name=f"pca{n_comp}_ridge_a10", family="pca_ridge",
            params={"components": n_comp, "alpha": 10.0},
        ))

    for n_comp in (5, 10, 20, 32, 64, 128):
        candidates.append(Candidate(
            name=f"pls{n_comp}", family="pls",
            params={"components": n_comp},
        ))

    for n_features in (64, 256, 1024, 4096, 8192, 16000):
        candidates.append(Candidate(
            name=f"corr{n_features}_ridge_a10", family="corr_ridge",
            params={"features": n_features, "alpha": 10.0},
        ))

    # Supervised PCA: keep features with the strongest training-fold age
    # association, then retain a lower-dimensional latent representation.
    for n_features in (1024, 4096, 8192):
        for n_comp in (32, 64, 128):
            candidates.append(Candidate(
                name=f"corr{n_features}_pca{n_comp}_ridge_a10",
                family="corr_pca_ridge",
                params={"features": n_features, "components": n_comp, "alpha": 10.0},
            ))

    # ElasticNet is tested on age-screened features. This is a practical
    # sparse supervised alternative; unregularized Lasso on all 29k highly
    # correlated voxels is needlessly unstable and expensive.
    for n_features in (1024, 4096, 8192):
        for alpha in (0.03, 0.1, 0.3):
            candidates.append(Candidate(
                name=f"corr{n_features}_enet_a{alpha:g}_l05",
                family="corr_elasticnet",
                params={"features": n_features, "alpha": alpha, "l1_ratio": 0.5},
            ))

    return candidates


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def age_band(age: float) -> str:
    if age < 18:
        return "under-18"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    if age < 75:
        return "65-74"
    return "75+"


def load_split(path: Path, archive: dict[str, np.ndarray]) -> dict[str, object]:
    rows = read_manifest(path)
    id_to_idx = {str(pid): i for i, pid in enumerate(archive["participant_id"].tolist())}
    missing = [r["sample_key"] for r in rows if r["sample_key"] not in id_to_idx]
    if missing:
        raise RuntimeError(f"{path} has {len(missing)} IDs missing from feature archive")

    indices = np.asarray([id_to_idx[r["sample_key"]] for r in rows], dtype=int)
    X = np.asarray(archive["features"][indices], dtype=np.float32)
    y_manifest = np.asarray([float(r["age"]) for r in rows], dtype=float)
    y_archive = np.asarray(archive["age"][indices], dtype=float)
    if not np.allclose(y_manifest, y_archive, atol=1e-4, rtol=0):
        raise RuntimeError(f"age mismatch between {path} and feature archive")

    # This operation is subject-wise and does not use the population or age;
    # it is safe to perform before folds. StandardScaler below remains fold
    # local, exactly as in FastBrainAgeModel.
    X = apply_feature_variant(X, "within_subject_z")
    group_values = [r.get("group_key") or r["participant_id"] for r in rows]
    bands = np.asarray([age_band(v) for v in y_manifest], dtype=object)
    return {
        "X": X,
        "y": y_manifest,
        "ids": np.asarray([r["sample_key"] for r in rows], dtype=object),
        "groups": np.asarray(group_values, dtype=object),
        "bands": bands,
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def metrics(y: np.ndarray, pred: np.ndarray, bands: np.ndarray) -> dict[str, float | int]:
    gap = pred - y
    band_bias = {}
    band_mae = {}
    for band in sorted(set(bands.tolist())):
        mask = bands == band
        band_bias[band] = float(np.mean(gap[mask]))
        band_mae[band] = float(np.mean(np.abs(gap[mask])))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(gap))),
        "rmse": float(np.sqrt(np.mean(gap * gap))),
        "pearson_age_pred": safe_corr(y, pred),
        "corr_age_gap": safe_corr(y, gap),
        "r2": float(1.0 - np.sum(gap * gap) / np.sum((y - np.mean(y)) ** 2)),
        "slope_age_pred": float(np.polyfit(y, pred, 1)[0]),
        "worst_abs_band_bias": float(max(abs(v) for v in band_bias.values())),
        "balanced_band_mae": float(np.mean(list(band_mae.values()))),
        "band_bias": band_bias,
        "band_mae": band_mae,
    }


def rank_age_features(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Rank standardized features by absolute training-fold correlation with age."""

    y_centered = y_train - np.mean(y_train)
    # X is standardized column-wise, so the dot product is proportional to
    # Pearson correlation. This avoids materializing a second p-sized matrix.
    scores = np.abs(np.asarray(X_train.T @ y_centered).ravel())
    scores[~np.isfinite(scores)] = -np.inf
    return np.argsort(scores)[::-1]


def fit_predict(
    candidate: Candidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one candidate after fold-local variance filtering/scaling."""

    variances = np.var(X_train, axis=0)
    keep = variances > 0
    Xtr = X_train[:, keep]
    Xva = X_valid[:, keep]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32, copy=False)
    Xva = scaler.transform(Xva).astype(np.float32, copy=False)
    params = candidate.params

    if candidate.family == "raw_ridge":
        model = Ridge(alpha=params["alpha"], solver="lsqr", fit_intercept=True)
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
        detail = {"features_after_variance_filter": int(keep.sum())}

    elif candidate.family == "pca_ridge":
        pca = PCA(
            n_components=min(params["components"], Xtr.shape[0], Xtr.shape[1]),
            svd_solver="randomized", random_state=seed,
        )
        Ztr = pca.fit_transform(Xtr).astype(np.float32, copy=False)
        Zva = pca.transform(Xva).astype(np.float32, copy=False)
        model = Ridge(alpha=params["alpha"], solver="lsqr", fit_intercept=True)
        model.fit(Ztr, y_train)
        pred = model.predict(Zva)
        detail = {
            "features_after_variance_filter": int(keep.sum()),
            "components_fitted": int(pca.n_components_),
            "variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        }

    elif candidate.family == "pls":
        n_comp = min(params["components"], Xtr.shape[0] - 1, Xtr.shape[1])
        model = PLSRegression(n_components=n_comp, scale=False, max_iter=1000)
        model.fit(Xtr, y_train)
        pred = model.predict(Xva).ravel()
        detail = {
            "features_after_variance_filter": int(keep.sum()),
            "components_fitted": int(n_comp),
            "pls_n_iter": [int(v) for v in np.asarray(model.n_iter_).ravel()],
        }

    elif candidate.family in {"corr_ridge", "corr_pca_ridge", "corr_elasticnet"}:
        ranking = rank_age_features(Xtr, y_train)
        n_features = min(params["features"], Xtr.shape[1])
        selected = ranking[:n_features]
        Xtr_sel = Xtr[:, selected]
        Xva_sel = Xva[:, selected]

        if candidate.family == "corr_ridge":
            model = Ridge(alpha=params["alpha"], solver="lsqr", fit_intercept=True)
            model.fit(Xtr_sel, y_train)
            pred = model.predict(Xva_sel)
        elif candidate.family == "corr_pca_ridge":
            n_comp = min(params["components"], Xtr_sel.shape[0], Xtr_sel.shape[1])
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
            Ztr = pca.fit_transform(Xtr_sel).astype(np.float32, copy=False)
            Zva = pca.transform(Xva_sel).astype(np.float32, copy=False)
            model = Ridge(alpha=params["alpha"], solver="lsqr", fit_intercept=True)
            model.fit(Ztr, y_train)
            pred = model.predict(Zva)
        else:
            model = ElasticNet(
                alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                fit_intercept=True, max_iter=3000, tol=1e-3,
                selection="random", random_state=seed,
            )
            model.fit(Xtr_sel, y_train)
            pred = model.predict(Xva_sel)

        detail = {
            "features_after_variance_filter": int(keep.sum()),
            "features_selected": int(n_features),
        }
        if candidate.family == "corr_pca_ridge":
            detail.update({
                "components_fitted": int(pca.n_components_),
                "variance_explained_after_screen": float(np.sum(pca.explained_variance_ratio_)),
            })
        if candidate.family == "corr_elasticnet":
            detail.update({"n_nonzero": int(np.count_nonzero(model.coef_))})

    else:
        raise ValueError(f"unknown candidate family: {candidate.family}")

    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise RuntimeError(f"non-finite predictions from {candidate.name}")
    return pred, detail


def make_folds(data: dict[str, object], n_splits: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = data["X"]
    bands = data["bands"]
    groups = data["groups"]
    folds = list(splitter.split(X, bands, groups))
    for fold_i, (train_idx, valid_idx) in enumerate(folds):
        overlap = set(groups[train_idx].tolist()) & set(groups[valid_idx].tolist())
        if overlap:
            raise RuntimeError(f"group leakage in fold {fold_i}: {len(overlap)} groups")
    return folds


def pc_age_diagnostic(data: dict[str, object], seed: int, max_components: int) -> list[dict[str, object]]:
    """Measure variance and held-out age association of ordinary PCA scores."""

    folds = make_folds(data, 5, seed)
    rows: list[dict[str, object]] = []
    for fold_i, (train_idx, valid_idx) in enumerate(folds):
        Xtr0 = data["X"][train_idx]
        Xva0 = data["X"][valid_idx]
        keep = np.var(Xtr0, axis=0) > 0
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr0[:, keep]).astype(np.float32, copy=False)
        Xva = scaler.transform(Xva0[:, keep]).astype(np.float32, copy=False)
        n_comp = min(max_components, Xtr.shape[0], Xtr.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
        Ztr = pca.fit_transform(Xtr)
        Zva = pca.transform(Xva)
        for j in range(n_comp):
            rows.append({
                "fold": fold_i,
                "component": j + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[j]),
                "cumulative_variance": float(np.sum(pca.explained_variance_ratio_[: j + 1])),
                "train_age_corr": safe_corr(data["y"][train_idx], Ztr[:, j]),
                "valid_age_corr": safe_corr(data["y"][valid_idx], Zva[:, j]),
                "valid_abs_age_corr": abs(safe_corr(data["y"][valid_idx], Zva[:, j])),
            })
    return rows


def run_cv(data: dict[str, object], candidates: list[Candidate], repeats: int, n_splits: int):
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        seed = BASE_SEED + repeat
        folds = make_folds(data, n_splits, seed)
        print(f"repeat {repeat + 1}/{repeats}: {n_splits} group-aware folds", flush=True)
        for fold_i, (train_idx, valid_idx) in enumerate(folds):
            Xtr = data["X"][train_idx]
            Xva = data["X"][valid_idx]
            ytr = data["y"][train_idx]
            yva = data["y"][valid_idx]
            for candidate in candidates:
                started = time.monotonic()
                pred, detail = fit_predict(candidate, Xtr, ytr, Xva, seed)
                row = {
                    "repeat": repeat,
                    "fold": fold_i,
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "seconds": round(time.monotonic() - started, 2),
                    **metrics(yva, pred, data["bands"][valid_idx]),
                    "detail": detail,
                }
                fold_rows.append(row)
                prediction_rows.extend({
                    "repeat": repeat,
                    "fold": fold_i,
                    "candidate": candidate.name,
                    "sample_key": str(sample_id),
                    "age": float(age),
                    "prediction": float(prediction),
                } for sample_id, age, prediction in zip(
                    data["ids"][valid_idx], yva, pred
                ))
                print(
                    f"  repeat={repeat} fold={fold_i} {candidate.name}: "
                    f"MAE={row['mae']:.3f} r_gap={row['corr_age_gap']:.3f} "
                    f"({row['seconds']:.1f}s)",
                    flush=True,
                )
    return fold_rows, prediction_rows


def aggregate_cv(fold_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(fold_rows)
    output = []
    for candidate, group in frame.groupby("candidate", sort=False):
        row = {
            "candidate": candidate,
            "family": group["family"].iloc[0],
            "folds": int(len(group)),
            "mae_mean": float(group["mae"].mean()),
            "mae_std": float(group["mae"].std(ddof=0)),
            "rmse_mean": float(group["rmse"].mean()),
            "pearson_age_pred_mean": float(group["pearson_age_pred"].mean()),
            "corr_age_gap_mean": float(group["corr_age_gap"].mean()),
            "r2_mean": float(group["r2"].mean()),
            "slope_mean": float(group["slope_age_pred"].mean()),
            "worst_abs_band_bias_mean": float(group["worst_abs_band_bias"].mean()),
            "balanced_band_mae_mean": float(group["balanced_band_mae"].mean()),
            "seconds_mean": float(group["seconds"].mean()),
        }
        output.append(row)
    return sorted(output, key=lambda r: (r["mae_mean"], r["worst_abs_band_bias_mean"]))


def fit_calibration(
    train: dict[str, object], calibration: dict[str, object], candidates: list[Candidate]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    predictions = []
    for candidate in candidates:
        started = time.monotonic()
        pred, detail = fit_predict(
            candidate, train["X"], train["y"], calibration["X"], BASE_SEED
        )
        row = {
            "candidate": candidate.name,
            "family": candidate.family,
            "seconds": round(time.monotonic() - started, 2),
            **metrics(calibration["y"], pred, calibration["bands"]),
            "detail": detail,
        }
        rows.append(row)
        predictions.extend({
            "candidate": candidate.name,
            "sample_key": str(sample_id),
            "age": float(age),
            "prediction": float(prediction),
        } for sample_id, age, prediction in zip(
            calibration["ids"], calibration["y"], pred
        ))
        print(
            f"calibration {candidate.name}: MAE={row['mae']:.3f} "
            f"r_gap={row['corr_age_gap']:.3f} ({row['seconds']:.1f}s)",
            flush=True,
        )
    return rows, predictions


def main() -> None:
    global OUT, FEATURES_NPZ, TRAIN_TSV, CAL_TSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES_NPZ)
    parser.add_argument("--train-manifest", type=Path, default=TRAIN_TSV)
    parser.add_argument("--calibration-manifest", type=Path, default=CAL_TSV)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--max-pca-diagnostic", type=int, default=435)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-pc-diagnostic", action="store_true")
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    if args.repeats < 1 or args.splits < 2:
        raise SystemExit("--repeats must be >=1 and --splits must be >=2")

    FEATURES_NPZ = args.features.resolve()
    TRAIN_TSV = args.train_manifest.resolve()
    CAL_TSV = args.calibration_manifest.resolve()
    OUT = Path(args.output_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    archive = np.load(FEATURES_NPZ, allow_pickle=True)
    train = load_split(TRAIN_TSV, archive)
    calibration = load_split(CAL_TSV, archive)
    print(
        f"train={len(train['y'])} rows/{len(set(train['groups'].tolist()))} groups; "
        f"calibration={len(calibration['y'])} rows; features={train['X'].shape[1]}",
        flush=True,
    )

    train_cal_overlap = set(train["groups"].tolist()) & set(calibration["groups"].tolist())
    if train_cal_overlap:
        raise RuntimeError(f"train/calibration group overlap: {len(train_cal_overlap)}")

    all_candidates = candidate_grid()
    by_name = {candidate.name: candidate for candidate in all_candidates}
    if args.candidates is None:
        candidates = all_candidates
    else:
        unknown = [name for name in args.candidates if name not in by_name]
        if unknown:
            raise SystemExit(f"unknown candidate(s): {', '.join(unknown)}")
        candidates = [by_name[name] for name in args.candidates]
    print(f"candidates={len(candidates)}", flush=True)
    fold_rows, prediction_rows = run_cv(train, candidates, args.repeats, args.splits)
    cv_summary = aggregate_cv(fold_rows)
    diagnostic = []
    if not args.skip_pc_diagnostic:
        diagnostic = pc_age_diagnostic(train, BASE_SEED, args.max_pca_diagnostic)

    calibration_rows: list[dict[str, object]] = []
    calibration_predictions: list[dict[str, object]] = []
    if not args.skip_calibration:
        calibration_rows, calibration_predictions = fit_calibration(
            train, calibration, candidates
        )
        calibration_rows = sorted(
            calibration_rows,
            key=lambda r: (r["mae"], r["worst_abs_band_bias"]),
        )

    with (OUT / "cv_fold_metrics.json").open("w") as handle:
        json.dump(fold_rows, handle, indent=2)
    with (OUT / "cv_summary.json").open("w") as handle:
        json.dump(cv_summary, handle, indent=2)
    with (OUT / "pc_age_diagnostic.json").open("w") as handle:
        json.dump(diagnostic, handle, indent=2)
    with (OUT / "calibration_metrics.json").open("w") as handle:
        json.dump(calibration_rows, handle, indent=2)
    pd.DataFrame(prediction_rows).to_csv(OUT / "cv_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(calibration_predictions).to_csv(
        OUT / "calibration_predictions.tsv", sep="\t", index=False
    )
    with (OUT / "run_config.json").open("w") as handle:
        json.dump({
            "features": str(FEATURES_NPZ),
            "train_manifest": str(TRAIN_TSV),
            "calibration_manifest": str(CAL_TSV),
            "feature_variant": "within_subject_z",
            "splitter": "StratifiedGroupKFold",
            "repeats": args.repeats,
            "splits": args.splits,
            "base_seed": BASE_SEED,
            "pc_age_diagnostic": not args.skip_pc_diagnostic,
            "candidates": [asdict(c) for c in candidates],
            "locked_test_read": False,
        }, handle, indent=2)

    print("\nTop CV candidates:", flush=True)
    for row in cv_summary[:12]:
        print(
            f"{row['candidate']:36s} MAE={row['mae_mean']:.3f} "
            f"r_pred={row['pearson_age_pred_mean']:.3f} "
            f"r_gap={row['corr_age_gap_mean']:.3f} "
            f"worst_bias={row['worst_abs_band_bias_mean']:.3f}",
            flush=True,
        )
    if calibration_rows:
        print("\nTop calibration candidates:", flush=True)
        for row in calibration_rows[:12]:
            print(
                f"{row['candidate']:36s} MAE={row['mae']:.3f} "
                f"r_pred={row['pearson_age_pred']:.3f} "
                f"r_gap={row['corr_age_gap']:.3f} "
                f"worst_bias={row['worst_abs_band_bias']:.3f}",
                flush=True,
            )
    print(f"saved results under {OUT}", flush=True)


if __name__ == "__main__":
    main()
