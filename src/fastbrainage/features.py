"""S4_R4 feature extraction from FastSPM GM maps.

The released representation smooths each modulated, normalized map at 4 mm,
resamples it to the fixed mask geometry, and retains the masked voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import gaussian_filter


@dataclass
class FeatureExtractionConfig:
    mask_path: Path
    fwhm_mm: float = 4.0
    resample_mm: float = 4.0
    mask_threshold: float = 0.5


def apply_feature_variant(features: np.ndarray, variant: str) -> np.ndarray:
    """Apply the selected per-subject normalization used before PCA."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"features must be 2-D, got {values.shape}")
    values = values.copy()
    row_mean = values.mean(axis=1, keepdims=True)
    if variant == "raw":
        return values
    if variant == "within_subject_center":
        return values - row_mean
    if variant == "within_subject_z":
        row_std = values.std(axis=1, keepdims=True)
        return (values - row_mean) / np.maximum(row_std, 1e-6)
    if variant == "mean_normalized":
        return values / np.maximum(row_mean, 1e-6)
    if variant == "rms_normalized":
        rms = np.sqrt(np.mean(values * values, axis=1, keepdims=True))
        return values / np.maximum(rms, 1e-6)
    raise ValueError(f"unknown feature variant: {variant}")


class FastSPMFeatureExtractor:
    """Extract the exact fixed-grid voxel representation used by FastBrainAge."""

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        mask_source = nib.load(str(config.mask_path))
        self.mask_image = resample_to_output(
            mask_source, [config.resample_mm] * 3, order=1
        )
        self.mask = self.mask_image.get_fdata(dtype=np.float32) > config.mask_threshold
        self.target = (self.mask_image.shape, self.mask_image.affine)
        if not np.any(self.mask):
            raise ValueError(f"brain mask is empty: {config.mask_path}")

    @property
    def feature_count(self) -> int:
        return int(self.mask.sum())

    def extract_map(self, map_path: Path) -> tuple[np.ndarray, dict]:
        """Extract one map and return features plus lightweight QC metadata."""

        image = nib.load(str(map_path))
        data = image.get_fdata(dtype=np.float32)
        if data.ndim != 3 or not np.isfinite(data).all() or not np.any(data):
            raise ValueError(f"invalid FastSPM map: {map_path}")
        zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if np.any(zooms <= 0) or not np.isfinite(zooms).all():
            raise ValueError(f"invalid voxel sizes in map: {map_path}")
        sigma = self.config.fwhm_mm / (
            2.0 * np.sqrt(2.0 * np.log(2.0)) * zooms
        )
        smooth = gaussian_filter(data, sigma=sigma, mode="nearest")
        sampled = resample_from_to(
            nib.Nifti1Image(smooth, image.affine), self.target, order=1
        )
        values = sampled.get_fdata(dtype=np.float32)[self.mask]
        if not np.isfinite(values).all() or not np.any(values):
            raise ValueError(f"invalid FastSPM features: {map_path}")
        qc = {
            "map": str(map_path),
            "shape": "x".join(map(str, image.shape)),
            "gm_integral_ml": float(
                data.sum() * abs(np.linalg.det(image.affine[:3, :3])) / 1000.0
            ),
            "feature_mean": float(values.mean()),
            "feature_std": float(values.std()),
        }
        return values.astype(np.float32, copy=False), qc

    def extract_paths(
        self, participant_ids: list[str] | np.ndarray, map_paths: list[Path]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if len(participant_ids) != len(map_paths):
            raise ValueError("participant IDs and map paths have different lengths")
        features = np.empty((len(map_paths), self.feature_count), dtype=np.float32)
        qc_rows = []
        for index, (participant_id, map_path) in enumerate(
            zip(participant_ids, map_paths)
        ):
            features[index], qc = self.extract_map(Path(map_path))
            qc_rows.append({"participant_id": str(participant_id), **qc})
            if (index + 1) % 25 == 0 or index + 1 == len(map_paths):
                print(
                    f"FastBrainAge S4_R4 features {index + 1}/{len(map_paths)}",
                    flush=True,
                )
        return features, pd.DataFrame(qc_rows)

    def geometry(self) -> dict:
        return {
            "target_shape": list(self.mask_image.shape),
            "target_affine": self.mask_image.affine.tolist(),
            "feature_count": self.feature_count,
            "smooth_fwhm_mm": self.config.fwhm_mm,
            "resample_mm": self.config.resample_mm,
            "mask_threshold": self.config.mask_threshold,
        }


def resolve_map_paths(
    manifest: pd.DataFrame, maps_dir: Path | None = None, manifest_path: Path | None = None
) -> list[Path]:
    """Resolve a manifest's map_path column or the standard FastSPM layout."""

    if "participant_id" not in manifest:
        raise ValueError("manifest must contain participant_id")
    base = manifest_path.parent if manifest_path else Path.cwd()
    if "map_path" in manifest:
        paths = []
        for value in manifest.map_path.astype(str):
            path = Path(value)
            paths.append(path if path.is_absolute() else base / path)
        return paths
    if maps_dir is None:
        raise ValueError("provide --maps-dir or a map_path column")
    paths = []
    for pid in manifest.participant_id.astype(str):
        candidates = [
            maps_dir / f"sub-{pid}" / f"sub-{pid}_mwc1.nii",
            maps_dir / f"sub-{pid}" / f"sub-{pid}_mwc1.nii.gz",
            maps_dir / f"sub-{pid}_mwc1.nii",
            maps_dir / f"sub-{pid}_mwc1.nii.gz",
        ]
        existing = next((path for path in candidates if path.exists()), None)
        if existing is None:
            raise FileNotFoundError(
                f"no FastSPM mwc1 map found for {pid}; checked: {candidates}"
            )
        paths.append(existing)
    return paths
