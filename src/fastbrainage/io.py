"""Small, explicit IO helpers for FastBrainAge manifests and archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
    if "participant_id" not in manifest:
        raise ValueError(f"manifest lacks participant_id: {path}")
    manifest["participant_id"] = manifest["participant_id"].astype(str)
    if manifest.participant_id.duplicated().any():
        raise ValueError(f"manifest has duplicate participant_id values: {path}")
    return manifest


def load_features(path: Path, participant_ids: np.ndarray | list[str] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as archive:
        if "features" not in archive or "participant_id" not in archive:
            raise ValueError(f"feature archive must contain features and participant_id: {path}")
        features = archive["features"].astype(np.float32, copy=False)
        source_ids = archive["participant_id"].astype(str)
        metadata = {key: archive[key] for key in archive.files if key not in {"features", "participant_id"}}
    if features.ndim != 2 or len(source_ids) != len(features):
        raise ValueError(f"invalid feature archive shape: {path}")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"feature archive has duplicate participant IDs: {path}")
    if participant_ids is None:
        return features, source_ids, metadata
    wanted = np.asarray(participant_ids, dtype=str)
    positions = {participant_id: index for index, participant_id in enumerate(source_ids)}
    missing = [participant_id for participant_id in wanted if participant_id not in positions]
    if missing:
        raise ValueError(f"feature archive is missing IDs: {missing[:5]}")
    order = [positions[participant_id] for participant_id in wanted]
    return features[order], wanted, metadata


def save_features(path: Path, features: np.ndarray, participant_ids: np.ndarray, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        participant_id=np.asarray(participant_ids, dtype=str),
        **metadata,
    )
