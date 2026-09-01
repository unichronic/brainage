"""Calibration-split affine age-bias correction.

The correction is fitted only on a calibration split and can then be applied
to predictions from any held-out split without refitting on those labels.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class AgeBiasCorrection:
    intercept: float
    slope: float
    n_calibration: int

    def apply(self, predicted_age: np.ndarray, chronological_age: np.ndarray) -> np.ndarray:
        predicted_age = np.asarray(predicted_age, dtype=float)
        chronological_age = np.asarray(chronological_age, dtype=float)
        return predicted_age - self.intercept - (self.slope - 1.0) * chronological_age

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "AgeBiasCorrection":
        return cls(**json.loads(Path(path).read_text()))


def fit_age_bias_correction(chronological_age: np.ndarray, predicted_age: np.ndarray) -> AgeBiasCorrection:
    """Fit predicted = intercept + slope*age via OLS on a calibration split."""
    chronological_age = np.asarray(chronological_age, dtype=float)
    predicted_age = np.asarray(predicted_age, dtype=float)
    if len(chronological_age) != len(predicted_age):
        raise ValueError("age and predicted_age must have the same length")
    if len(chronological_age) < 3:
        raise ValueError("at least three calibration subjects are required")
    slope, intercept = np.polyfit(chronological_age, predicted_age, 1)
    return AgeBiasCorrection(intercept=float(intercept), slope=float(slope), n_calibration=int(len(chronological_age)))
