"""Configuration for the frozen FastBrainAge S4_R4 workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FastBrainAgeConfig:
    """Parameters that define the packaged model and feature representation."""

    feature_variant: str = "within_subject_z"
    fwhm_mm: float = 4.0
    resample_mm: float = 4.0
    mask_threshold: float = 0.5
    pca_components: int = 128
    gpr_kernel: str = "rbf"  # legacy default for generic retraining; Exp2 passes matern_white explicitly
    gpr_length_scale: float = 30.0
    gpr_alpha: float = 1e-6
    gpr_restarts: int = 3
    random_state: int = 20260806
    age_expansion_factor: float = 1.010

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "FastBrainAgeConfig":
        fields = cls().__dict__.keys()
        return cls(**{key: values[key] for key in fields if key in values})
