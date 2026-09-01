"""FastSPM -> PCA -> Gaussian-process brain-age model."""

from __future__ import annotations

from pathlib import Path

import copy

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .config import FastBrainAgeConfig
from .features import apply_feature_variant


def _build_gpr(config: FastBrainAgeConfig) -> GaussianProcessRegressor:
    """Build the configured legacy RBF or current Matern+White GPR."""
    if config.gpr_kernel == "rbf":
        return GaussianProcessRegressor(
            kernel=RBF(config.gpr_length_scale, (1e-7, 1e7)),
            normalize_y=True,
            alpha=config.gpr_alpha,
            n_restarts_optimizer=config.gpr_restarts,
            random_state=config.random_state,
        )
    if config.gpr_kernel == "matern_white":
        return GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(config.gpr_length_scale, (1e-3, 1e5), nu=1.5)
            + WhiteKernel(1.0, (1e-3, 1e3)),
            normalize_y=True,
            alpha=0.0,
            n_restarts_optimizer=max(config.gpr_restarts, 2),
            random_state=config.random_state,
        )
    raise ValueError(f"unknown gpr_kernel: {config.gpr_kernel!r}")


class FastBrainAgeModel:
    """A serializable implementation of the current FastSPM model.

    The GPR is fitted to chronological age.  The selected 1.010 affine age
    expansion is applied at prediction time around the training-age mean; this
    is mathematically equivalent for this normalized-y GPR and keeps the raw
    model output available for auditing.
    """

    def __init__(self, config: FastBrainAgeConfig | None = None):
        self.config = config or FastBrainAgeConfig()
        self.age_mean_: float | None = None
        self.feature_selector_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.gpr_: GaussianProcessRegressor | None = None
        self.n_features_in_: int | None = None
        self.training_n_: int | None = None
        self.training_age_min_: float | None = None
        self.training_age_max_: float | None = None

    @staticmethod
    def _check_features(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be 2-D, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("features contain NaN or infinite values")
        return values

    def fit(self, features: np.ndarray, ages: np.ndarray) -> "FastBrainAgeModel":
        values = self._check_features(features)
        targets = np.asarray(ages, dtype=float)
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("ages must be a 1-D vector with one value per subject")
        if not np.isfinite(targets).all():
            raise ValueError("ages contain NaN or infinite values")
        if len(values) < 3:
            raise ValueError("at least three training subjects are required")

        values = apply_feature_variant(values, self.config.feature_variant)
        self.n_features_in_ = int(values.shape[1])
        self.feature_selector_ = np.var(values, axis=0) > 0
        selected = values[:, self.feature_selector_]
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(selected).astype(np.float32)
        components = min(self.config.pca_components, scaled.shape[0], scaled.shape[1])
        self.pca_ = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=self.config.random_state,
        )
        reduced = self.pca_.fit_transform(scaled).astype(np.float32)
        self.gpr_ = _build_gpr(self.config)
        self.gpr_.fit(reduced, targets)
        self.age_mean_ = float(targets.mean())
        self.training_n_ = int(len(targets))
        self.training_age_min_ = float(targets.min())
        self.training_age_max_ = float(targets.max())
        return self

    def _transform(self, features: np.ndarray) -> np.ndarray:
        if self.feature_selector_ is None or self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("model has not been fitted")
        values = apply_feature_variant(self._check_features(features), self.config.feature_variant)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} input features, got {values.shape[1]}"
            )
        values = values[:, self.feature_selector_]
        values = self.scaler_.transform(values).astype(np.float32)
        return self.pca_.transform(values).astype(np.float32)

    def predict_raw(self, features: np.ndarray, return_std: bool = False):
        if self.gpr_ is None:
            raise RuntimeError("model has not been fitted")
        return self.gpr_.predict(self._transform(features), return_std=return_std)

    def predict(self, features: np.ndarray, return_std: bool = False):
        raw = self.predict_raw(features, return_std=return_std)
        if return_std:
            raw_prediction, raw_std = raw
            prediction = self._expand_age(raw_prediction)
            return prediction, np.abs(self.config.age_expansion_factor) * raw_std
        return self._expand_age(raw)

    def _expand_age(self, raw_prediction: np.ndarray) -> np.ndarray:
        if self.age_mean_ is None:
            raise RuntimeError("model has not been fitted")
        return self.age_mean_ + self.config.age_expansion_factor * (
            np.asarray(raw_prediction) - self.age_mean_
        )

    def metadata(self) -> dict:
        if self.gpr_ is None or self.pca_ is None or self.feature_selector_ is None:
            raise RuntimeError("model has not been fitted")
        kernel_name = {
            "rbf": "RBF-GPR",
            "matern_white": "Matern+White-GPR",
        }.get(self.config.gpr_kernel, self.config.gpr_kernel)
        return {
            "package": "FastBrainAge",
            "package_version": "0.1.0",
            "workflow": f"FastSPM S4_R4 -> within-subject-z -> StandardScaler -> PCA -> {kernel_name}",
            "config": self.config.as_dict(),
            "training_age_mean": self.age_mean_,
            "training_n": self.training_n_,
            "training_age_min": self.training_age_min_,
            "training_age_max": self.training_age_max_,
            "input_features": self.n_features_in_,
            "variance_selected_features": int(self.feature_selector_.sum()),
            "pca_components_fitted": int(self.pca_.n_components_),
            "pca_explained_variance": float(self.pca_.explained_variance_ratio_.sum()),
            "optimized_kernel": str(self.gpr_.kernel_),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "FastBrainAgeModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"not a FastBrainAgeModel artifact: {path}")
        return model

    def refit_regression_head(self, gpr_kernel: str) -> "FastBrainAgeModel":
        """Return a copy with only the GPR head refit under a new kernel.

        Reuses this model's already-fitted feature_selector_/scaler_/pca_ and
        the PCA-space training points already embedded in gpr_.X_train_/
        y_train_ (denormalized via gpr_._y_train_mean/_y_train_std). This is
        for swapping the regression head when the original raw (pre-PCA)
        training features aren't available, only the already-trained model.
        """
        if self.gpr_ is None:
            raise RuntimeError("model has not been fitted")
        reduced = np.asarray(self.gpr_.X_train_, dtype=np.float32)
        targets = np.asarray(self.gpr_.y_train_, dtype=np.float64)
        targets = targets * self.gpr_._y_train_std + self.gpr_._y_train_mean

        new_model = copy.deepcopy(self)
        new_model.config = copy.deepcopy(self.config)
        new_model.config.gpr_kernel = gpr_kernel
        new_model.gpr_ = _build_gpr(new_model.config)
        new_model.gpr_.fit(reduced, targets)
        return new_model
