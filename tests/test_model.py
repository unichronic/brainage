import numpy as np

from fastbrainage.config import FastBrainAgeConfig
from fastbrainage.features import apply_feature_variant
from fastbrainage.model import FastBrainAgeModel


def test_within_subject_z_has_zero_mean_and_unit_scale():
    values = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    transformed = apply_feature_variant(values, "within_subject_z")
    np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=1), 1.0, atol=1e-6)


def test_model_fit_predict_and_age_expansion():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 40)).astype(np.float32)
    ages = np.linspace(20.0, 80.0, len(features))
    config = FastBrainAgeConfig(pca_components=5, gpr_restarts=0, age_expansion_factor=1.010)
    model = FastBrainAgeModel(config).fit(features, ages)
    raw = model.predict_raw(features[:3])
    prediction, std = model.predict(features[:3], return_std=True)
    expected = ages.mean() + 1.010 * (raw - ages.mean())
    np.testing.assert_allclose(prediction, expected)
    assert np.isfinite(prediction).all()
    assert np.isfinite(std).all()
    assert model.metadata()["pca_components_fitted"] == 5
