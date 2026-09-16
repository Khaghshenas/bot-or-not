import numpy as np
import pandas as pd

from bot_or_not.artifacts import load_artifact
from bot_or_not.config import MODEL_FEATURES, MODEL_PATH
from bot_or_not.inference import artifact, pipeline, threshold


MODEL_INPUT = {
    "country_by_ip_address": "IT",
    "region_by_ip_address": "LI",
    "visitor_recognition_type": "ANONYMOUS",
}


def create_input() -> pd.DataFrame:
    return pd.DataFrame([MODEL_INPUT])[MODEL_FEATURES]


def test_model_artifact_exists():
    assert MODEL_PATH.is_file()


def test_model_artifact_has_required_values():
    loaded_artifact = load_artifact(MODEL_PATH)

    assert "pipeline" in loaded_artifact
    assert "threshold" in loaded_artifact
    assert "version" in loaded_artifact
    assert "metadata" in loaded_artifact


def test_threshold_is_valid():
    assert 0 <= threshold <= 1


def test_model_has_expected_pipeline_steps():
    assert "preprocessing" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_model_returns_binary_prediction():
    prediction = pipeline.predict(create_input())

    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1}


def test_probabilities_are_valid():
    probabilities = pipeline.predict_proba(create_input())

    assert probabilities.shape == (1, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities[0].sum(), 1.0)


def test_reloaded_artifact_returns_same_probabilities():
    loaded_artifact = load_artifact(MODEL_PATH)
    loaded_pipeline = loaded_artifact["pipeline"]

    original_probabilities = pipeline.predict_proba(create_input())
    loaded_probabilities = loaded_pipeline.predict_proba(create_input())

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
    )


def test_loaded_artifact_matches_active_artifact():
    loaded_artifact = load_artifact(MODEL_PATH)

    assert loaded_artifact["version"] == artifact["version"]
    assert loaded_artifact["threshold"] == artifact["threshold"]