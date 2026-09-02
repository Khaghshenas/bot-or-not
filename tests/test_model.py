import joblib
import numpy as np
import pandas as pd

from app.config import MODEL_FEATURES, MODEL_PATH
from app.prediction import pipeline


MODEL_INPUT = {
    "country_by_ip_address": "IT",
    "region_by_ip_address": "LI",
    "visitor_recognition_type": "ANONYMOUS",
}


def create_input() -> pd.DataFrame:
    return pd.DataFrame([MODEL_INPUT])[MODEL_FEATURES]


def test_model_artifact_exists():
    assert MODEL_PATH.is_file()


def test_model_artifact_loads():
    artifact = joblib.load(MODEL_PATH)
    loaded_pipeline = artifact["pipeline"]

    assert hasattr(loaded_pipeline, "predict")
    assert hasattr(loaded_pipeline, "predict_proba")


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


def test_reloaded_pipeline_returns_same_results():
    loaded_pipeline = joblib.load(MODEL_PATH)["pipeline"]
    model_input = create_input()

    original_prediction = pipeline.predict(model_input)
    loaded_prediction = loaded_pipeline.predict(model_input)

    original_probabilities = pipeline.predict_proba(model_input)
    loaded_probabilities = loaded_pipeline.predict_proba(model_input)

    np.testing.assert_array_equal(original_prediction, loaded_prediction)
    np.testing.assert_allclose(original_probabilities, loaded_probabilities)