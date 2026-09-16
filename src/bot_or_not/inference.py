import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from bot_or_not.artifacts import load_artifact
from bot_or_not.config import MODEL_FEATURES, MODEL_PATH, NHT_CLASS_LABEL


logger = logging.getLogger(__name__)

logger.info("Loading model artifact from %s", MODEL_PATH)

artifact = load_artifact(MODEL_PATH)
pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])

logger.info("Loaded model version %s with threshold %.3f", artifact["version"], threshold)


def predict_traffic(features: Mapping[str, Any]) -> dict[str, str | float]:
    """Predict whether one log line represents NHT or HT."""

    input_data = pd.DataFrame([features])[MODEL_FEATURES]

    class_labels = list(pipeline.classes_)

    if NHT_CLASS_LABEL not in class_labels:
        raise RuntimeError(f"NHT class {NHT_CLASS_LABEL!r} is missing from the model")

    nht_class_index = class_labels.index(NHT_CLASS_LABEL)

    probabilities = pipeline.predict_proba(input_data)[0]
    probability_nht = float(probabilities[nht_class_index])

    prediction = "NHT" if probability_nht >= threshold else "HT"

    return {
        "prediction": prediction,
        "probability_nht": probability_nht,
    }