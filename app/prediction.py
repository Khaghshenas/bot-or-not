import logging

import joblib
import pandas as pd

from app.config import MODEL_FEATURES, MODEL_PATH, NHT_CLASS_LABEL
from app.schemas import LogLine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the pipeline once when the application starts.
logger.info("Loading model from %s", MODEL_PATH)

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])

if not 0 <= threshold <= 1:
    raise RuntimeError(f"Invalid prediction threshold: {threshold}")

# Cache the NHT class index and fail fast if the model has an unexpected class mapping.
try:
    _nht_class_index = list(pipeline.classes_).index(NHT_CLASS_LABEL)
except ValueError as error:
    raise RuntimeError(
        f"NHT class {NHT_CLASS_LABEL!r} is missing from the model"
    ) from error

logger.info("Model artifact loaded successfully with threshold %.4f", threshold)



def predict_traffic(log_line: LogLine) -> dict:
    """Predict whether one log line represents NHT or HT."""

    # Preserve the column names expected by the preprocessing pipeline.
    input_data = pd.DataFrame([log_line.model_dump()])[MODEL_FEATURES]

    # Run preprocessing and inference once, then derive the prediction class 
    # from the returned probabilities.
    probabilities = pipeline.predict_proba(input_data)[0]

    predicted_class_index = probabilities.argmax()
    #prediction = int(pipeline.classes_[predicted_class_index])

    # Find the NHT probability by class label instead of assuming its index.
    probability_nht = float(probabilities[_nht_class_index])

    # Apply the tuned threshold instead of using the highest probability.
    prediction = "NHT" if probability_nht >= threshold else "HT"

    return {
        "prediction": "NHT" if prediction == NHT_CLASS_LABEL else "HT",
        "probability_nht": probability_nht,
    }