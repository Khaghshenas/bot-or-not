import joblib
import pandas as pd

from app.config import MODEL_FEATURES, MODEL_PATH, NHT_CLASS_LABEL

from app.schemas import LogLine

# Load the pipeline once when the application starts.
# Loading it for every request would add unnecessary latency.
pipeline = joblib.load(MODEL_PATH)


def predict_traffic(log_line: LogLine) -> dict:
    """Predict whether one log line represents NHT or human traffic (HT)."""

    # Preserve the column names expected by the preprocessing pipeline.
    input_data = pd.DataFrame([log_line.model_dump()])[MODEL_FEATURES]

    prediction = int(pipeline.predict(input_data)[0])
    probabilities = pipeline.predict_proba(input_data)[0]

    # Find the NHT probability by class label instead of assuming its index.
    class_labels = list(pipeline.classes_)

    nht_class_index = class_labels.index(NHT_CLASS_LABEL)
    probability_nht = float(probabilities[nht_class_index])

    return {
        "prediction": "NHT" if prediction == NHT_CLASS_LABEL else "HT",
        "probability_nht": probability_nht,
    }