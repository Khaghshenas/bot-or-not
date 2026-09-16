from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from bot_or_not.config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, NHT_CLASS_LABEL


def predict_positive_probability(pipeline: Pipeline, X: pd.DataFrame, positive_class: int = NHT_CLASS_LABEL) -> np.ndarray:
    """Return probabilities for the configured positive class."""

    class_labels = list(pipeline.classes_)

    if positive_class not in class_labels:
        raise ValueError(f"Positive class {positive_class!r} is not present in the model")

    class_index = class_labels.index(positive_class)

    return pipeline.predict_proba(X)[:, class_index]


def evaluate_thresholds(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Iterable[float],
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> pd.DataFrame:
    """Calculate classification metrics across thresholds."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = list(thresholds)

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty")

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal lengths")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    results: list[dict[str, float | int]] = []

    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1")

        predictions = (probabilities >= threshold).astype(int)

        tp = int(((y_true == 1) & (predictions == 1)).sum())
        fp = int(((y_true == 0) & (predictions == 1)).sum())
        tn = int(((y_true == 0) & (predictions == 0)).sum())
        fn = int(((y_true == 1) & (predictions == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        predicted_positive_rate = (tp + fp) / len(y_true)

        business_cost = false_positive_cost * fp + false_negative_cost * fn

        results.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "specificity": specificity,
                "f1": f1,
                "predicted_positive_rate": predicted_positive_rate,
                "business_cost": business_cost,
            }
        )

    return pd.DataFrame(results)


def select_best_threshold(results: pd.DataFrame) -> float:
    """Select the threshold with the lowest estimated business cost."""

    if results.empty:
        raise ValueError("Threshold results cannot be empty")

    best_index = results["business_cost"].idxmin()

    return float(results.loc[best_index, "threshold"])


def create_error_table(
    X: pd.DataFrame,
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> pd.DataFrame:
    """Create a row-level table for error analysis."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(X) != len(y_true) or len(y_true) != len(probabilities):
        raise ValueError("X, y_true, and probabilities must have equal lengths")

    result = X.copy()
    result["y_true"] = y_true
    result["probability"] = probabilities
    result["prediction"] = (probabilities >= threshold).astype(int)

    result["error_type"] = np.select(
        [
            (result["y_true"] == 0) & (result["prediction"] == 1),
            (result["y_true"] == 1) & (result["prediction"] == 0),
        ],
        [
            "false_positive",
            "false_negative",
        ],
        default="correct",
    )

    return result