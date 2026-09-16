from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "threshold",
    "version",
    "metadata",
}


def save_artifact(
    pipeline: Pipeline,
    threshold: float,
    version: str,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save the fitted pipeline and its decision threshold."""

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "version": version,
        "metadata": metadata or {},
    }

    joblib.dump(artifact, artifact_path)

    return artifact_path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a saved model artifact."""

    artifact_path = Path(path)

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact_path}")

    artifact = joblib.load(artifact_path)

    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary")

    missing_keys = REQUIRED_ARTIFACT_KEYS - set(artifact)

    if missing_keys:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing_keys)}")

    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise ValueError("Artifact pipeline does not support predict_proba")

    if not 0 <= artifact["threshold"] <= 1:
        raise ValueError("Artifact threshold must be between 0 and 1")

    return artifact