import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from bot_or_not.artifacts import load_artifact, save_artifact
from bot_or_not.config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST, MODEL_FEATURES, RANDOM_STATE
from bot_or_not.data import GROUP_COLUMN, TARGET_SOURCE_COLUMN, create_binary_target, load_data, split_by_session
from bot_or_not.evaluation import evaluate_thresholds, predict_positive_probability, select_best_threshold
from bot_or_not.training import train_model


logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    """Fingerprint the contents of an input dataset."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_training_data(data: pd.DataFrame) -> None:
    """Validate the dataset before creating labels or splitting."""
    required_columns = set(MODEL_FEATURES) | {
        GROUP_COLUMN,
        TARGET_SOURCE_COLUMN,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Training dataset is empty")

    for column in (GROUP_COLUMN, TARGET_SOURCE_COLUMN):
        if data[column].isna().any():
            raise ValueError(f"{column} contains missing values")

        if data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contains empty values")

    # Categorical feature values may be missing, but non-null
    # values must be strings, matching the inference contract.
    for column in MODEL_FEATURES:
        values = data[column].dropna()

        if not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain strings or missing values")


def validate_splits(data: pd.DataFrame, splits) -> None:
    """Check class coverage and session separation."""
    split_names = ("train", "validation", "test")

    for name in split_names:
        labels = getattr(splits, f"y_{name}")

        if set(labels.unique()) != {0, 1}:
            raise ValueError(
                f"{name} split must contain both target classes"
            )

    session_sets = {
        name: set(data.loc[getattr(splits, f"X_{name}").index, GROUP_COLUMN])
        for name in split_names}

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if session_sets[left] & session_sets[right]:
            raise ValueError(
                f"Session leakage between {left} and {right}"
            )


def run_training(data_path: str | Path, output_dir: str | Path, model_version: str) -> Path:
    
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    model_version = model_version.strip()

    if not model_version:
        raise ValueError("model_version must not be empty")

    if not data_path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {data_path}")

    started_at = datetime.now(timezone.utc).isoformat()

    logger.info("Loading dataset: %s", data_path)

    # The input snapshot must remain unchanged throughout the job.
    data_hash = sha256_file(data_path)

    data = load_data(data_path)
    validate_training_data(data)

    data = create_binary_target(data)

    splits = split_by_session(data, random_state=RANDOM_STATE)

    validate_splits(data, splits)

    split_rows = {name: len(getattr(splits, f"y_{name}")) for name in ("train", "validation", "test")}

    logger.info("Split sizes: train=%d validation=%d test=%d",
        split_rows["train"],
        split_rows["validation"],
        split_rows["test"],
    )

    # Refuse to overwrite an existing run.
    output_dir.mkdir(parents=True, exist_ok=False)

    logger.info("Fitting preprocessing and classifier")

    pipeline = train_model(splits.X_train, splits.y_train)

    # Select the threshold exclusively on validation data.
    logger.info("Evaluating validation thresholds")

    validation_probabilities = predict_positive_probability(pipeline, splits.X_validation)

    threshold_results = evaluate_thresholds(
        y_true=splits.y_validation,
        probabilities=validation_probabilities,
        thresholds=np.arange(0.05, 1.0, 0.05),
        false_positive_cost=FALSE_POSITIVE_COST,
        false_negative_cost=FALSE_NEGATIVE_COST,
    )

    threshold = select_best_threshold(threshold_results)

    logger.info("Selected validation threshold: %.3f", threshold)

    # Evaluate the fixed candidate on the held-out test split.
    logger.info("Evaluating candidate on test data")

    test_probabilities = predict_positive_probability(pipeline, splits.X_test)

    test_results = evaluate_thresholds(
        y_true=splits.y_test,
        probabilities=test_probabilities,
        thresholds=[threshold],
        false_positive_cost=FALSE_POSITIVE_COST,
        false_negative_cost=FALSE_NEGATIVE_COST,
    )

    test_metrics = test_results.iloc[0].to_dict()

    logger.info("Test metrics: %s", json.dumps(test_metrics, allow_nan=False))

    metadata = {
        "model_version": model_version,
        "started_at": started_at,
        "data_sha256": data_hash,
        "random_state": RANDOM_STATE,
        "features": list(MODEL_FEATURES),
        "split_strategy": "stratified_group_by_session",
        "split_rows": split_rows,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "dependencies": {
            package: version(package)
            for package in (
                "pandas",
                "scikit-learn",
                "lightgbm",
                "joblib",
            )
        },
        "threshold": threshold,
        "test_metrics": test_metrics,
    }

    logger.info("Saving candidate artifact")

    artifact_path = save_artifact(
        pipeline=pipeline,
        threshold=threshold,
        version=model_version,
        path=output_dir / "model.joblib",
        metadata=metadata,
    )

    # Verify that serialization preserves model behavior.
    logger.info("Reloading and verifying candidate artifact")

    restored = load_artifact(artifact_path)

    restored_probabilities = predict_positive_probability(restored["pipeline"], splits.X_test)

    np.testing.assert_allclose(
        restored_probabilities,
        test_probabilities,
        rtol=0,
        atol=0,
    )

    if restored["threshold"] != threshold:
        raise RuntimeError("Saved threshold differs from selected threshold")

    if restored["version"] != model_version:
        raise RuntimeError("Saved version differs from requested version")

    threshold_results.to_csv(output_dir / "validation_thresholds.csv", index=False)

    report = {
        **metadata,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifact_reload_verified": True,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    # Write last. Partial or failed runs must not have this marker.
    (output_dir / "SUCCESS").write_text("Training completed and artifact verified.\n", encoding="utf-8")

    logger.info("Training completed: %s", artifact_path)

    return artifact_path


def main() -> None:

    parser = argparse.ArgumentParser(description="Train and save a bot-detection candidate.")
    parser.add_argument("--data-path", type=Path, required=True, help="Local CSV snapshot containing features and source labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory. Must not already exist.")

    parser.add_argument("--model-version", required=True, help="Candidate model version, for example v002.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    try:
        run_training(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_version=args.model_version,
        )
    except Exception:
        logger.exception("Training job failed")
        raise


if __name__ == "__main__":
    main()