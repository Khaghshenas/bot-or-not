import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from bot_or_not.config import RANDOM_STATE
from bot_or_not.features import build_preprocessor


def build_lightgbm_pipeline() -> Pipeline:

    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Build and fit the LightGBM pipeline."""

    pipeline = build_lightgbm_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline