import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from bot_or_not.config import MODEL_FEATURES


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select the columns expected by the model."""

    missing_columns = set(MODEL_FEATURES) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing model features: {sorted(missing_columns)}")

    return data[MODEL_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    """Build categorical preprocessing for the model."""

    return ColumnTransformer(transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), MODEL_FEATURES)], remainder="drop")