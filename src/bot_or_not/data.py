from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from bot_or_not.config import MODEL_FEATURES, RANDOM_STATE


TARGET_SOURCE_COLUMN = "ua_agent_class"
TARGET_COLUMN = "is_nht"
GROUP_COLUMN = "session_id"

# This assumed mapping must be confirmed by the data owner.
HUMAN_CLASSES = {"Browser", "Browser Webview", "Mobile App"}


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_data(path: str | Path) -> pd.DataFrame:

    return pd.read_csv(path)


def create_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    """Convert the original eight classes into HT=0 and NHT=1."""

    if TARGET_SOURCE_COLUMN not in data.columns:
        raise ValueError(f"Missing target source column: {TARGET_SOURCE_COLUMN}")

    if data[TARGET_SOURCE_COLUMN].isna().any():
        raise ValueError("Target source column contains missing values")

    result = data.copy()

    result[TARGET_COLUMN] = (~result[TARGET_SOURCE_COLUMN].isin(HUMAN_CLASSES)).astype(int)

    return result


def split_by_session(data: pd.DataFrame, random_state: int = RANDOM_STATE) -> DataSplits:
    """Create session-separated train, validation, and test sets."""

    required_columns = set(MODEL_FEATURES) | {TARGET_COLUMN, GROUP_COLUMN}

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    X = data[MODEL_FEATURES]
    y = data[TARGET_COLUMN]
    groups = data[GROUP_COLUMN]

    # Reserve approximately 20% for the final test set.
    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    train_validation_index, test_index = next(outer_splitter.split(X, y, groups))

    X_train_validation = X.iloc[train_validation_index]
    y_train_validation = y.iloc[train_validation_index]
    groups_train_validation = groups.iloc[train_validation_index]

    # Split the remaining 80% into approximately 60% train and 20% validation.
    inner_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=random_state,
    )

    train_index_local, validation_index_local = next(inner_splitter.split(X_train_validation, y_train_validation, groups_train_validation))

    return DataSplits(
        X_train=X_train_validation.iloc[train_index_local],
        X_validation=X_train_validation.iloc[validation_index_local],
        X_test=X.iloc[test_index],
        y_train=y_train_validation.iloc[train_index_local],
        y_validation=y_train_validation.iloc[validation_index_local],
        y_test=y.iloc[test_index],
    )