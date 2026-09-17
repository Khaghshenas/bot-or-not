
import pytest

from build.lib.bot_or_not.evaluation import evaluate_thresholds


def test_evaluate_thresholds_basic():
    result = evaluate_thresholds(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.4, 0.2],
        [0.5],
    )

    row = result.iloc[0]

    assert row["tp"] == 1
    assert row["fp"] == 1
    assert row["tn"] == 1
    assert row["fn"] == 1
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["f1"] == 0.5

def test_evaluate_thresholds_multiple_thresholds():
    result = evaluate_thresholds(
        [1, 0, 1, 0],
        [0.9, 0.8, 0.4, 0.2],
        [0.3, 0.7],
    )

    assert result["tp"].tolist() == [2, 1]
    assert result["fp"].tolist() == [1, 1]
    assert result["fn"].tolist() == [0, 1]
    assert result["tn"].tolist() == [1, 1]

def test_empty_data():
    with pytest.raises(ValueError):
        evaluate_thresholds([], [], [0.5])


def test_probability_out_of_range():
    with pytest.raises(ValueError):
        evaluate_thresholds([1, 0], [1.2, 0.3], [0.5])


def test_confusion_matrix_counts_equal_observations():
    y_true = [1, 0, 1, 0, 1]
    probabilities = [0.9, 0.8, 0.6, 0.3, 0.2]
    threshold = 0.5

    result = evaluate_thresholds(y_true, probabilities, [threshold])

    row = result.iloc[0]

    assert (row["tp"] + row["fp"] + row["tn"] + row["fn"] == len(y_true))


def test_metrics_are_between_zero_and_one():
    result = evaluate_thresholds(
        y_true=[1, 0, 1, 0],
        probabilities=[0.9, 0.8, 0.4, 0.2],
        thresholds=[0.3, 0.5, 0.7],
    )

    metrics = [
        "precision",
        "recall",
        "false_positive_rate",
        "specificity",
        "f1",
        "predicted_positive_rate",
    ]

    for metric in metrics:
        assert result[metric].between(0, 1).all()