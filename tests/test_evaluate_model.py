"""Tests for LBPH model evaluation and output generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluate_model import (
    EXPECTED_IMAGE_SHAPE,
    Prediction,
    EvaluationData,
    build_evaluation_summary,
    calculate_confusion_matrix,
    collect_predictions,
    load_recognizer,
    load_test_data,
    save_confusion_matrix_plot,
    save_distance_distribution_plot,
    save_misclassified_samples_plot,
    save_predictions_csv,
    save_summary_json,
)


def write_valid_test_data(processed_dir: Path) -> None:
    """Write a valid synthetic test set with two samples per class."""

    processed_dir.mkdir(parents=True)
    images = np.zeros((80, *EXPECTED_IMAGE_SHAPE), dtype=np.uint8)
    labels = np.repeat(np.arange(40, dtype=np.int64), 2)
    label_map = {f"s{number}": number - 1 for number in range(1, 41)}
    np.save(processed_dir / "X_test.npy", images)
    np.save(processed_dir / "y_test.npy", labels)
    (processed_dir / "label_map.json").write_text(
        json.dumps(label_map),
        encoding="utf-8",
    )


@pytest.fixture()
def processed_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "processed"
    write_valid_test_data(directory)
    return directory


@pytest.fixture()
def label_map() -> dict[str, int]:
    return {f"s{number}": number - 1 for number in range(1, 41)}


def make_predictions(correct: bool = True) -> list[Prediction]:
    """Create one deterministic prediction per class."""

    predictions = []
    for label in range(40):
        predicted_label = label if correct else (label + 1) % 40
        predictions.append(
            Prediction(
                sample_index=label,
                true_label=label,
                predicted_label=predicted_label,
                true_subject_name=f"s{label + 1}",
                predicted_subject_name=f"s{predicted_label + 1}",
                distance_score=40.0 + label,
                correct=predicted_label == label,
            )
        )
    return predictions


def test_load_valid_test_data(processed_dir: Path) -> None:
    test_data = load_test_data(processed_dir)

    assert test_data.images.shape == (80, 112, 92)
    assert test_data.images.dtype == np.uint8
    assert test_data.labels.shape == (80,)
    assert len(test_data.label_map) == 40


def test_missing_model_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Missing LBPH model file"):
        load_recognizer(tmp_path / "missing.yml")


def test_invalid_test_data_shape_raises_clear_error(processed_dir: Path) -> None:
    np.save(
        processed_dir / "X_test.npy",
        np.zeros((80, 92, 112), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match=r"X_test must have shape"):
        load_test_data(processed_dir)


def test_invalid_test_data_dtype_raises_clear_error(processed_dir: Path) -> None:
    np.save(
        processed_dir / "X_test.npy",
        np.zeros((80, *EXPECTED_IMAGE_SHAPE), dtype=np.float32),
    )

    with pytest.raises(TypeError, match=r"X_test must have dtype uint8"):
        load_test_data(processed_dir)


def test_mismatched_sample_counts_raise_clear_error(processed_dir: Path) -> None:
    np.save(processed_dir / "y_test.npy", np.arange(40, dtype=np.int64))

    with pytest.raises(ValueError, match=r"sample counts differ"):
        load_test_data(processed_dir)


def test_invalid_label_mapping_raises_clear_error(processed_dir: Path) -> None:
    invalid_mapping = {f"s{number}": number - 1 for number in range(1, 41)}
    invalid_mapping["s40"] = 38
    (processed_dir / "label_map.json").write_text(
        json.dumps(invalid_mapping),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"map s1\.\.s40 to labels 0\.\.39"):
        load_test_data(processed_dir)


def test_successful_prediction_collection(label_map: dict[str, int]) -> None:
    class FakeRecognizer:
        def __init__(self) -> None:
            self.next_label = 0

        def predict(self, image: np.ndarray) -> tuple[int, float]:
            label = self.next_label
            self.next_label += 1
            return label, 50.0 + label

    test_data = EvaluationData(
        images=np.zeros((40, *EXPECTED_IMAGE_SHAPE), dtype=np.uint8),
        labels=np.arange(40, dtype=np.int64),
        label_map=label_map,
    )

    predictions = collect_predictions(FakeRecognizer(), test_data)

    assert len(predictions) == 40
    assert predictions[0] == Prediction(0, 0, 0, "s1", "s1", 50.0, True)
    assert predictions[-1].predicted_subject_name == "s40"


def test_accuracy_calculation(label_map: dict[str, int]) -> None:
    predictions = make_predictions()
    predictions[-1] = Prediction(39, 39, 0, "s40", "s1", 90.0, False)

    summary = build_evaluation_summary(predictions, label_map)

    assert summary["overall_accuracy"] == pytest.approx(39 / 40)
    assert summary["correct_predictions"] == 39
    assert summary["incorrect_predictions"] == 1
    assert summary["per_class_accuracy"]["s40"]["accuracy"] == 0.0


def test_confusion_matrix_has_all_40_classes() -> None:
    matrix = calculate_confusion_matrix(make_predictions())

    assert matrix.shape == (40, 40)
    assert np.array_equal(np.diag(matrix), np.ones(40))


def test_prediction_csv_creation(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.csv"
    save_predictions_csv(make_predictions(), output_path)

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 40
    assert rows[0]["true_subject_name"] == "s1"
    assert rows[0]["correct"] == "True"


def test_summary_json_creation(
    label_map: dict[str, int],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "evaluation_summary.json"
    summary = build_evaluation_summary(make_predictions(), label_map)
    save_summary_json(summary, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["test_samples"] == 40
    assert saved["class_count"] == 40
    assert "threshold_analysis" in saved


def test_plot_creation(label_map: dict[str, int], tmp_path: Path) -> None:
    predictions = make_predictions()
    matrix_path = tmp_path / "confusion_matrix.png"
    distance_path = tmp_path / "distance_distribution.png"

    save_confusion_matrix_plot(
        calculate_confusion_matrix(predictions),
        label_map,
        matrix_path,
    )
    save_distance_distribution_plot(predictions, distance_path)

    assert matrix_path.is_file() and matrix_path.stat().st_size > 0
    assert distance_path.is_file() and distance_path.stat().st_size > 0


def test_empty_incorrect_prediction_handling(
    label_map: dict[str, int],
    tmp_path: Path,
) -> None:
    predictions = make_predictions(correct=True)
    summary = build_evaluation_summary(predictions, label_map)
    distance_path = tmp_path / "correct_only_distances.png"
    misclassified_path = tmp_path / "misclassified_samples.png"

    save_distance_distribution_plot(predictions, distance_path)
    plot_created = save_misclassified_samples_plot(
        EvaluationData(
            images=np.zeros((40, *EXPECTED_IMAGE_SHAPE), dtype=np.uint8),
            labels=np.arange(40, dtype=np.int64),
            label_map=label_map,
        ),
        predictions,
        misclassified_path,
    )

    incorrect = summary["distance_statistics"]["incorrect"]
    assert incorrect == {
        "count": 0,
        "mean": None,
        "minimum": None,
        "maximum": None,
        "standard_deviation": None,
    }
    assert (
        summary["threshold_analysis"]["method"]
        == "correct_distance_95th_percentile"
    )
    assert distance_path.is_file()
    assert plot_created is False
    assert not misclassified_path.exists()
