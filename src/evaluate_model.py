"""Evaluate a trained OpenCV LBPH recognizer on prepared test data."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import matplotlib
import numpy as np
from numpy.typing import NDArray


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lbph_model.yml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

EXPECTED_IMAGE_SHAPE = (112, 92)
EXPECTED_CLASS_COUNT = 40


class LBPHRecognizer(Protocol):
    """Prediction operation used from an OpenCV LBPH recognizer."""

    def predict(self, src: NDArray[np.uint8]) -> tuple[int, float]:
        """Predict a label and LBPH distance for one image."""


@dataclass(frozen=True)
class EvaluationData:
    """Validated test inputs and subject-label mapping."""

    images: NDArray[np.uint8]
    labels: NDArray[np.integer[Any]]
    label_map: dict[str, int]


@dataclass(frozen=True)
class Prediction:
    """One test-image prediction and its evaluation fields."""

    sample_index: int
    true_label: int
    predicted_label: int
    true_subject_name: str
    predicted_subject_name: str
    distance_score: float
    correct: bool


def _load_label_map(label_map_path: Path) -> dict[str, int]:
    """Load and structurally validate a subject-to-label JSON mapping."""

    try:
        mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in label map: {label_map_path}") from exc

    if not isinstance(mapping, dict):
        raise ValueError(f"Label map must be a JSON object: {label_map_path}")
    if not all(
        isinstance(subject, str)
        and isinstance(label, int)
        and not isinstance(label, bool)
        for subject, label in mapping.items()
    ):
        raise ValueError("Label map must contain string keys and integer values")

    expected_mapping = {
        f"s{subject_number}": subject_number - 1
        for subject_number in range(1, EXPECTED_CLASS_COUNT + 1)
    }
    if mapping != expected_mapping:
        raise ValueError("label_map.json must map s1..s40 to labels 0..39")

    return mapping


def validate_test_data(
    images: NDArray[Any],
    labels: NDArray[Any],
    label_map: dict[str, int],
) -> None:
    """Validate test array shapes, dtypes, sample counts, and label coverage."""

    if images.ndim != 3 or images.shape[1:] != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"X_test must have shape (N, {EXPECTED_IMAGE_SHAPE[0]}, "
            f"{EXPECTED_IMAGE_SHAPE[1]}), got {images.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"y_test must have shape (N,), got {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "X_test and y_test sample counts differ: "
            f"{images.shape[0]} != {labels.shape[0]}"
        )
    if images.dtype != np.uint8:
        raise TypeError(f"X_test must have dtype uint8, got {images.dtype}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"y_test must contain integer labels, got {labels.dtype}")

    expected_labels = set(label_map.values())
    test_labels = {int(label) for label in np.unique(labels)}
    if len(label_map) != EXPECTED_CLASS_COUNT or test_labels != expected_labels:
        raise ValueError(
            "Test labels must match all 40 labels defined by label_map.json"
        )


def load_test_data(processed_dir: Path) -> EvaluationData:
    """Load and validate the prepared test arrays and label map."""

    images_path = processed_dir / "X_test.npy"
    labels_path = processed_dir / "y_test.npy"
    label_map_path = processed_dir / "label_map.json"
    required_paths = (images_path, labels_path, label_map_path)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required evaluation file(s): {missing}")

    images = np.load(images_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    label_map = _load_label_map(label_map_path)
    validate_test_data(images, labels, label_map)
    return EvaluationData(images=images, labels=labels, label_map=label_map)


def load_recognizer(model_path: Path) -> LBPHRecognizer:
    """Create an LBPH recognizer and load a persisted trained model."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing LBPH model file: {model_path}")

    face_module = getattr(cv2, "face", None)
    factory = getattr(face_module, "LBPHFaceRecognizer_create", None)
    if factory is None:
        raise RuntimeError(
            "OpenCV LBPH support is unavailable; install opencv-contrib-python"
        )

    recognizer = factory()
    try:
        recognizer.read(str(model_path))
    except cv2.error as exc:
        raise ValueError(f"Could not load LBPH model: {model_path}") from exc

    if not recognizer.getHistograms():
        raise ValueError(f"LBPH model contains no trained histograms: {model_path}")
    return recognizer


def collect_predictions(
    recognizer: LBPHRecognizer,
    test_data: EvaluationData,
) -> list[Prediction]:
    """Run LBPH prediction for every test image."""

    label_to_subject = {
        label: subject for subject, label in test_data.label_map.items()
    }
    predictions: list[Prediction] = []

    for sample_index, (image, true_label_value) in enumerate(
        zip(test_data.images, test_data.labels, strict=True)
    ):
        predicted_label_value, distance_value = recognizer.predict(image)
        true_label = int(true_label_value)
        predicted_label = int(predicted_label_value)
        if predicted_label not in label_to_subject:
            raise ValueError(
                f"Model predicted label {predicted_label}, which is absent from "
                "label_map.json"
            )

        predictions.append(
            Prediction(
                sample_index=sample_index,
                true_label=true_label,
                predicted_label=predicted_label,
                true_subject_name=label_to_subject[true_label],
                predicted_subject_name=label_to_subject[predicted_label],
                distance_score=float(distance_value),
                correct=predicted_label == true_label,
            )
        )

    return predictions


def calculate_confusion_matrix(
    predictions: Sequence[Prediction],
    class_count: int = EXPECTED_CLASS_COUNT,
) -> NDArray[np.int64]:
    """Return a true-label by predicted-label count matrix."""

    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for prediction in predictions:
        matrix[prediction.true_label, prediction.predicted_label] += 1
    return matrix


def summarize_distances(distances: Sequence[float]) -> dict[str, int | float | None]:
    """Calculate count, mean, extrema, and population standard deviation."""

    if not distances:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
        }

    values = np.asarray(distances, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
    }


def analyze_thresholds(predictions: Sequence[Prediction]) -> dict[str, object]:
    """Recommend a provisional threshold from known-identity distances."""

    correct = np.asarray(
        [prediction.distance_score for prediction in predictions if prediction.correct],
        dtype=np.float64,
    )
    incorrect = np.asarray(
        [
            prediction.distance_score
            for prediction in predictions
            if not prediction.correct
        ],
        dtype=np.float64,
    )
    if correct.size == 0:
        raise ValueError("Cannot recommend a threshold without correct predictions")

    correct_p90 = float(np.percentile(correct, 90))
    correct_p95 = float(np.percentile(correct, 95))
    candidates: dict[str, float | None] = {
        "correct_distance_p90": correct_p90,
        "correct_distance_p95": correct_p95,
        "correct_distance_maximum": float(correct.max()),
        "incorrect_distance_minimum": (
            float(incorrect.min()) if incorrect.size else None
        ),
        "incorrect_distance_p05": (
            float(np.percentile(incorrect, 5)) if incorrect.size else None
        ),
    }

    if incorrect.size:
        thresholds = np.unique(np.concatenate((correct, incorrect)))
        best_threshold = float(thresholds[0])
        best_balanced_accuracy = -1.0
        for threshold in thresholds:
            correct_acceptance = float(np.mean(correct <= threshold))
            incorrect_rejection = float(np.mean(incorrect > threshold))
            balanced_accuracy = (correct_acceptance + incorrect_rejection) / 2.0
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_threshold = float(threshold)

        recommended = best_threshold
        method = "known_identity_balanced_separation"
        reasoning = (
            "Selected the observed distance threshold that best balances accepting "
            "correct known-identity predictions and rejecting incorrect "
            "known-identity predictions."
        )
        candidates["balanced_separation_threshold"] = best_threshold
        candidates["balanced_separation_score"] = best_balanced_accuracy
    else:
        recommended = correct_p95
        method = "correct_distance_95th_percentile"
        reasoning = (
            "All known-identity predictions were correct, so the 95th percentile "
            "of correct distances is used as a conservative initial cutoff."
        )

    return {
        "recommended_initial_threshold": recommended,
        "method": method,
        "candidate_thresholds": candidates,
        "reasoning": reasoning,
        "limitations": (
            "This test set contains only enrolled AT&T identities. A reliable "
            "unknown-face rejection threshold cannot be fully validated without "
            "representative non-enrolled identities and operational camera data. "
            "Treat this value as an initial candidate, not a production cutoff."
        ),
    }


def build_evaluation_summary(
    predictions: Sequence[Prediction],
    label_map: dict[str, int],
) -> dict[str, object]:
    """Build the complete serializable evaluation summary."""

    if not predictions:
        raise ValueError("Cannot summarize an empty prediction collection")

    correct_count = sum(prediction.correct for prediction in predictions)
    incorrect_count = len(predictions) - correct_count
    per_class_accuracy: dict[str, dict[str, int | float]] = {}

    for subject, label in label_map.items():
        class_predictions = [
            prediction for prediction in predictions if prediction.true_label == label
        ]
        class_correct = sum(
            prediction.correct for prediction in class_predictions
        )
        sample_count = len(class_predictions)
        per_class_accuracy[subject] = {
            "label": label,
            "samples": sample_count,
            "correct": class_correct,
            "accuracy": class_correct / sample_count if sample_count else 0.0,
        }

    all_distances = [prediction.distance_score for prediction in predictions]
    correct_distances = [
        prediction.distance_score for prediction in predictions if prediction.correct
    ]
    incorrect_distances = [
        prediction.distance_score
        for prediction in predictions
        if not prediction.correct
    ]

    return {
        "test_samples": len(predictions),
        "class_count": len(label_map),
        "overall_accuracy": correct_count / len(predictions),
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "per_class_accuracy": per_class_accuracy,
        "distance_statistics": {
            "overall": summarize_distances(all_distances),
            "correct": summarize_distances(correct_distances),
            "incorrect": summarize_distances(incorrect_distances),
        },
        "threshold_analysis": analyze_thresholds(predictions),
    }


def save_predictions_csv(
    predictions: Sequence[Prediction],
    output_path: Path,
) -> None:
    """Write all prediction records to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Prediction.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(prediction) for prediction in predictions)


def save_summary_json(summary: dict[str, object], output_path: Path) -> None:
    """Write the evaluation summary as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def _ordered_subject_names(label_map: dict[str, int]) -> list[str]:
    """Return subject names ordered by numeric label."""

    return [
        subject
        for subject, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]


def save_confusion_matrix_plot(
    matrix: NDArray[np.int64],
    label_map: dict[str, int],
    output_path: Path,
) -> None:
    """Save a headless confusion-matrix plot with all subject labels."""

    subject_names = _ordered_subject_names(label_map)
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    ticks = np.arange(len(subject_names))
    axis.set(
        title="LBPH Confusion Matrix",
        xlabel="Predicted subject",
        ylabel="True subject",
        xticks=ticks,
        yticks=ticks,
        xticklabels=subject_names,
        yticklabels=subject_names,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_distance_distribution_plot(
    predictions: Sequence[Prediction],
    output_path: Path,
) -> None:
    """Save a headless histogram comparing correct and incorrect distances."""

    correct = [
        prediction.distance_score for prediction in predictions if prediction.correct
    ]
    incorrect = [
        prediction.distance_score
        for prediction in predictions
        if not prediction.correct
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    bins = max(5, min(20, int(np.sqrt(max(1, len(predictions))))))
    if correct:
        axis.hist(correct, bins=bins, alpha=0.7, label="Correct", color="tab:green")
    if incorrect:
        axis.hist(
            incorrect,
            bins=bins,
            alpha=0.7,
            label="Incorrect",
            color="tab:red",
        )
    axis.set(
        title="LBPH Distance Distribution",
        xlabel="LBPH distance (lower is more similar)",
        ylabel="Sample count",
    )
    if correct or incorrect:
        axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_misclassified_samples_plot(
    test_data: EvaluationData,
    predictions: Sequence[Prediction],
    output_path: Path,
) -> bool:
    """Save a grid of misclassified faces, returning whether a plot was made."""

    misclassified = [
        prediction for prediction in predictions if not prediction.correct
    ]
    if not misclassified:
        return False

    shown = misclassified[:20]
    columns = 5
    rows = int(np.ceil(len(shown) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 2.8 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, prediction in zip(axes.flat, shown, strict=False):
        axis.imshow(test_data.images[prediction.sample_index], cmap="gray")
        axis.set_title(
            f"{prediction.true_subject_name} → "
            f"{prediction.predicted_subject_name}\n"
            f"distance={prediction.distance_score:.2f}",
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(
        f"Misclassified Samples (showing {len(shown)} of {len(misclassified)})"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def save_evaluation_outputs(
    test_data: EvaluationData,
    predictions: Sequence[Prediction],
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Persist evaluation data, matrices, and headless plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = calculate_confusion_matrix(predictions)
    save_summary_json(summary, output_dir / "evaluation_summary.json")
    save_predictions_csv(predictions, output_dir / "predictions.csv")
    np.save(output_dir / "confusion_matrix.npy", matrix)
    save_confusion_matrix_plot(
        matrix,
        test_data.label_map,
        output_dir / "confusion_matrix.png",
    )
    save_distance_distribution_plot(
        predictions,
        output_dir / "distance_distribution.png",
    )
    misclassified_path = output_dir / "misclassified_samples.png"
    misclassified_plot_created = save_misclassified_samples_plot(
        test_data,
        predictions,
        misclassified_path,
    )
    if not misclassified_plot_created and misclassified_path.is_file():
        misclassified_path.unlink()


def print_evaluation_summary(
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Print the requested concise evaluation summary."""

    distance_statistics = summary["distance_statistics"]
    threshold_analysis = summary["threshold_analysis"]
    if not isinstance(distance_statistics, dict) or not isinstance(
        threshold_analysis, dict
    ):
        raise TypeError("Evaluation summary has an invalid structure")
    overall_distances = distance_statistics["overall"]
    if not isinstance(overall_distances, dict):
        raise TypeError("Overall distance statistics have an invalid structure")

    print("Model evaluation complete")
    print(f"Test samples: {summary['test_samples']}")
    print(f"Overall accuracy: {float(summary['overall_accuracy']):.2%}")
    print(f"Correct predictions: {summary['correct_predictions']}")
    print(f"Incorrect predictions: {summary['incorrect_predictions']}")
    print(f"Mean distance: {float(overall_distances['mean']):.4f}")
    print(
        "Recommended initial threshold: "
        f"{float(threshold_analysis['recommended_initial_threshold']):.4f}"
    )
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    """Load the test set/model, evaluate every image, and save outputs."""

    test_data = load_test_data(DEFAULT_PROCESSED_DIR)
    recognizer = load_recognizer(DEFAULT_MODEL_PATH)
    predictions = collect_predictions(recognizer, test_data)
    summary = build_evaluation_summary(predictions, test_data.label_map)
    save_evaluation_outputs(test_data, predictions, summary, DEFAULT_OUTPUT_DIR)
    print_evaluation_summary(summary, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
