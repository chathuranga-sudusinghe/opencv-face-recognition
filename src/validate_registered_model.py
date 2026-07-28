"""Validate the combined LBPH model on held-out registered face images."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from src.evaluate_model import load_recognizer
except ModuleNotFoundError:  # Support direct execution as src/<script>.py.
    from evaluate_model import load_recognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED_DIR = PROJECT_ROOT / "data" / "combined"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lbph_combined_model.yml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "registered_validation"
EXPECTED_IMAGE_SHAPE = (112, 92)


class Recognizer(Protocol):
    """Prediction operation used from a loaded LBPH recognizer."""

    def predict(self, image: NDArray[np.uint8]) -> tuple[int, float]:
        """Predict a label and distance for one validation image."""


@dataclass(frozen=True)
class RegisteredValidationData:
    """Validated held-out images, labels, and source manifest."""

    images: NDArray[np.uint8]
    labels: NDArray[np.integer[Any]]
    label_map: dict[str, int]
    manifest: list[dict[str, object]]


@dataclass(frozen=True)
class RegisteredPrediction:
    """One prediction for a held-out registered image."""

    validation_index: int
    filename: str
    subject_key: str
    display_name: str
    true_label: int
    predicted_label: int
    predicted_subject_key: str
    distance_score: float
    correct: bool


def load_registered_validation_data(
    combined_dir: Path,
) -> RegisteredValidationData:
    """Load and validate registered validation arrays and their manifest."""

    images_path = combined_dir / "X_registered_validation.npy"
    labels_path = combined_dir / "y_registered_validation.npy"
    label_map_path = combined_dir / "label_map_combined.json"
    manifest_path = combined_dir / "registered_validation_manifest.json"
    required_paths = (images_path, labels_path, label_map_path, manifest_path)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing registered validation file(s): {missing}")

    images = np.load(images_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    try:
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid registered validation JSON artifact") from exc

    if images.ndim != 3 or images.shape[1:] != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Registered validation images must have shape (N, 112, 92), "
            f"got {images.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"Registered validation labels must have shape (N,), got {labels.shape}"
        )
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "Registered validation image and label sample counts differ"
        )
    if images.dtype != np.uint8:
        raise TypeError(
            f"Registered validation images must have dtype uint8, got {images.dtype}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Registered validation labels must be integers")
    if not isinstance(label_map, dict) or not all(
        isinstance(subject, str)
        and isinstance(label, int)
        and not isinstance(label, bool)
        for subject, label in label_map.items()
    ):
        raise ValueError("Combined label map is invalid")
    if not isinstance(manifest, list) or len(manifest) != images.shape[0]:
        raise ValueError(
            "Registered validation manifest length must match validation samples"
        )

    mapped_labels = set(label_map.values())
    for index, (label_value, record) in enumerate(
        zip(labels, manifest, strict=True)
    ):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid validation manifest record at index {index}")
        subject_key = record.get("subject_key")
        assigned_label = record.get("assigned_label")
        if (
            record.get("validation_index") != index
            or not isinstance(subject_key, str)
            or assigned_label != int(label_value)
            or label_map.get(subject_key) != assigned_label
        ):
            raise ValueError(f"Invalid validation manifest record at index {index}")
        if int(label_value) not in mapped_labels:
            raise ValueError(f"Validation label is absent from label map: {label_value}")

    return RegisteredValidationData(
        images=images,
        labels=labels,
        label_map=label_map,
        manifest=manifest,
    )


def collect_registered_predictions(
    recognizer: Recognizer,
    validation_data: RegisteredValidationData,
) -> list[RegisteredPrediction]:
    """Predict every held-out registered image."""

    label_to_subject = {
        label: subject for subject, label in validation_data.label_map.items()
    }
    predictions: list[RegisteredPrediction] = []
    for index, (image, true_label_value, record) in enumerate(
        zip(
            validation_data.images,
            validation_data.labels,
            validation_data.manifest,
            strict=True,
        )
    ):
        predicted_label_value, distance_value = recognizer.predict(image)
        true_label = int(true_label_value)
        predicted_label = int(predicted_label_value)
        predicted_subject = label_to_subject.get(
            predicted_label,
            f"unmapped_label_{predicted_label}",
        )
        predictions.append(
            RegisteredPrediction(
                validation_index=index,
                filename=str(record["filename"]),
                subject_key=str(record["subject_key"]),
                display_name=str(record["display_name"]),
                true_label=true_label,
                predicted_label=predicted_label,
                predicted_subject_key=predicted_subject,
                distance_score=float(distance_value),
                correct=predicted_label == true_label,
            )
        )
    return predictions


def _distance_summary(distances: list[float]) -> dict[str, int | float]:
    """Return count, mean, minimum, and maximum distance."""

    if not distances:
        raise ValueError("Cannot summarize an empty distance collection")
    values = np.asarray(distances, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def build_validation_summary(
    predictions: list[RegisteredPrediction],
) -> dict[str, object]:
    """Build local held-out accuracy and per-subject distance summaries."""

    if not predictions:
        raise ValueError("Cannot summarize an empty registered validation set")
    correct_count = sum(prediction.correct for prediction in predictions)
    subject_keys = sorted({prediction.subject_key for prediction in predictions})
    subject_summaries: dict[str, dict[str, object]] = {}
    for subject_key in subject_keys:
        subject_predictions = [
            prediction
            for prediction in predictions
            if prediction.subject_key == subject_key
        ]
        subject_correct = sum(
            prediction.correct for prediction in subject_predictions
        )
        subject_summaries[subject_key] = {
            "display_name": subject_predictions[0].display_name,
            "true_label": subject_predictions[0].true_label,
            "sample_count": len(subject_predictions),
            "correct_predictions": subject_correct,
            "accuracy": subject_correct / len(subject_predictions),
            "predicted_labels": [
                prediction.predicted_label for prediction in subject_predictions
            ],
            "distance_statistics": _distance_summary(
                [
                    prediction.distance_score
                    for prediction in subject_predictions
                ]
            ),
        }

    return {
        "validation_samples": len(predictions),
        "correct_predictions": correct_count,
        "incorrect_predictions": len(predictions) - correct_count,
        "accuracy": correct_count / len(predictions),
        "predicted_labels": [
            prediction.predicted_label for prediction in predictions
        ],
        "distance_statistics": _distance_summary(
            [prediction.distance_score for prediction in predictions]
        ),
        "subjects": subject_summaries,
    }


def save_validation_outputs(
    predictions: list[RegisteredPrediction],
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Save registered validation summary JSON and prediction CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(RegisteredPrediction.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(prediction) for prediction in predictions)


def print_validation_summary(
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Print concise held-out registered validation results."""

    overall_distances = summary["distance_statistics"]
    if not isinstance(overall_distances, dict):
        raise TypeError("Validation distance statistics are invalid")
    print("Registered-face validation complete")
    print(f"Validation samples: {summary['validation_samples']}")
    print(f"Accuracy: {float(summary['accuracy']):.2%}")
    print(f"Predicted labels: {summary['predicted_labels']}")
    print(f"Mean distance: {float(overall_distances['mean']):.4f}")
    print(f"Minimum distance: {float(overall_distances['minimum']):.4f}")
    print(f"Maximum distance: {float(overall_distances['maximum']):.4f}")
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    """Load the combined model and evaluate held-out registered images."""

    validation_data = load_registered_validation_data(DEFAULT_COMBINED_DIR)
    recognizer = load_recognizer(DEFAULT_MODEL_PATH)
    predictions = collect_registered_predictions(recognizer, validation_data)
    summary = build_validation_summary(predictions)
    save_validation_outputs(predictions, summary, DEFAULT_OUTPUT_DIR)
    print_validation_summary(summary, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
