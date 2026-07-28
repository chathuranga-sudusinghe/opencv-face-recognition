"""Train and persist an OpenCV LBPH face recognizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lbph_model.yml"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"

EXPECTED_IMAGE_SHAPE = (112, 92)
EXPECTED_CLASS_COUNT = 40
ALGORITHM_NAME = "OpenCV LBPHFaceRecognizer"

# Explicit OpenCV LBPH defaults. The maximum finite threshold disables
# rejection until an appropriate cutoff is selected during later evaluation.
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD = float(np.finfo(np.float64).max)


class LBPHRecognizer(Protocol):
    """Operations used from an OpenCV LBPH recognizer."""

    def train(self, src: list[NDArray[np.uint8]], labels: NDArray[np.int32]) -> None:
        """Train the recognizer."""

    def write(self, filename: str) -> None:
        """Write the trained recognizer to disk."""


@dataclass(frozen=True)
class TrainingData:
    """Validated inputs required for LBPH training."""

    images: NDArray[np.uint8]
    labels: NDArray[np.integer[Any]]
    label_map: dict[str, int]


def _load_label_map(label_map_path: Path) -> dict[str, int]:
    """Load a subject-to-integer label map from JSON."""

    try:
        raw_mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in label map: {label_map_path}") from exc

    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Label map must be a JSON object: {label_map_path}")
    if not all(
        isinstance(subject, str)
        and isinstance(label, int)
        and not isinstance(label, bool)
        for subject, label in raw_mapping.items()
    ):
        raise ValueError("Label map must contain string keys and integer values")

    return raw_mapping


def validate_training_data(
    images: NDArray[Any],
    labels: NDArray[Any],
    label_map: dict[str, int],
) -> None:
    """Validate training array structure, types, sample counts, and labels."""

    if images.ndim != 3 or images.shape[1:] != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"X_train must have shape (N, {EXPECTED_IMAGE_SHAPE[0]}, "
            f"{EXPECTED_IMAGE_SHAPE[1]}), got {images.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"y_train must have shape (N,), got {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "X_train and y_train sample counts differ: "
            f"{images.shape[0]} != {labels.shape[0]}"
        )
    if images.dtype != np.uint8:
        raise TypeError(f"X_train must have dtype uint8, got {images.dtype}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"y_train must contain integer labels, got {labels.dtype}")

    mapped_labels = set(label_map.values())
    expected_labels = set(range(EXPECTED_CLASS_COUNT))
    if len(label_map) != EXPECTED_CLASS_COUNT or mapped_labels != expected_labels:
        raise ValueError(
            "label_map.json must define exactly 40 unique labels from 0 to 39"
        )

    training_labels = {int(label) for label in np.unique(labels)}
    if training_labels != mapped_labels:
        raise ValueError(
            "Training label values are inconsistent with label_map.json: "
            f"training={sorted(training_labels)}, mapping={sorted(mapped_labels)}"
        )


def load_training_data(processed_dir: Path) -> TrainingData:
    """Load and validate the prepared training arrays and label map."""

    images_path = processed_dir / "X_train.npy"
    labels_path = processed_dir / "y_train.npy"
    label_map_path = processed_dir / "label_map.json"
    required_paths = (images_path, labels_path, label_map_path)

    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required training file(s): {missing}")

    images = np.load(images_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    label_map = _load_label_map(label_map_path)
    validate_training_data(images, labels, label_map)

    return TrainingData(images=images, labels=labels, label_map=label_map)


def create_recognizer() -> LBPHRecognizer:
    """Create an LBPH recognizer with the project's explicit parameters."""

    face_module = getattr(cv2, "face", None)
    factory = getattr(face_module, "LBPHFaceRecognizer_create", None)
    if factory is None:
        raise RuntimeError(
            "OpenCV LBPH support is unavailable; install opencv-contrib-python"
        )

    return factory(
        LBPH_RADIUS,
        LBPH_NEIGHBORS,
        LBPH_GRID_X,
        LBPH_GRID_Y,
        LBPH_THRESHOLD,
    )


def train_and_save_model(
    recognizer: LBPHRecognizer,
    training_data: TrainingData,
    model_path: Path,
) -> None:
    """Train an LBPH recognizer and write its model file."""

    labels = training_data.labels.astype(np.int32, copy=False)
    recognizer.train(list(training_data.images), labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(model_path))
    if not model_path.is_file():
        raise RuntimeError(f"OpenCV did not create the model file: {model_path}")


def create_training_metadata(
    training_data: TrainingData,
    model_path: Path,
    *,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Build serializable metadata describing one LBPH training run."""

    training_time = timestamp or datetime.now(timezone.utc)
    if training_time.tzinfo is None:
        raise ValueError("Training timestamp must include timezone information")
    utc_timestamp = training_time.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    return {
        "algorithm": ALGORITHM_NAME,
        "training_samples": int(training_data.images.shape[0]),
        "class_count": len(training_data.label_map),
        "image_dimensions": list(training_data.images.shape[1:]),
        "label_range": {
            "minimum": int(training_data.labels.min()),
            "maximum": int(training_data.labels.max()),
        },
        "model_path": str(model_path.resolve()),
        "lbph_parameters": {
            "radius": LBPH_RADIUS,
            "neighbors": LBPH_NEIGHBORS,
            "grid_x": LBPH_GRID_X,
            "grid_y": LBPH_GRID_Y,
            "threshold": LBPH_THRESHOLD,
        },
        "training_timestamp_utc": utc_timestamp,
    }


def save_training_metadata(metadata: dict[str, object], metadata_path: Path) -> None:
    """Write training metadata as formatted JSON."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_training_summary(
    training_data: TrainingData,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Print a concise summary of the completed training run."""

    print("Model training complete")
    print(f"Algorithm: {ALGORITHM_NAME}")
    print(f"Training samples: {training_data.images.shape[0]}")
    print(f"Class count: {len(training_data.label_map)}")
    print(f"Image dimensions: {training_data.images.shape[1:]}")
    print(f"Model output: {model_path.resolve()}")
    print(f"Metadata output: {metadata_path.resolve()}")


def main() -> None:
    """Load prepared data, train LBPH, and persist the model and metadata."""

    training_data = load_training_data(DEFAULT_PROCESSED_DIR)
    recognizer = create_recognizer()
    train_and_save_model(recognizer, training_data, DEFAULT_MODEL_PATH)
    metadata = create_training_metadata(training_data, DEFAULT_MODEL_PATH)
    save_training_metadata(metadata, DEFAULT_METADATA_PATH)
    print_training_summary(
        training_data,
        DEFAULT_MODEL_PATH,
        DEFAULT_METADATA_PATH,
    )


if __name__ == "__main__":
    main()
