"""Train a separate LBPH model from combined original and registered data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from src.train_model import (
        ALGORITHM_NAME,
        LBPH_GRID_X,
        LBPH_GRID_Y,
        LBPH_NEIGHBORS,
        LBPH_RADIUS,
        LBPH_THRESHOLD,
        TrainingData,
        create_recognizer,
        train_and_save_model,
    )
except ModuleNotFoundError:  # Support direct execution as src/<script>.py.
    from train_model import (
        ALGORITHM_NAME,
        LBPH_GRID_X,
        LBPH_GRID_Y,
        LBPH_NEIGHBORS,
        LBPH_RADIUS,
        LBPH_THRESHOLD,
        TrainingData,
        create_recognizer,
        train_and_save_model,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMBINED_DIR = PROJECT_ROOT / "data" / "combined"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lbph_combined_model.yml"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "combined_training_metadata.json"

EXPECTED_IMAGE_SHAPE = (112, 92)
CombinedTrainingData = tuple[TrainingData, list[dict[str, object]], int, int]


def _load_json(path: Path, description: str) -> Any:
    """Load JSON with a context-rich validation error."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {description} JSON: {path}") from exc


def load_combined_training_data(combined_dir: Path) -> CombinedTrainingData:
    """Load and validate combined arrays and identity metadata."""

    images_path = combined_dir / "X_train_combined.npy"
    labels_path = combined_dir / "y_train_combined.npy"
    label_map_path = combined_dir / "label_map_combined.json"
    registered_path = combined_dir / "registered_subjects.json"
    required_paths = (images_path, labels_path, label_map_path, registered_path)
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing combined training file(s): {missing}")

    images = np.load(images_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    label_map = _load_json(label_map_path, "combined label map")
    registered_records = _load_json(registered_path, "registered subjects")

    if images.ndim != 3 or images.shape[1:] != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Combined images must have shape (N, 112, 92), got {images.shape}"
        )
    if labels.ndim != 1:
        raise ValueError(f"Combined labels must have shape (N,), got {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            "Combined image and label sample counts differ: "
            f"{images.shape[0]} != {labels.shape[0]}"
        )
    if images.dtype != np.uint8:
        raise TypeError(f"Combined images must have dtype uint8, got {images.dtype}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Combined labels must be integers, got {labels.dtype}")
    if not isinstance(label_map, dict) or not all(
        isinstance(subject, str)
        and isinstance(label, int)
        and not isinstance(label, bool)
        for subject, label in label_map.items()
    ):
        raise ValueError("Combined label map must contain string keys and integer values")

    expected_labels = set(range(len(label_map)))
    mapped_labels = set(label_map.values())
    actual_labels = {int(label) for label in np.unique(labels)}
    if mapped_labels != expected_labels or actual_labels != expected_labels:
        raise ValueError(
            "Combined labels and label map must contain contiguous labels from zero"
        )
    if not isinstance(registered_records, list) or not all(
        isinstance(record, dict) for record in registered_records
    ):
        raise ValueError("registered_subjects.json must contain a list of objects")

    registered_sample_count = 0
    for record in registered_records:
        subject_key = record.get("subject_key")
        assigned_label = record.get("assigned_label")
        training_sample_count = record.get("training_sample_count")
        if (
            not isinstance(subject_key, str)
            or not isinstance(assigned_label, int)
            or isinstance(assigned_label, bool)
            or not isinstance(training_sample_count, int)
            or isinstance(training_sample_count, bool)
            or training_sample_count < 1
            or label_map.get(subject_key) != assigned_label
        ):
            raise ValueError(f"Invalid registered subject record: {record}")
        registered_sample_count += training_sample_count

    labels_from_registered_records = {
        int(record["assigned_label"]) for record in registered_records
    }
    actual_registered_count = int(
        np.isin(labels, list(labels_from_registered_records)).sum()
    )
    if actual_registered_count != registered_sample_count:
        raise ValueError(
            "Registered sample counts do not match the combined label array"
        )

    original_sample_count = int(images.shape[0]) - registered_sample_count
    if original_sample_count < 1:
        raise ValueError("Combined data contains no original training samples")

    training_data = TrainingData(
        images=images,
        labels=labels,
        label_map=label_map,
    )
    return (
        training_data,
        registered_records,
        original_sample_count,
        registered_sample_count,
    )


def create_combined_training_metadata(
    training_data: TrainingData,
    registered_records: list[dict[str, object]],
    original_sample_count: int,
    registered_sample_count: int,
    model_path: Path,
    *,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Build metadata for one combined LBPH training run."""

    training_time = timestamp or datetime.now(timezone.utc)
    if training_time.tzinfo is None:
        raise ValueError("Training timestamp must include timezone information")
    timestamp_utc = training_time.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    return {
        "algorithm": ALGORITHM_NAME,
        "total_training_samples": int(training_data.images.shape[0]),
        "original_sample_count": original_sample_count,
        "registered_sample_count": registered_sample_count,
        "total_class_count": len(training_data.label_map),
        "image_dimensions": list(training_data.images.shape[1:]),
        "label_range": {
            "minimum": int(training_data.labels.min()),
            "maximum": int(training_data.labels.max()),
        },
        "registered_identities": registered_records,
        "lbph_parameters": {
            "radius": LBPH_RADIUS,
            "neighbors": LBPH_NEIGHBORS,
            "grid_x": LBPH_GRID_X,
            "grid_y": LBPH_GRID_Y,
            "threshold": LBPH_THRESHOLD,
        },
        "model_path": str(model_path.resolve()),
        "training_timestamp_utc": timestamp_utc,
    }


def save_combined_training_metadata(
    metadata: dict[str, object],
    metadata_path: Path,
) -> None:
    """Save combined training metadata as formatted JSON."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def print_training_summary(
    training_data: TrainingData,
    original_sample_count: int,
    registered_sample_count: int,
    model_path: Path,
    metadata_path: Path,
) -> None:
    """Print a concise combined-model training summary."""

    print("Combined LBPH model training complete")
    print(f"Algorithm: {ALGORITHM_NAME}")
    print(f"Original samples: {original_sample_count}")
    print(f"Registered samples: {registered_sample_count}")
    print(f"Total samples: {training_data.images.shape[0]}")
    print(f"Total classes: {len(training_data.label_map)}")
    print(f"Image dimensions: {training_data.images.shape[1:]}")
    print(f"Model output: {model_path.resolve()}")
    print(f"Metadata output: {metadata_path.resolve()}")


def main() -> None:
    """Train and save the separate combined LBPH model and metadata."""

    (
        training_data,
        registered_records,
        original_sample_count,
        registered_sample_count,
    ) = load_combined_training_data(DEFAULT_COMBINED_DIR)
    recognizer = create_recognizer()
    train_and_save_model(recognizer, training_data, DEFAULT_MODEL_PATH)
    metadata = create_combined_training_metadata(
        training_data,
        registered_records,
        original_sample_count,
        registered_sample_count,
        DEFAULT_MODEL_PATH,
    )
    save_combined_training_metadata(metadata, DEFAULT_METADATA_PATH)
    print_training_summary(
        training_data,
        original_sample_count,
        registered_sample_count,
        DEFAULT_MODEL_PATH,
        DEFAULT_METADATA_PATH,
    )


if __name__ == "__main__":
    main()
