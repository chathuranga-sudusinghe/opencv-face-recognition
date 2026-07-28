"""Combine original training data with registered local face images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    from src.train_model import TrainingData, load_training_data
except ModuleNotFoundError:  # Support direct execution as src/<script>.py.
    from train_model import TrainingData, load_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REGISTERED_DIR = PROJECT_ROOT / "data" / "registered_faces"
DEFAULT_COMBINED_DIR = PROJECT_ROOT / "data" / "combined"

EXPECTED_IMAGE_SHAPE = (112, 92)
MINIMUM_REGISTERED_SAMPLES = 10
MINIMUM_VALIDATION_SAMPLES = 2
REGISTERED_TRAIN_FRACTION = 0.8

CombinedData = tuple[
    NDArray[np.uint8],
    NDArray[np.integer[Any]],
    dict[str, int],
    list[dict[str, object]],
    NDArray[np.uint8],
    NDArray[np.integer[Any]],
    list[dict[str, object]],
]


@dataclass(frozen=True)
class RegisteredSubject:
    """Validated images and identity details for one registered subject."""

    subject_key: str
    display_name: str
    training_images: NDArray[np.uint8]
    validation_images: NDArray[np.uint8]
    training_image_paths: tuple[Path, ...]
    validation_image_paths: tuple[Path, ...]
    source_folder: Path


def _numeric_png_paths(subject_dir: Path) -> list[Path]:
    """Return PNG files in numeric stem order."""

    image_paths = [
        path
        for path in subject_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    invalid_names = [path.name for path in image_paths if not path.stem.isdigit()]
    if invalid_names:
        raise ValueError(
            f"Registered PNG filenames must be numeric in {subject_dir}: "
            f"{', '.join(sorted(invalid_names))}"
        )

    return sorted(image_paths, key=lambda path: (int(path.stem), path.name))


def _load_display_name(metadata_path: Path, subject_key: str) -> str:
    """Load and validate the registered subject's display name."""

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing registration metadata for {subject_key}: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid registration metadata JSON: {metadata_path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Registration metadata must be a JSON object: {metadata_path}")
    display_name = metadata.get("original_name")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(
            f"Registration metadata has no valid original_name: {metadata_path}"
        )
    metadata_key = metadata.get("safe_folder_name")
    if metadata_key is not None and metadata_key != subject_key:
        raise ValueError(
            f"Registration metadata safe_folder_name does not match folder "
            f"{subject_key}: {metadata_path}"
        )
    return display_name.strip()


def load_registered_subject(subject_dir: Path) -> RegisteredSubject:
    """Load and validate one registered subject folder."""

    subject_key = subject_dir.name
    display_name = _load_display_name(subject_dir / "metadata.json", subject_key)
    image_paths = _numeric_png_paths(subject_dir)
    if len(image_paths) < MINIMUM_REGISTERED_SAMPLES:
        raise ValueError(
            f"Registered subject {subject_key} requires at least "
            f"{MINIMUM_REGISTERED_SAMPLES} PNG images, found {len(image_paths)}"
        )

    images: list[NDArray[np.uint8]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV could not read registered image: {image_path}")
        if image.ndim != 2 or image.shape != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Registered image must be grayscale with shape "
                f"{EXPECTED_IMAGE_SHAPE}, got {image.shape}: {image_path}"
            )
        if image.dtype != np.uint8:
            raise TypeError(
                f"Registered image must have dtype uint8, got {image.dtype}: "
                f"{image_path}"
            )
        images.append(image)

    split_index = int(len(images) * REGISTERED_TRAIN_FRACTION)
    validation_count = len(images) - split_index
    if validation_count < MINIMUM_VALIDATION_SAMPLES:
        raise ValueError(
            f"Registered subject {subject_key} requires at least "
            f"{MINIMUM_VALIDATION_SAMPLES} validation images after the 80/20 "
            f"split, found {validation_count}"
        )

    return RegisteredSubject(
        subject_key=subject_key,
        display_name=display_name,
        training_images=np.stack(images[:split_index]),
        validation_images=np.stack(images[split_index:]),
        training_image_paths=tuple(image_paths[:split_index]),
        validation_image_paths=tuple(image_paths[split_index:]),
        source_folder=subject_dir,
    )


def load_registered_subjects(registered_dir: Path) -> list[RegisteredSubject]:
    """Load registered subject folders in deterministic alphabetical order."""

    if not registered_dir.is_dir():
        raise FileNotFoundError(
            f"Registered faces directory does not exist: {registered_dir}"
        )
    subject_dirs = sorted(
        (path for path in registered_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not subject_dirs:
        raise ValueError(f"No registered subject folders found in {registered_dir}")

    return [load_registered_subject(subject_dir) for subject_dir in subject_dirs]


def combine_training_data(
    original_data: TrainingData,
    registered_subjects: list[RegisteredSubject],
) -> CombinedData:
    """Append registered images while preserving every original label."""

    combined_label_map = dict(original_data.label_map)
    combined_images = [original_data.images]
    combined_labels: list[NDArray[np.integer[Any]]] = [original_data.labels]
    registered_records: list[dict[str, object]] = []
    validation_images: list[NDArray[np.uint8]] = []
    validation_labels: list[NDArray[np.int64]] = []
    validation_manifest: list[dict[str, object]] = []
    next_label = max(original_data.label_map.values()) + 1

    for offset, subject in enumerate(registered_subjects):
        if subject.subject_key in combined_label_map:
            raise ValueError(
                f"Registered subject key conflicts with original label map: "
                f"{subject.subject_key}"
            )
        assigned_label = next_label + offset
        combined_label_map[subject.subject_key] = assigned_label
        combined_images.append(subject.training_images)
        combined_labels.append(
            np.full(
                subject.training_images.shape[0],
                assigned_label,
                dtype=np.int64,
            )
        )
        validation_images.append(subject.validation_images)
        validation_labels.append(
            np.full(
                subject.validation_images.shape[0],
                assigned_label,
                dtype=np.int64,
            )
        )
        registered_records.append(
            {
                "subject_key": subject.subject_key,
                "display_name": subject.display_name,
                "assigned_label": assigned_label,
                "sample_count": int(
                    subject.training_images.shape[0]
                    + subject.validation_images.shape[0]
                ),
                "training_sample_count": int(subject.training_images.shape[0]),
                "validation_sample_count": int(
                    subject.validation_images.shape[0]
                ),
                "source_folder": str(subject.source_folder.resolve()),
            }
        )
        for image_path in subject.validation_image_paths:
            validation_manifest.append(
                {
                    "validation_index": len(validation_manifest),
                    "subject_key": subject.subject_key,
                    "display_name": subject.display_name,
                    "assigned_label": assigned_label,
                    "filename": image_path.name,
                    "source_path": str(image_path.resolve()),
                }
            )

    return (
        np.concatenate(combined_images),
        np.concatenate(combined_labels),
        combined_label_map,
        registered_records,
        np.concatenate(validation_images),
        np.concatenate(validation_labels),
        validation_manifest,
    )


def prepare_combined_training_data(
    processed_dir: Path,
    registered_dir: Path,
) -> CombinedData:
    """Load original and registered data and return combined artifacts."""

    original_data = load_training_data(processed_dir)
    registered_subjects = load_registered_subjects(registered_dir)
    return combine_training_data(original_data, registered_subjects)


def save_combined_artifacts(data: CombinedData, combined_dir: Path) -> None:
    """Save combined arrays, label map, and registered-subject records."""

    (
        images,
        labels,
        label_map,
        registered_records,
        validation_images,
        validation_labels,
        validation_manifest,
    ) = data
    combined_dir.mkdir(parents=True, exist_ok=True)
    np.save(combined_dir / "X_train_combined.npy", images)
    np.save(combined_dir / "y_train_combined.npy", labels)
    (combined_dir / "label_map_combined.json").write_text(
        json.dumps(label_map, indent=2) + "\n",
        encoding="utf-8",
    )
    (combined_dir / "registered_subjects.json").write_text(
        json.dumps(registered_records, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(
        combined_dir / "X_registered_validation.npy",
        validation_images,
    )
    np.save(
        combined_dir / "y_registered_validation.npy",
        validation_labels,
    )
    (combined_dir / "registered_validation_manifest.json").write_text(
        json.dumps(validation_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def print_preparation_summary(
    data: CombinedData,
    original_sample_count: int,
    output_dir: Path,
) -> None:
    """Print a concise combined-dataset preparation summary."""

    images, _, label_map, registered_records, validation_images, _, _ = data
    registered_sample_count = sum(
        int(record["training_sample_count"]) for record in registered_records
    )
    print("Combined training data preparation complete")
    print(f"Original samples: {original_sample_count}")
    print(f"Registered training samples: {registered_sample_count}")
    print(f"Combined training samples: {images.shape[0]}")
    print(f"Registered validation samples: {validation_images.shape[0]}")
    print(f"Total classes: {len(label_map)}")
    print(f"Image dimensions: {images.shape[1:]}")
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    """Prepare and save original-plus-registered training data."""

    original_data = load_training_data(DEFAULT_PROCESSED_DIR)
    registered_subjects = load_registered_subjects(DEFAULT_REGISTERED_DIR)
    combined_data = combine_training_data(original_data, registered_subjects)
    save_combined_artifacts(combined_data, DEFAULT_COMBINED_DIR)
    print_preparation_summary(
        combined_data,
        original_data.images.shape[0],
        DEFAULT_COMBINED_DIR,
    )


if __name__ == "__main__":
    main()
