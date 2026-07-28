"""Validate the raw face dataset and create deterministic NumPy splits."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUBJECT_COUNT = 40
IMAGES_PER_SUBJECT = 10
TRAIN_IMAGE_NUMBERS = range(1, 9)
TEST_IMAGE_NUMBERS = range(9, 11)
EXPECTED_IMAGE_SHAPE = (112, 92)

ImageArray = NDArray[np.uint8]
LabelArray = NDArray[np.int64]
PreparedData = tuple[ImageArray, LabelArray, ImageArray, LabelArray, dict[str, int]]


def subject_names() -> list[str]:
    """Return all expected subject names in numeric order."""

    return [f"s{number}" for number in range(1, SUBJECT_COUNT + 1)]


def validate_subject_folders(raw_dir: Path) -> None:
    """Raise a clear error when the raw root or an expected subject is missing."""

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    missing = [name for name in subject_names() if not (raw_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing subject folder(s) in {raw_dir}: {', '.join(missing)}"
        )


def load_image(image_path: Path) -> ImageArray:
    """Load and validate one grayscale face image."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    if image.ndim != 2:
        raise ValueError(
            f"Image must be grayscale (2 dimensions), got shape {image.shape}: "
            f"{image_path}"
        )
    if image.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"Image must have shape {EXPECTED_IMAGE_SHAPE}, got {image.shape}: "
            f"{image_path}"
        )

    return image


def prepare_dataset(raw_dir: Path) -> PreparedData:
    """Load, validate, label, and split the complete raw dataset."""

    validate_subject_folders(raw_dir)

    train_images: list[ImageArray] = []
    train_labels: list[int] = []
    test_images: list[ImageArray] = []
    test_labels: list[int] = []
    label_map: dict[str, int] = {}

    for label, subject_name in enumerate(subject_names()):
        subject_dir = raw_dir / subject_name
        label_map[subject_name] = label

        for image_number in TRAIN_IMAGE_NUMBERS:
            train_images.append(load_image(subject_dir / f"{image_number}.pgm"))
            train_labels.append(label)

        for image_number in TEST_IMAGE_NUMBERS:
            test_images.append(load_image(subject_dir / f"{image_number}.pgm"))
            test_labels.append(label)

    return (
        np.stack(train_images),
        np.asarray(train_labels, dtype=np.int64),
        np.stack(test_images),
        np.asarray(test_labels, dtype=np.int64),
        label_map,
    )


def save_prepared_data(data: PreparedData, output_dir: Path) -> None:
    """Save prepared arrays and the subject-to-label mapping."""

    X_train, y_train, X_test, y_test, label_map = data
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(data: PreparedData, output_dir: Path) -> None:
    """Print a concise summary of the prepared dataset."""

    X_train, y_train, X_test, y_test, label_map = data
    all_labels = np.concatenate((y_train, y_test))

    print("Dataset preparation complete")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Class count: {len(label_map)}")
    print(f"Image dimensions: {X_train.shape[1:]}")
    print(f"Label range: {int(all_labels.min())}-{int(all_labels.max())}")
    print(f"Output directory: {output_dir.resolve()}")


def main() -> None:
    """Prepare the repository's raw dataset and save generated artifacts."""

    data = prepare_dataset(DEFAULT_RAW_DIR)
    save_prepared_data(data, DEFAULT_OUTPUT_DIR)
    print_summary(data, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
