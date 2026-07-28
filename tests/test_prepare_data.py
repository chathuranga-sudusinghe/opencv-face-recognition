"""Tests for deterministic face dataset preparation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.prepare_data import EXPECTED_IMAGE_SHAPE, prepare_dataset


def create_raw_dataset(raw_dir: Path) -> None:
    """Create a complete synthetic dataset with identifiable pixel values."""

    for subject_number in range(1, 41):
        subject_dir = raw_dir / f"s{subject_number}"
        subject_dir.mkdir(parents=True)
        for image_number in range(1, 11):
            pixel_value = subject_number + image_number
            image = np.full(EXPECTED_IMAGE_SHAPE, pixel_value, dtype=np.uint8)
            assert cv2.imwrite(str(subject_dir / f"{image_number}.pgm"), image)


@pytest.fixture()
def raw_dataset(tmp_path: Path) -> Path:
    """Return a valid synthetic raw dataset."""

    raw_dir = tmp_path / "raw"
    create_raw_dataset(raw_dir)
    return raw_dir


def test_prepare_dataset_shapes_and_sample_counts(raw_dataset: Path) -> None:
    X_train, y_train, X_test, y_test, label_map = prepare_dataset(raw_dataset)

    assert X_train.shape == (320, 112, 92)
    assert y_train.shape == (320,)
    assert X_test.shape == (80, 112, 92)
    assert y_test.shape == (80,)
    assert len(label_map) == 40


def test_prepare_dataset_has_deterministic_labels(raw_dataset: Path) -> None:
    _, y_train, _, y_test, label_map = prepare_dataset(raw_dataset)

    assert label_map == {f"s{number}": number - 1 for number in range(1, 41)}
    assert np.array_equal(y_train, np.repeat(np.arange(40), 8))
    assert np.array_equal(y_test, np.repeat(np.arange(40), 2))


def test_prepared_images_have_correct_grayscale_dimensions(
    raw_dataset: Path,
) -> None:
    X_train, _, X_test, _, _ = prepare_dataset(raw_dataset)

    assert X_train.ndim == 3
    assert X_test.ndim == 3
    assert X_train.shape[1:] == EXPECTED_IMAGE_SHAPE
    assert X_test.shape[1:] == EXPECTED_IMAGE_SHAPE


def test_missing_subject_folder_raises_clear_error(raw_dataset: Path) -> None:
    missing_subject = raw_dataset / "s7"
    for image_path in missing_subject.iterdir():
        image_path.unlink()
    missing_subject.rmdir()

    with pytest.raises(FileNotFoundError, match=r"Missing subject folder.*s7"):
        prepare_dataset(raw_dataset)


def test_missing_image_raises_clear_error(raw_dataset: Path) -> None:
    (raw_dataset / "s12" / "4.pgm").unlink()

    with pytest.raises(FileNotFoundError, match=r"Missing image:.*s12.*4\.pgm"):
        prepare_dataset(raw_dataset)


def test_unreadable_image_raises_clear_error(raw_dataset: Path) -> None:
    (raw_dataset / "s20" / "9.pgm").write_bytes(b"not a valid image")

    with pytest.raises(ValueError, match=r"OpenCV could not read image:.*s20.*9\.pgm"):
        prepare_dataset(raw_dataset)
