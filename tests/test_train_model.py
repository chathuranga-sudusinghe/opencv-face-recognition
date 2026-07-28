"""Tests for loading training data and persisting an LBPH model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.train_model import (
    ALGORITHM_NAME,
    EXPECTED_IMAGE_SHAPE,
    LBPH_GRID_X,
    LBPH_GRID_Y,
    LBPH_NEIGHBORS,
    LBPH_RADIUS,
    LBPH_THRESHOLD,
    create_recognizer,
    create_training_metadata,
    load_training_data,
    save_training_metadata,
    train_and_save_model,
)


def write_valid_training_data(processed_dir: Path) -> None:
    """Write a small valid dataset containing all 40 required classes."""

    processed_dir.mkdir(parents=True)
    images = np.zeros((40, *EXPECTED_IMAGE_SHAPE), dtype=np.uint8)
    labels = np.arange(40, dtype=np.int64)
    label_map = {f"s{number}": number - 1 for number in range(1, 41)}

    np.save(processed_dir / "X_train.npy", images)
    np.save(processed_dir / "y_train.npy", labels)
    (processed_dir / "label_map.json").write_text(
        json.dumps(label_map),
        encoding="utf-8",
    )


@pytest.fixture()
def processed_dir(tmp_path: Path) -> Path:
    """Return a directory containing valid prepared training artifacts."""

    directory = tmp_path / "processed"
    write_valid_training_data(directory)
    return directory


def test_load_valid_training_data(processed_dir: Path) -> None:
    training_data = load_training_data(processed_dir)

    assert training_data.images.shape == (40, 112, 92)
    assert training_data.images.dtype == np.uint8
    assert np.array_equal(training_data.labels, np.arange(40))
    assert training_data.label_map["s1"] == 0
    assert training_data.label_map["s40"] == 39


def test_missing_training_file_raises_clear_error(processed_dir: Path) -> None:
    (processed_dir / "X_train.npy").unlink()

    with pytest.raises(FileNotFoundError, match=r"X_train\.npy"):
        load_training_data(processed_dir)


def test_mismatched_sample_counts_raise_clear_error(processed_dir: Path) -> None:
    np.save(processed_dir / "y_train.npy", np.arange(39, dtype=np.int64))

    with pytest.raises(ValueError, match="sample counts differ"):
        load_training_data(processed_dir)


def test_invalid_image_shape_raises_clear_error(processed_dir: Path) -> None:
    invalid_images = np.zeros((40, 92, 112), dtype=np.uint8)
    np.save(processed_dir / "X_train.npy", invalid_images)

    with pytest.raises(ValueError, match=r"X_train must have shape"):
        load_training_data(processed_dir)


def test_invalid_image_dtype_raises_clear_error(processed_dir: Path) -> None:
    invalid_images = np.zeros((40, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    np.save(processed_dir / "X_train.npy", invalid_images)

    with pytest.raises(TypeError, match=r"dtype uint8"):
        load_training_data(processed_dir)


def test_invalid_label_mapping_raises_clear_error(processed_dir: Path) -> None:
    invalid_mapping = {f"s{number}": number - 1 for number in range(1, 41)}
    invalid_mapping["s40"] = 38
    (processed_dir / "label_map.json").write_text(
        json.dumps(invalid_mapping),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"40 unique labels from 0 to 39"):
        load_training_data(processed_dir)


def test_successful_model_creation_and_saving(
    processed_dir: Path,
    tmp_path: Path,
) -> None:
    training_data = load_training_data(processed_dir)
    recognizer = create_recognizer()
    model_path = tmp_path / "models" / "lbph_model.yml"

    train_and_save_model(recognizer, training_data, model_path)

    assert model_path.is_file()
    assert model_path.stat().st_size > 0
    assert recognizer.getRadius() == LBPH_RADIUS
    assert recognizer.getNeighbors() == LBPH_NEIGHBORS
    assert recognizer.getGridX() == LBPH_GRID_X
    assert recognizer.getGridY() == LBPH_GRID_Y
    assert recognizer.getThreshold() == LBPH_THRESHOLD


def test_successful_metadata_creation(
    processed_dir: Path,
    tmp_path: Path,
) -> None:
    training_data = load_training_data(processed_dir)
    model_path = tmp_path / "models" / "lbph_model.yml"
    metadata_path = tmp_path / "models" / "training_metadata.json"
    timestamp = datetime(2026, 7, 28, 10, 30, tzinfo=timezone.utc)

    metadata = create_training_metadata(
        training_data,
        model_path,
        timestamp=timestamp,
    )
    save_training_metadata(metadata, metadata_path)
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert saved_metadata["algorithm"] == ALGORITHM_NAME
    assert saved_metadata["training_samples"] == 40
    assert saved_metadata["class_count"] == 40
    assert saved_metadata["image_dimensions"] == [112, 92]
    assert saved_metadata["label_range"] == {"minimum": 0, "maximum": 39}
    assert saved_metadata["model_path"] == str(model_path.resolve())
    assert saved_metadata["lbph_parameters"] == {
        "radius": LBPH_RADIUS,
        "neighbors": LBPH_NEIGHBORS,
        "grid_x": LBPH_GRID_X,
        "grid_y": LBPH_GRID_Y,
        "threshold": LBPH_THRESHOLD,
    }
    assert saved_metadata["training_timestamp_utc"] == "2026-07-28T10:30:00Z"
