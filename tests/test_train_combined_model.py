"""Tests for training and documenting the separate combined LBPH model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.train_combined_model import (
    create_combined_training_metadata,
    load_combined_training_data,
    save_combined_training_metadata,
)
from src.train_model import create_recognizer, train_and_save_model


def write_combined_artifacts(combined_dir: Path) -> None:
    """Write valid synthetic combined artifacts with 41 classes."""

    combined_dir.mkdir(parents=True)
    original_images = np.zeros((40, 112, 92), dtype=np.uint8)
    registered_images = np.full((8, 112, 92), 128, dtype=np.uint8)
    images = np.concatenate((original_images, registered_images))
    labels = np.concatenate(
        (np.arange(40, dtype=np.int64), np.full(8, 40, dtype=np.int64))
    )
    label_map = {
        **{f"s{number}": number - 1 for number in range(1, 41)},
        "chathuranga": 40,
    }
    registered_records = [
        {
            "subject_key": "chathuranga",
            "display_name": "Chathuranga",
            "assigned_label": 40,
            "sample_count": 10,
            "training_sample_count": 8,
            "validation_sample_count": 2,
            "source_folder": "/registered/chathuranga",
        }
    ]
    np.save(combined_dir / "X_train_combined.npy", images)
    np.save(combined_dir / "y_train_combined.npy", labels)
    (combined_dir / "label_map_combined.json").write_text(
        json.dumps(label_map),
        encoding="utf-8",
    )
    (combined_dir / "registered_subjects.json").write_text(
        json.dumps(registered_records),
        encoding="utf-8",
    )


def test_successful_combined_model_creation_without_overwriting_original(
    tmp_path: Path,
) -> None:
    combined_dir = tmp_path / "combined"
    write_combined_artifacts(combined_dir)
    training_data, _, _, _ = load_combined_training_data(combined_dir)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    original_model_path = models_dir / "lbph_model.yml"
    original_model_path.write_text("original model sentinel", encoding="utf-8")
    combined_model_path = models_dir / "lbph_combined_model.yml"

    recognizer = create_recognizer()
    train_and_save_model(recognizer, training_data, combined_model_path)

    assert combined_model_path.is_file()
    assert combined_model_path.stat().st_size > 0
    assert original_model_path.read_text(encoding="utf-8") == (
        "original model sentinel"
    )


def test_successful_combined_metadata_creation(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    write_combined_artifacts(combined_dir)
    (
        training_data,
        registered_records,
        original_sample_count,
        registered_sample_count,
    ) = load_combined_training_data(combined_dir)
    model_path = tmp_path / "models" / "lbph_combined_model.yml"
    metadata_path = tmp_path / "models" / "combined_training_metadata.json"
    timestamp = datetime(2026, 7, 28, 14, 0, tzinfo=timezone.utc)

    metadata = create_combined_training_metadata(
        training_data,
        registered_records,
        original_sample_count,
        registered_sample_count,
        model_path,
        timestamp=timestamp,
    )
    save_combined_training_metadata(metadata, metadata_path)
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert saved["algorithm"] == "OpenCV LBPHFaceRecognizer"
    assert saved["total_training_samples"] == 48
    assert saved["original_sample_count"] == 40
    assert saved["registered_sample_count"] == 8
    assert saved["total_class_count"] == 41
    assert saved["image_dimensions"] == [112, 92]
    assert saved["label_range"] == {"minimum": 0, "maximum": 40}
    assert saved["registered_identities"] == registered_records
    assert saved["lbph_parameters"] == {
        "radius": 1,
        "neighbors": 8,
        "grid_x": 8,
        "grid_y": 8,
        "threshold": np.finfo(np.float64).max,
    }
    assert saved["model_path"] == str(model_path.resolve())
    assert saved["training_timestamp_utc"] == "2026-07-28T14:00:00Z"
