"""Tests for held-out registered-face model validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.validate_registered_model import (
    build_validation_summary,
    collect_registered_predictions,
    load_registered_validation_data,
    save_validation_outputs,
)


def write_validation_artifacts(combined_dir: Path) -> None:
    """Write two valid held-out samples for one registered identity."""

    combined_dir.mkdir(parents=True)
    images = np.stack(
        (
            np.full((112, 92), 100, dtype=np.uint8),
            np.full((112, 92), 110, dtype=np.uint8),
        )
    )
    labels = np.full(2, 40, dtype=np.int64)
    label_map = {
        **{f"s{number}": number - 1 for number in range(1, 41)},
        "chathuranga": 40,
    }
    manifest = [
        {
            "validation_index": index,
            "subject_key": "chathuranga",
            "display_name": "Chathuranga",
            "assigned_label": 40,
            "filename": f"{index + 17:03d}.png",
            "source_path": f"/registered/chathuranga/{index + 17:03d}.png",
        }
        for index in range(2)
    ]
    np.save(combined_dir / "X_registered_validation.npy", images)
    np.save(combined_dir / "y_registered_validation.npy", labels)
    (combined_dir / "label_map_combined.json").write_text(
        json.dumps(label_map),
        encoding="utf-8",
    )
    (combined_dir / "registered_validation_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_successful_held_out_local_prediction(tmp_path: Path) -> None:
    class ChathurangaRecognizer:
        def __init__(self) -> None:
            self.call_count = 0

        def predict(self, image: np.ndarray) -> tuple[int, float]:
            distance = 42.0 + self.call_count
            self.call_count += 1
            return 40, distance

    combined_dir = tmp_path / "combined"
    write_validation_artifacts(combined_dir)
    validation_data = load_registered_validation_data(combined_dir)

    predictions = collect_registered_predictions(
        ChathurangaRecognizer(),
        validation_data,
    )
    summary = build_validation_summary(predictions)

    assert len(predictions) == 2
    assert all(prediction.correct for prediction in predictions)
    assert summary["accuracy"] == 1.0
    assert summary["predicted_labels"] == [40, 40]
    assert summary["subjects"]["chathuranga"]["distance_statistics"] == {
        "count": 2,
        "mean": 42.5,
        "minimum": 42.0,
        "maximum": 43.0,
    }


def test_registered_validation_output_creation(tmp_path: Path) -> None:
    class ChathurangaRecognizer:
        def predict(self, image: np.ndarray) -> tuple[int, float]:
            return 40, 50.0

    combined_dir = tmp_path / "combined"
    output_dir = tmp_path / "outputs"
    write_validation_artifacts(combined_dir)
    validation_data = load_registered_validation_data(combined_dir)
    predictions = collect_registered_predictions(
        ChathurangaRecognizer(),
        validation_data,
    )
    summary = build_validation_summary(predictions)

    save_validation_outputs(predictions, summary, output_dir)

    saved_summary = json.loads(
        (output_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    with (output_dir / "predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        saved_predictions = list(csv.DictReader(csv_file))
    assert saved_summary["validation_samples"] == 2
    assert saved_summary["accuracy"] == 1.0
    assert len(saved_predictions) == 2
    assert saved_predictions[0]["predicted_label"] == "40"
