"""Tests for real-time LBPH recognition helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.realtime_recognition import (
    EXPECTED_IMAGE_SHAPE,
    RawPrediction,
    apply_threshold,
    classify_face,
    load_artifacts,
    open_camera,
    preprocess_detected_face,
    resolve_display_names,
    save_annotated_frame,
    smooth_predictions,
)


def complete_label_map() -> dict[str, int]:
    return {
        **{f"s{number}": number - 1 for number in range(1, 41)},
        "chathuranga": 40,
    }


def test_artifact_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.yml"
    model_path.write_text("model", encoding="utf-8")
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(json.dumps(complete_label_map()), encoding="utf-8")
    registered_path = tmp_path / "registered_subjects.json"
    registered_path.write_text(
        json.dumps(
            [
                {
                    "subject_key": "chathuranga",
                    "display_name": "Chathuranga",
                    "assigned_label": 40,
                }
            ]
        ),
        encoding="utf-8",
    )
    cascade_dir = tmp_path / "cascades"
    cascade_dir.mkdir()
    (cascade_dir / "haarcascade_frontalface_default.xml").write_text(
        "cascade",
        encoding="utf-8",
    )

    class Detector:
        def empty(self) -> bool:
            return False

    recognizer = object()
    monkeypatch.setattr(
        "src.realtime_recognition.load_recognizer",
        lambda _: recognizer,
    )
    monkeypatch.setattr(
        "src.realtime_recognition.cv2.data.haarcascades",
        str(cascade_dir) + "/",
    )
    monkeypatch.setattr(
        "src.realtime_recognition.cv2.CascadeClassifier",
        lambda _: Detector(),
    )

    artifacts = load_artifacts(model_path, label_map_path, registered_path)

    assert artifacts.recognizer is recognizer
    assert artifacts.label_map["chathuranga"] == 40
    assert artifacts.display_names[40] == "Chathuranga"
    assert isinstance(artifacts.detector, Detector)


def test_registered_display_name_resolution() -> None:
    display_names = resolve_display_names(
        complete_label_map(),
        [
            {
                "subject_key": "chathuranga",
                "display_name": "Chathuranga",
                "assigned_label": 40,
            }
        ],
    )

    assert display_names[0] == "s1"
    assert display_names[39] == "s40"
    assert display_names[40] == "Chathuranga"


def test_face_preprocessing_output_shape() -> None:
    grayscale = np.arange(180 * 180, dtype=np.uint8).reshape(180, 180)

    processed = preprocess_detected_face(grayscale, (20, 30, 100, 120))

    assert processed.shape == EXPECTED_IMAGE_SHAPE
    assert processed.dtype == np.uint8
    assert processed.ndim == 2


def test_known_prediction_below_threshold() -> None:
    result = apply_threshold(
        RawPrediction(label=40, distance=90.0),
        {40: "Chathuranga"},
        threshold=105.0,
        unknown_label="Unknown",
    )

    assert result.display_name == "Chathuranga"
    assert result.is_known is True
    assert result.diagnostic_mode is False


def test_unknown_prediction_above_threshold() -> None:
    result = apply_threshold(
        RawPrediction(label=40, distance=110.0),
        {40: "Chathuranga"},
        threshold=105.0,
        unknown_label="Unknown",
    )

    assert result.display_name == "Unknown"
    assert result.is_known is False


def test_diagnostic_mode_without_threshold() -> None:
    result = apply_threshold(
        RawPrediction(label=40, distance=99.0),
        {40: "Chathuranga"},
        threshold=None,
        unknown_label="Unknown",
    )

    assert result.display_name == "Chathuranga"
    assert result.is_known is None
    assert result.diagnostic_mode is True


def test_invalid_predicted_label() -> None:
    class InvalidRecognizer:
        def predict(self, image: np.ndarray) -> tuple[int, float]:
            return 99, 50.0

    with pytest.raises(ValueError, match=r"absent from the label map"):
        classify_face(
            InvalidRecognizer(),
            np.zeros(EXPECTED_IMAGE_SHAPE, dtype=np.uint8),
            {0, 40},
        )


def test_smoothing_uses_majority_label() -> None:
    result = smooth_predictions(
        [
            RawPrediction(40, 80.0),
            RawPrediction(2, 70.0),
            RawPrediction(40, 90.0),
        ]
    )

    assert result.label == 40


def test_smoothing_uses_mean_distance() -> None:
    result = smooth_predictions(
        [
            RawPrediction(40, 60.0),
            RawPrediction(40, 80.0),
            RawPrediction(40, 100.0),
        ]
    )

    assert result.distance == pytest.approx(80.0)


def test_failed_camera_opening_releases_camera() -> None:
    class ClosedCamera:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            self.released = True

    camera = ClosedCamera()

    with pytest.raises(RuntimeError, match=r"Could not open webcam"):
        open_camera(0, camera_factory=lambda _: camera)

    assert camera.released is True


def test_screenshot_save_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.realtime_recognition.cv2.imwrite", lambda *_: False)

    with pytest.raises(OSError, match=r"could not save annotated frame"):
        save_annotated_frame(
            np.zeros((120, 160, 3), dtype=np.uint8),
            tmp_path,
        )
