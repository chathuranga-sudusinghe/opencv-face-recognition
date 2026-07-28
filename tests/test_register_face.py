"""Tests for safe, testable face-registration helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.register_face import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    create_registration_metadata,
    open_camera,
    prepare_registration_directory,
    preprocess_face,
    sanitize_person_name,
    save_face_image,
    save_registration_metadata,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Chathuranga", "chathuranga"),
        ("Local_User-01", "local_user-01"),
        ("  Person2  ", "person2"),
    ],
)
def test_valid_name_sanitization(name: str, expected: str) -> None:
    assert sanitize_person_name(name) == expected


@pytest.mark.parametrize(
    "name",
    ["", "   ", "../person", "person/name", r"person\name", "Jane Doe", "---"],
)
def test_invalid_name_rejection(name: str) -> None:
    with pytest.raises(ValueError):
        sanitize_person_name(name)


def test_existing_dataset_protection(tmp_path: Path) -> None:
    target_dir = tmp_path / "registered_faces" / "person"
    target_dir.mkdir(parents=True)
    (target_dir / "001.png").write_bytes(b"existing image")

    with pytest.raises(FileExistsError, match=r"Use --overwrite"):
        prepare_registration_directory(
            tmp_path / "registered_faces",
            "person",
            overwrite=False,
        )


def test_overwrite_behavior(tmp_path: Path) -> None:
    target_dir = tmp_path / "registered_faces" / "person"
    target_dir.mkdir(parents=True)
    (target_dir / "001.png").write_bytes(b"existing image")
    (target_dir / "metadata.json").write_text("{}", encoding="utf-8")

    prepared_dir = prepare_registration_directory(
        tmp_path / "registered_faces",
        "person",
        overwrite=True,
    )

    assert prepared_dir == target_dir
    assert prepared_dir.is_dir()
    assert not (prepared_dir / "001.png").exists()
    assert not (prepared_dir / "metadata.json").exists()


def test_face_preprocessing_output_shape() -> None:
    grayscale = np.arange(160 * 160, dtype=np.uint8).reshape(160, 160)

    processed = preprocess_face(grayscale, (20, 30, 100, 110))

    assert processed.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert processed.dtype == np.uint8
    assert processed.ndim == 2


def test_metadata_creation(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 28, 12, 30, tzinfo=timezone.utc)
    metadata = create_registration_metadata(
        "Local_User",
        "local_user",
        20,
        12,
        timestamp=timestamp,
    )
    metadata_path = tmp_path / "metadata.json"
    save_registration_metadata(metadata, metadata_path)
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert saved["original_name"] == "Local_User"
    assert saved["safe_folder_name"] == "local_user"
    assert saved["requested_sample_count"] == 20
    assert saved["captured_sample_count"] == 12
    assert saved["image_dimensions"] == [112, 92]
    assert saved["registration_timestamp_utc"] == "2026-07-28T12:30:00Z"
    assert saved["preprocessing_steps"] == [
        "convert frame to grayscale",
        "crop largest detected face",
        "resize to 112 x 92 pixels",
        "apply histogram equalization",
    ]


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


def test_failed_image_saving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.register_face.cv2.imwrite", lambda *_: False)
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

    with pytest.raises(OSError, match=r"could not save captured face"):
        save_face_image(image, tmp_path / "001.png")
