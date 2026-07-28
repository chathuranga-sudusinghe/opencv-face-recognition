"""Tests for combining original and registered face training data."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.prepare_registered_training import (
    EXPECTED_IMAGE_SHAPE,
    combine_training_data,
    load_registered_subject,
    load_registered_subjects,
    save_combined_artifacts,
)
from src.train_model import TrainingData


def make_original_data() -> TrainingData:
    """Return one original sample for each of the 40 original labels."""

    return TrainingData(
        images=np.zeros((40, *EXPECTED_IMAGE_SHAPE), dtype=np.uint8),
        labels=np.arange(40, dtype=np.int64),
        label_map={f"s{number}": number - 1 for number in range(1, 41)},
    )


def write_registered_subject(
    registered_dir: Path,
    subject_key: str,
    *,
    sample_count: int = 10,
    invalid_shape_at: int | None = None,
    include_metadata: bool = True,
) -> Path:
    """Write a synthetic registered subject folder."""

    subject_dir = registered_dir / subject_key
    subject_dir.mkdir(parents=True)
    if include_metadata:
        (subject_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "original_name": subject_key.title(),
                    "safe_folder_name": subject_key,
                }
            ),
            encoding="utf-8",
        )

    for image_number in range(1, sample_count + 1):
        shape = (
            (92, 112)
            if image_number == invalid_shape_at
            else EXPECTED_IMAGE_SHAPE
        )
        image = np.full(shape, image_number, dtype=np.uint8)
        assert cv2.imwrite(str(subject_dir / f"{image_number:03d}.png"), image)
    return subject_dir


def test_deterministic_new_label_assignment(tmp_path: Path) -> None:
    registered_dir = tmp_path / "registered"
    write_registered_subject(registered_dir, "zeta")
    write_registered_subject(registered_dir, "alpha")

    subjects = load_registered_subjects(registered_dir)
    _, labels, label_map, records, _, validation_labels, _ = combine_training_data(
        make_original_data(),
        subjects,
    )

    assert [subject.subject_key for subject in subjects] == ["alpha", "zeta"]
    assert label_map["alpha"] == 40
    assert label_map["zeta"] == 41
    assert [record["assigned_label"] for record in records] == [40, 41]
    assert np.array_equal(labels[-16:], np.repeat([40, 41], 8))
    assert np.array_equal(validation_labels, np.repeat([40, 41], 2))


def test_original_labels_remain_unchanged(tmp_path: Path) -> None:
    registered_dir = tmp_path / "registered"
    write_registered_subject(registered_dir, "local_user")
    original = make_original_data()

    _, labels, label_map, _, _, _, _ = combine_training_data(
        original,
        load_registered_subjects(registered_dir),
    )

    assert np.array_equal(labels[:40], original.labels)
    assert {key: label_map[key] for key in original.label_map} == original.label_map


def test_registered_images_load_in_numeric_order(tmp_path: Path) -> None:
    subject_dir = write_registered_subject(
        tmp_path / "registered",
        "person",
        sample_count=12,
    )

    subject = load_registered_subject(subject_dir)

    all_images = np.concatenate(
        (subject.training_images, subject.validation_images)
    )
    assert subject.training_images.shape == (9, 112, 92)
    assert subject.validation_images.shape == (3, 112, 92)
    assert all_images.dtype == np.uint8
    assert [int(image[0, 0]) for image in all_images] == list(range(1, 13))
    assert subject.display_name == "Person"


def test_deterministic_80_20_split_has_no_file_overlap(tmp_path: Path) -> None:
    subject_dir = write_registered_subject(
        tmp_path / "registered",
        "chathuranga",
        sample_count=20,
    )

    subject = load_registered_subject(subject_dir)
    training_names = [path.name for path in subject.training_image_paths]
    validation_names = [path.name for path in subject.validation_image_paths]

    assert training_names == [f"{number:03d}.png" for number in range(1, 17)]
    assert validation_names == [f"{number:03d}.png" for number in range(17, 21)]
    assert set(training_names).isdisjoint(validation_names)


def test_invalid_registered_image_shape(tmp_path: Path) -> None:
    subject_dir = write_registered_subject(
        tmp_path / "registered",
        "person",
        invalid_shape_at=4,
    )

    with pytest.raises(ValueError, match=r"grayscale with shape \(112, 92\)"):
        load_registered_subject(subject_dir)


def test_insufficient_registered_samples(tmp_path: Path) -> None:
    subject_dir = write_registered_subject(
        tmp_path / "registered",
        "person",
        sample_count=9,
    )

    with pytest.raises(ValueError, match=r"at least 10 PNG images"):
        load_registered_subject(subject_dir)


def test_missing_registration_metadata(tmp_path: Path) -> None:
    subject_dir = write_registered_subject(
        tmp_path / "registered",
        "person",
        include_metadata=False,
    )

    with pytest.raises(FileNotFoundError, match=r"Missing registration metadata"):
        load_registered_subject(subject_dir)


def test_combined_array_shapes_and_label_map(tmp_path: Path) -> None:
    registered_dir = tmp_path / "registered"
    write_registered_subject(
        registered_dir,
        "chathuranga",
        sample_count=20,
    )

    combined = combine_training_data(
        make_original_data(),
        load_registered_subjects(registered_dir),
    )
    (
        images,
        labels,
        label_map,
        records,
        validation_images,
        validation_labels,
        validation_manifest,
    ) = combined
    output_dir = tmp_path / "combined"
    save_combined_artifacts(combined, output_dir)

    assert images.shape == (56, 112, 92)
    assert labels.shape == (56,)
    assert validation_images.shape == (4, 112, 92)
    assert validation_labels.shape == (4,)
    assert len(label_map) == 41
    assert label_map["chathuranga"] == 40
    assert records[0]["sample_count"] == 20
    assert records[0]["training_sample_count"] == 16
    assert records[0]["validation_sample_count"] == 4
    assert [record["filename"] for record in validation_manifest] == [
        "017.png",
        "018.png",
        "019.png",
        "020.png",
    ]
    assert np.load(output_dir / "X_train_combined.npy").shape == (56, 112, 92)
    assert np.load(output_dir / "X_registered_validation.npy").shape == (
        4,
        112,
        92,
    )
    assert np.load(output_dir / "y_registered_validation.npy").shape == (4,)
    assert json.loads(
        (output_dir / "label_map_combined.json").read_text(encoding="utf-8")
    ) == label_map
    assert json.loads(
        (
            output_dir / "registered_validation_manifest.json"
        ).read_text(encoding="utf-8")
    ) == validation_manifest
