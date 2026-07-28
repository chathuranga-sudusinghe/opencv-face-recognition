"""Run real-time webcam identification with a combined OpenCV LBPH model."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    from src.evaluate_model import load_recognizer
except ModuleNotFoundError:  # Support direct execution as src/<script>.py.
    from evaluate_model import load_recognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "lbph_combined_model.yml"
DEFAULT_LABEL_MAP_PATH = PROJECT_ROOT / "data" / "combined" / "label_map_combined.json"
DEFAULT_REGISTERED_SUBJECTS_PATH = (
    PROJECT_ROOT / "data" / "combined" / "registered_subjects.json"
)
DEFAULT_SCREENSHOT_DIR = PROJECT_ROOT / "outputs" / "realtime"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
EXPECTED_IMAGE_SHAPE = (112, 92)
SMOOTHING_WINDOW_SIZE = 5
MAX_CONSECUTIVE_READ_FAILURES = 10
WINDOW_NAME = "Real-Time Face Identification"

FaceBox = tuple[int, int, int, int]


class Recognizer(Protocol):
    """LBPH prediction operation used by real-time classification."""

    def predict(self, image: NDArray[np.uint8]) -> tuple[int, float]:
        """Return the closest label and LBPH distance."""


class Camera(Protocol):
    """Minimal OpenCV camera interface."""

    def isOpened(self) -> bool:
        """Return whether the camera opened successfully."""

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read one webcam frame."""

    def release(self) -> None:
        """Release the camera resource."""


@dataclass(frozen=True)
class RealtimeArtifacts:
    """Loaded recognizer, labels, display names, and face detector."""

    recognizer: Recognizer
    label_map: dict[str, int]
    display_names: dict[int, str]
    detector: Any


@dataclass(frozen=True)
class RawPrediction:
    """Closest LBPH label and its distance before thresholding."""

    label: int
    distance: float


@dataclass(frozen=True)
class DisplayPrediction:
    """Prediction state ready for annotation."""

    closest_label: int
    display_name: str
    distance: float
    is_known: bool | None
    diagnostic_mode: bool


def _load_json(path: Path, description: str) -> Any:
    """Load a required JSON file with a clear error."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {description} JSON: {path}") from exc


def resolve_display_names(
    label_map: dict[str, int],
    registered_subjects: list[dict[str, object]],
) -> dict[int, str]:
    """Resolve AT&T keys and registered identity display names by label."""

    display_names = {label: subject for subject, label in label_map.items()}
    for record in registered_subjects:
        subject_key = record.get("subject_key")
        display_name = record.get("display_name")
        assigned_label = record.get("assigned_label")
        if (
            not isinstance(subject_key, str)
            or not isinstance(display_name, str)
            or not display_name.strip()
            or not isinstance(assigned_label, int)
            or isinstance(assigned_label, bool)
            or label_map.get(subject_key) != assigned_label
        ):
            raise ValueError(f"Invalid registered subject record: {record}")
        display_names[assigned_label] = display_name.strip()
    return display_names


def load_artifacts(
    model_path: Path,
    label_map_path: Path,
    registered_subjects_path: Path,
) -> RealtimeArtifacts:
    """Load and validate all model, identity, and detector artifacts."""

    label_map = _load_json(label_map_path, "combined label map")
    if not isinstance(label_map, dict) or not all(
        isinstance(subject, str)
        and isinstance(label, int)
        and not isinstance(label, bool)
        for subject, label in label_map.items()
    ):
        raise ValueError("Combined label map must contain string keys and integer values")
    if set(label_map.values()) != set(range(len(label_map))):
        raise ValueError("Combined label map labels must be contiguous from zero")

    registered_subjects = _load_json(
        registered_subjects_path,
        "registered subjects metadata",
    )
    if not isinstance(registered_subjects, list) or not all(
        isinstance(record, dict) for record in registered_subjects
    ):
        raise ValueError("Registered subjects metadata must be a list of objects")

    display_names = resolve_display_names(label_map, registered_subjects)
    recognizer = load_recognizer(model_path)

    cascade_path = Path(cv2.data.haarcascades) / CASCADE_FILENAME
    if not cascade_path.is_file():
        raise FileNotFoundError(f"Missing Haar cascade: {cascade_path}")
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV could not load Haar cascade: {cascade_path}")

    return RealtimeArtifacts(
        recognizer=recognizer,
        label_map=label_map,
        display_names=display_names,
        detector=detector,
    )


def preprocess_detected_face(
    grayscale_frame: NDArray[np.uint8],
    face_box: FaceBox,
) -> NDArray[np.uint8]:
    """Crop, resize to 112x92, and equalize one detected face."""

    if grayscale_frame.ndim != 2:
        raise ValueError(
            f"Expected a grayscale frame, got shape {grayscale_frame.shape}"
        )
    x, y, width, height = face_box
    if width <= 0 or height <= 0:
        raise ValueError(f"Detected face has invalid dimensions: {face_box}")
    face = grayscale_frame[y : y + height, x : x + width]
    if face.size == 0:
        raise ValueError(f"Detected face crop is empty: {face_box}")
    resized = cv2.resize(
        face,
        (EXPECTED_IMAGE_SHAPE[1], EXPECTED_IMAGE_SHAPE[0]),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.equalizeHist(resized)


def classify_face(
    recognizer: Recognizer,
    face_image: NDArray[np.uint8],
    valid_labels: set[int],
) -> RawPrediction:
    """Predict one face and reject labels absent from the combined map."""

    predicted_label, distance = recognizer.predict(face_image)
    label = int(predicted_label)
    if label not in valid_labels:
        raise ValueError(
            f"LBPH model predicted label {label}, which is absent from the label map"
        )
    return RawPrediction(label=label, distance=float(distance))


def apply_threshold(
    prediction: RawPrediction,
    display_names: dict[int, str],
    threshold: float | None,
    unknown_label: str,
) -> DisplayPrediction:
    """Apply an optional rejection threshold or retain diagnostic identity."""

    if prediction.label not in display_names:
        raise ValueError(f"No display name exists for label {prediction.label}")
    if threshold is None:
        return DisplayPrediction(
            closest_label=prediction.label,
            display_name=display_names[prediction.label],
            distance=prediction.distance,
            is_known=None,
            diagnostic_mode=True,
        )

    is_known = prediction.distance <= threshold
    return DisplayPrediction(
        closest_label=prediction.label,
        display_name=(
            display_names[prediction.label] if is_known else unknown_label
        ),
        distance=prediction.distance,
        is_known=is_known,
        diagnostic_mode=False,
    )


def smooth_predictions(history: Sequence[RawPrediction]) -> RawPrediction:
    """Return majority label and mean distance from a recent history."""

    if not history:
        raise ValueError("Cannot smooth an empty prediction history")
    counts = Counter(prediction.label for prediction in history)
    highest_count = max(counts.values())
    tied_labels = {
        label for label, count in counts.items() if count == highest_count
    }
    majority_label = next(
        prediction.label
        for prediction in reversed(history)
        if prediction.label in tied_labels
    )
    mean_distance = float(
        np.mean([prediction.distance for prediction in history])
    )
    return RawPrediction(label=majority_label, distance=mean_distance)


def draw_annotation(
    frame: NDArray[np.uint8],
    face_box: FaceBox,
    prediction: DisplayPrediction,
    *,
    show_distance: bool,
) -> None:
    """Draw a face box, identity text, and diagnostic threshold warning."""

    x, y, width, height = face_box
    color = (0, 0, 255) if prediction.is_known is False else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    label_text = prediction.display_name
    if show_distance:
        label_text += f" ({prediction.distance:.2f})"
    cv2.putText(
        frame,
        label_text,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
    if prediction.diagnostic_mode:
        cv2.putText(
            frame,
            "Threshold not calibrated",
            (x, min(frame.shape[0] - 10, y + height + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 165, 255),
            2,
        )


def open_camera(
    camera_index: int,
    *,
    camera_factory: Callable[[int], Camera] | None = None,
) -> Camera:
    """Open a webcam and release a failed handle before raising."""

    factory = camera_factory or cv2.VideoCapture
    camera = factory(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam at camera index {camera_index}")
    return camera


def save_annotated_frame(
    frame: NDArray[np.uint8],
    output_dir: Path,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Save a timestamped annotated frame or raise on write failure."""

    capture_time = timestamp or datetime.now(timezone.utc)
    if capture_time.tzinfo is None:
        raise ValueError("Screenshot timestamp must include timezone information")
    filename_time = capture_time.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"realtime_{filename_time}.png"
    if not cv2.imwrite(str(output_path), frame):
        raise OSError(f"OpenCV could not save annotated frame: {output_path}")
    return output_path


def _print_latest_distances(predictions: Sequence[DisplayPrediction]) -> None:
    """Print the latest visible identities and smoothed distances."""

    if not predictions:
        print("No faces detected in the latest frame.")
        return
    for face_index, prediction in enumerate(predictions, start=1):
        print(
            f"Face {face_index}: closest_label={prediction.closest_label}, "
            f"display={prediction.display_name}, "
            f"distance={prediction.distance:.4f}"
        )


def run_realtime_recognition(
    camera_index: int,
    model_path: Path,
    label_map_path: Path,
    registered_subjects_path: Path,
    threshold: float | None,
    show_distance: bool,
    min_face_size: int,
    unknown_label: str,
) -> list[Path]:
    """Run the interactive native webcam recognition loop."""

    if min_face_size <= 0:
        raise ValueError("--min-face-size must be greater than zero")
    if threshold is not None and threshold < 0:
        raise ValueError("--threshold must be zero or greater")
    if not unknown_label.strip():
        raise ValueError("--unknown-label must not be empty")

    artifacts = load_artifacts(
        model_path,
        label_map_path,
        registered_subjects_path,
    )
    camera: Camera | None = None
    saved_screenshots: list[Path] = []
    histories: dict[int, deque[RawPrediction]] = {}
    consecutive_read_failures = 0
    latest_predictions: list[DisplayPrediction] = []

    if threshold is None:
        print(
            "Diagnostic mode: threshold is not calibrated; showing closest "
            "identity and distance."
        )
    else:
        print(
            f"Experimental threshold mode: {threshold:.4f}. "
            "This is not a production-calibrated unknown-person threshold."
        )
    print("Controls: Q=quit, S=save annotated frame, T=print distances")

    try:
        camera = open_camera(camera_index)
        while True:
            readable, frame = camera.read()
            if not readable or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    raise RuntimeError(
                        "Webcam returned too many consecutive unreadable frames"
                    )
                continue

            consecutive_read_failures = 0
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = artifacts.detector.detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size),
            )
            face_boxes = sorted(
                (
                    tuple(int(value) for value in face)
                    for face in detected_faces
                ),
                key=lambda face: (face[0], face[1]),
            )
            latest_predictions = []
            active_slots: set[int] = set()
            for slot, face_box_values in enumerate(face_boxes):
                face_box: FaceBox = face_box_values  # type: ignore[assignment]
                active_slots.add(slot)
                face_image = preprocess_detected_face(grayscale, face_box)
                raw_prediction = classify_face(
                    artifacts.recognizer,
                    face_image,
                    set(artifacts.display_names),
                )
                history = histories.setdefault(
                    slot,
                    deque(maxlen=SMOOTHING_WINDOW_SIZE),
                )
                history.append(raw_prediction)
                smoothed = smooth_predictions(history)
                display_prediction = apply_threshold(
                    smoothed,
                    artifacts.display_names,
                    threshold,
                    unknown_label,
                )
                latest_predictions.append(display_prediction)
                draw_annotation(
                    frame,
                    face_box,
                    display_prediction,
                    show_distance=show_distance,
                )
            histories = {
                slot: history
                for slot, history in histories.items()
                if slot in active_slots
            }

            cv2.putText(
                frame,
                "Q: quit | S: screenshot | T: print distances",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                screenshot_path = save_annotated_frame(
                    frame,
                    DEFAULT_SCREENSHOT_DIR,
                )
                saved_screenshots.append(screenshot_path)
                print(f"Saved annotated frame: {screenshot_path.resolve()}")
            if key in (ord("t"), ord("T")):
                _print_latest_distances(latest_predictions)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

    return saved_screenshots


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for real-time face identification."""

    parser = argparse.ArgumentParser(
        description="Run real-time LBPH face identification from a webcam."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument(
        "--registered-subjects",
        type=Path,
        default=DEFAULT_REGISTERED_SUBJECTS_PATH,
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--show-distance", action="store_true")
    parser.add_argument("--min-face-size", type=int, default=80)
    parser.add_argument("--unknown-label", default="Unknown")
    return parser.parse_args()


def main() -> None:
    """Run real-time identification using parsed CLI options."""

    args = parse_args()
    run_realtime_recognition(
        camera_index=args.camera_index,
        model_path=args.model_path,
        label_map_path=args.label_map,
        registered_subjects_path=args.registered_subjects,
        threshold=args.threshold,
        show_distance=args.show_distance,
        min_face_size=args.min_face_size,
        unknown_label=args.unknown_label,
    )


if __name__ == "__main__":
    main()
