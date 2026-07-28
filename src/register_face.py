"""Capture consent-based face samples from a webcam for later retraining."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "registered_faces"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 92
MAX_CONSECUTIVE_READ_FAILURES = 10
WINDOW_NAME = "Face Registration"
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
ALPHANUMERIC_PATTERN = re.compile(r"[A-Za-z0-9]")

FaceBox = tuple[int, int, int, int]


class Camera(Protocol):
    """Minimal webcam interface used by the registration workflow."""

    def isOpened(self) -> bool:
        """Return whether the camera opened successfully."""

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read one frame."""

    def release(self) -> None:
        """Release the camera resource."""


class FaceDetector(Protocol):
    """Minimal Haar cascade interface used by the capture loop."""

    def detectMultiScale(
        self,
        image: NDArray[np.uint8],
        *,
        scaleFactor: float,
        minNeighbors: int,
        minSize: tuple[int, int],
    ) -> Any:
        """Detect face rectangles."""


def sanitize_person_name(name: str) -> str:
    """Validate a person name and return its lowercase folder-safe form."""

    normalized = name.strip()
    if not normalized:
        raise ValueError("Person name must not be empty")
    if not SAFE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Person name may contain only letters, numbers, hyphens, "
            "and underscores"
        )
    if not ALPHANUMERIC_PATTERN.search(normalized):
        raise ValueError("Person name must contain at least one letter or number")
    return normalized.lower()


def prepare_registration_directory(
    output_dir: Path,
    safe_name: str,
    *,
    overwrite: bool,
) -> Path:
    """Create a protected subject directory, optionally clearing old samples."""

    target_dir = output_dir / safe_name
    existing_images = (
        [
            path
            for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ]
        if target_dir.is_dir()
        else []
    )
    if existing_images and not overwrite:
        raise FileExistsError(
            f"Registration folder already contains images: {target_dir}. "
            "Use --overwrite to replace them."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for image_path in existing_images:
            image_path.unlink()
        metadata_path = target_dir / "metadata.json"
        if metadata_path.is_file():
            metadata_path.unlink()

    return target_dir


def load_face_detector() -> FaceDetector:
    """Load OpenCV's default frontal-face Haar cascade."""

    cascade_path = Path(cv2.data.haarcascades) / CASCADE_FILENAME
    if not cascade_path.is_file():
        raise FileNotFoundError(f"Haar cascade file does not exist: {cascade_path}")

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV could not load Haar cascade: {cascade_path}")
    return detector


def open_camera(
    camera_index: int,
    *,
    camera_factory: Callable[[int], Camera] | None = None,
) -> Camera:
    """Open a webcam or raise a clear error while releasing failed handles."""

    factory = camera_factory or cv2.VideoCapture
    camera = factory(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam at camera index {camera_index}")
    return camera


def select_largest_face(faces: Any) -> FaceBox | None:
    """Return the largest detected face rectangle by area."""

    face_list = [tuple(int(value) for value in face) for face in faces]
    if not face_list:
        return None
    return max(face_list, key=lambda face: face[2] * face[3])  # type: ignore[return-value]


def preprocess_face(
    grayscale_frame: NDArray[np.uint8],
    face_box: FaceBox,
) -> NDArray[np.uint8]:
    """Crop, resize, and equalize one detected grayscale face."""

    if grayscale_frame.ndim != 2:
        raise ValueError(
            f"Expected a grayscale frame with 2 dimensions, got "
            f"{grayscale_frame.shape}"
        )

    x, y, width, height = face_box
    if width <= 0 or height <= 0:
        raise ValueError(f"Detected face has invalid dimensions: {face_box}")
    face = grayscale_frame[y : y + height, x : x + width]
    if face.size == 0:
        raise ValueError(f"Detected face crop is empty: {face_box}")

    resized = cv2.resize(
        face,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.equalizeHist(resized)


def save_face_image(image: NDArray[np.uint8], output_path: Path) -> None:
    """Save one preprocessed grayscale PNG or raise on failure."""

    saved = cv2.imwrite(str(output_path), image)
    if not saved:
        raise OSError(f"OpenCV could not save captured face: {output_path}")


def create_registration_metadata(
    original_name: str,
    safe_name: str,
    requested_samples: int,
    captured_samples: int,
    *,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    """Build serializable metadata for a completed or partial registration."""

    registration_time = timestamp or datetime.now(timezone.utc)
    if registration_time.tzinfo is None:
        raise ValueError("Registration timestamp must include timezone information")
    timestamp_utc = registration_time.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )

    return {
        "original_name": original_name,
        "safe_folder_name": safe_name,
        "requested_sample_count": requested_samples,
        "captured_sample_count": captured_samples,
        "image_dimensions": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "preprocessing_steps": [
            "convert frame to grayscale",
            "crop largest detected face",
            "resize to 112 x 92 pixels",
            "apply histogram equalization",
        ],
        "registration_timestamp_utc": timestamp_utc,
    }


def save_registration_metadata(
    metadata: dict[str, object],
    metadata_path: Path,
) -> None:
    """Write registration metadata as formatted JSON."""

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _draw_capture_overlay(
    frame: NDArray[np.uint8],
    face_box: FaceBox | None,
    captured_samples: int,
    requested_samples: int,
) -> None:
    """Draw the selected face, progress, and keyboard instructions."""

    if face_box is not None:
        x, y, width, height = face_box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        face_status = "Face detected"
    else:
        face_status = "No face detected"

    cv2.putText(
        frame,
        f"Captured {captured_samples}/{requested_samples}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        face_status,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "SPACE: capture | Q: quit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def capture_samples(
    camera: Camera,
    detector: FaceDetector,
    target_dir: Path,
    requested_samples: int,
) -> int:
    """Run the interactive webcam loop and return the captured sample count."""

    captured_samples = 0
    consecutive_read_failures = 0

    print("Capture only with the person's informed consent.")
    print("Press SPACE to capture the largest detected face; press Q to quit.")

    while captured_samples < requested_samples:
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
        faces = detector.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        selected_face = select_largest_face(faces)
        display_frame = frame.copy()
        _draw_capture_overlay(
            display_frame,
            selected_face,
            captured_samples,
            requested_samples,
        )
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("Registration ended early by user request.")
            break
        if key == ord(" ") and selected_face is not None:
            processed_face = preprocess_face(grayscale, selected_face)
            sample_number = captured_samples + 1
            save_face_image(
                processed_face,
                target_dir / f"{sample_number:03d}.png",
            )
            captured_samples = sample_number
            print(f"Captured {captured_samples}/{requested_samples}")

    return captured_samples


def register_face(
    name: str,
    samples: int,
    camera_index: int,
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    """Run face registration and return the registered subject directory."""

    if samples <= 0:
        raise ValueError("--samples must be greater than zero")
    if camera_index < 0:
        raise ValueError("--camera-index must be zero or greater")

    original_name = name.strip()
    safe_name = sanitize_person_name(name)
    detector = load_face_detector()
    camera: Camera | None = None

    try:
        camera = open_camera(camera_index)
        target_dir = prepare_registration_directory(
            output_dir,
            safe_name,
            overwrite=overwrite,
        )
        captured_samples = capture_samples(
            camera,
            detector,
            target_dir,
            samples,
        )
        metadata = create_registration_metadata(
            original_name,
            safe_name,
            samples,
            captured_samples,
        )
        save_registration_metadata(metadata, target_dir / "metadata.json")
        return target_dir
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for webcam registration."""

    parser = argparse.ArgumentParser(
        description=(
            "Capture consent-based face samples for later model retraining."
        )
    )
    parser.add_argument("--name", required=True, help="Person name or local identifier")
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of images to capture (default: 20)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Registration dataset root (default: data/registered_faces)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNG samples for this person",
    )
    return parser.parse_args()


def main() -> None:
    """Run registration from parsed command-line arguments."""

    args = parse_args()
    target_dir = register_face(
        name=args.name,
        samples=args.samples,
        camera_index=args.camera_index,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Registration data saved to: {target_dir.resolve()}")


if __name__ == "__main__":
    main()
