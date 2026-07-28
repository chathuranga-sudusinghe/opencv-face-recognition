from pathlib import Path

import cv2
import numpy as np


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("outputs/face_preview_grid.png")

SUBJECT_COUNT = 10
IMAGES_PER_SUBJECT = 5
SCALE = 2


def main() -> None:
    rows: list[np.ndarray] = []

    for subject_number in range(1, SUBJECT_COUNT + 1):
        subject_images: list[np.ndarray] = []

        for image_number in range(1, IMAGES_PER_SUBJECT + 1):
            image_path = (
                RAW_DIR
                / f"s{subject_number}"
                / f"{image_number}.pgm"
            )

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                raise FileNotFoundError(
                    f"Could not load image: {image_path}"
                )

            resized = cv2.resize(
                image,
                None,
                fx=SCALE,
                fy=SCALE,
                interpolation=cv2.INTER_NEAREST,
            )

            subject_images.append(resized)

        rows.append(np.hstack(subject_images))

    preview_grid = np.vstack(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), preview_grid):
        raise RuntimeError(
            f"Could not save preview grid: {OUTPUT_PATH}"
        )

    print(f"Saved preview grid to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()