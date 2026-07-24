from pathlib import Path

import cv2


IMAGE_PATH = Path("data/raw/s1/1.pgm")
OUTPUT_PATH = Path("outputs/inspected_face.png")


def main() -> None:
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    print(f"Image path: {IMAGE_PATH}")
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Minimum pixel value: {image.min()}")
    print(f"Maximum pixel value: {image.max()}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(str(OUTPUT_PATH), image)

    if not saved:
        raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

    print(f"Saved preview to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()