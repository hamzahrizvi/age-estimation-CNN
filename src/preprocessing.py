"""Image preprocessing utilities.

The original dissertation code used MTCNN/face alignment, OpenCV, and 224x224
resizing. This module keeps that workflow but adds clearer error logging.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import pandas as pd


IMAGE_SIZE = (224, 224)


def resize_image(input_path: str | Path, output_path: str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> bool:
    """Read, resize, and save one image. Returns True on success."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = cv2.imread(str(input_path))
    if image is None:
        return False
    image = cv2.resize(image, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def preprocess_from_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    path_col: str = "full_path",
    image_size: tuple[int, int] = IMAGE_SIZE,
    failed_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Resize images listed in a CSV and write them into output_dir.

    If face-alignment is required, add it before the resize_image call.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    df = pd.read_csv(csv_path)

    if path_col not in df.columns:
        raise KeyError(f"Missing image path column: {path_col}")

    failed = []
    processed_paths = []

    for idx, row in df.iterrows():
        source = Path(str(row[path_col]))
        destination = output_dir / source.name
        try:
            ok = resize_image(source, destination, image_size=image_size)
            if not ok:
                raise ValueError("OpenCV could not read or write the image")
            processed_paths.append(str(destination))
        except Exception as exc:
            failed.append({"row": idx, "path": str(source), "error": str(exc)})
            processed_paths.append(None)

    out = df.copy()
    out["processed_path"] = processed_paths

    if failed_csv is not None:
        failed_csv = Path(failed_csv)
        failed_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed).to_csv(failed_csv, index=False)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess image files from a CSV.")
    parser.add_argument("--csv", required=True, help="Input CSV containing image paths.")
    parser.add_argument("--output-dir", required=True, help="Directory to save processed images.")
    parser.add_argument("--path-col", default="full_path", help="Column containing image paths.")
    parser.add_argument("--failed-csv", default="outputs/results/failed_preprocessing.csv")
    args = parser.parse_args()

    result = preprocess_from_csv(args.csv, args.output_dir, path_col=args.path_col, failed_csv=args.failed_csv)
    print(f"Processed rows: {len(result)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
