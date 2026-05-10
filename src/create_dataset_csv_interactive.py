from pathlib import Path
import argparse
import re
import cv2
import pandas as pd
from mtcnn import MTCNN


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


AGE_GROUPS = [
    (0, 2, 0),
    (3, 5, 1),
    (6, 13, 2),
    (14, 18, 3),
    (19, 24, 4),
    (25, 33, 5),
    (34, 48, 6),
    (49, 64, 7),
    (65, 120, 8),
]


def age_to_group(age: int) -> int:
    for low, high, group in AGE_GROUPS:
        if low <= age <= high:
            return group
    return -1


def blur_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def detect_face_quality(image_path, detector):
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        return {
            "face_status": "rejected",
            "face_reason": "cannot_read_image",
            "face_count": 0,
            "detection_confidence": 0.0,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "quality_score": 0.0,
        }

    blur = blur_score(image_bgr)
    brightness = brightness_score(image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    if len(faces) == 0:
        confidence = 0.0
        face_status = "rejected"
        face_reason = "no_face_detected"
    else:
        best_face = max(faces, key=lambda f: f.get("confidence", 0))
        confidence = float(best_face.get("confidence", 0.0))

        if confidence < 0.90:
            face_status = "rejected"
            face_reason = "low_confidence_face"
        else:
            face_status = "accepted"
            face_reason = "clear_face_detected"

    quality = round(
        (confidence * 70)
        + min(blur / 100, 1) * 20
        + (1 - abs(brightness - 128) / 128) * 10,
        2,
    )

    return {
        "face_status": face_status,
        "face_reason": face_reason,
        "face_count": len(faces),
        "detection_confidence": round(confidence, 4),
        "blur_score": round(blur, 2),
        "brightness_score": round(brightness, 2),
        "quality_score": quality,
    }


def parse_by_pattern(filename, pattern, age_group_index, gender_group_index=None):
    """
    Uses regex groups from filename.

    Example pattern for UTKFace:
    ^(?P<age>\\d+)_(?P<gender>\\d+)_(?P<race>\\d+)_.*$

    Example pattern for FG-Net:
    .*A(?P<age>\\d+).*
    """

    match = re.match(pattern, filename)

    if not match:
        return None

    groups = match.groupdict()

    age = int(groups["age"]) if "age" in groups and groups["age"] is not None else None

    gender = None
    if "gender" in groups and groups["gender"] is not None:
        gender = int(groups["gender"])

    race = None
    if "race" in groups and groups["race"] is not None:
        race = int(groups["race"])

    age_group = age_to_group(age) if age is not None else -1

    final_label = None
    if age_group >= 0 and gender is not None:
        final_label = (age_group * 2) + gender

    return {
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "race": race,
        "final_label": final_label,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True, help="Folder containing dataset images")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--dataset-name", default="custom_dataset")

    parser.add_argument(
        "--pattern",
        required=True,
        help="Regex pattern with named groups, e.g. ^(?P<age>\\d+)_(?P<gender>\\d+)_(?P<race>\\d+)_.*$",
    )

    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include images rejected by face detection in CSV",
    )

    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    detector = MTCNN()

    image_paths = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Dataset: {args.dataset_name}")
    print(f"Source: {source}")
    print(f"Images found: {len(image_paths)}")

    rows = []

    for i, image_path in enumerate(image_paths, start=1):
        parsed = parse_by_pattern(
            filename=image_path.name,
            pattern=args.pattern,
            age_group_index=None,
            gender_group_index=None,
        )

        if parsed is None:
            rows.append({
                "dataset": args.dataset_name,
                "full_path": str(image_path),
                "filename": image_path.name,
                "parse_status": "failed",
                "parse_reason": "filename_did_not_match_pattern",
            })
            continue

        quality = detect_face_quality(image_path, detector)

        if quality["face_status"] == "rejected" and not args.include_rejected:
            pass
        else:
            rows.append({
                "dataset": args.dataset_name,
                "full_path": str(image_path),
                "filename": image_path.name,
                "parse_status": "accepted",
                "parse_reason": "filename_parsed",
                **parsed,
                **quality,
            })

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print("\nCSV created")
    print(f"Saved to: {output}")
    print(f"Rows: {len(df)}")

    if len(df) > 0:
        print("\nColumns:")
        print(list(df.columns))

        if "final_label" in df.columns:
            print("\nFinal label distribution:")
            print(df["final_label"].value_counts(dropna=False).sort_index())

        if "quality_score" in df.columns:
            print("\nQuality summary:")
            print(df["quality_score"].describe())


if __name__ == "__main__":
    main()