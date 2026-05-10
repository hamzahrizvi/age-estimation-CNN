from pathlib import Path
import argparse
import shutil
import time
import sys

import cv2
import pandas as pd
from mtcnn import MTCNN


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def timed_continue_prompt(seconds=60):
    print(f"\nPress CTRL+C to cancel. Starting full processing in {seconds} seconds...")
    print("Or press Enter to start now.")

    start = time.time()

    if sys.platform.startswith("win"):
        import msvcrt

        while time.time() - start < seconds:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == "\r":
                    print("Starting now.")
                    return
            time.sleep(0.2)

    else:
        import select

        while time.time() - start < seconds:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                sys.stdin.readline()
                print("Starting now.")
                return

    print("No response. Starting full processing.")


def blur_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def face_area_ratio(face, image_bgr):
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = face.get("box", [0, 0, 0, 0])

    if img_w == 0 or img_h == 0:
        return 0.0

    face_area = max(0, w) * max(0, h)
    image_area = img_w * img_h

    return round(face_area / image_area, 6)


def is_bad_pose(face, max_nose_offset=0.28, max_mouth_offset=0.35):
    keypoints = face.get("keypoints", {})

    required = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
    for point in required:
        if point not in keypoints:
            return True, f"missing_{point}"

    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    nose = keypoints["nose"]
    mouth_left = keypoints["mouth_left"]
    mouth_right = keypoints["mouth_right"]

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_distance = abs(right_eye[0] - left_eye[0])

    if eye_distance <= 1:
        return True, "invalid_eye_distance"

    nose_offset = abs(nose[0] - eye_center_x) / eye_distance
    mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
    mouth_offset = abs(mouth_center_x - eye_center_x) / eye_distance

    if nose_offset > max_nose_offset:
        return True, "side_pose_nose_offset"

    if mouth_offset > max_mouth_offset:
        return True, "side_pose_mouth_offset"

    return False, "pose_ok"


def crop_face(image_bgr, box, image_size=224, margin=0.25):
    x, y, w, h = box
    img_h, img_w = image_bgr.shape[:2]

    x = max(0, x)
    y = max(0, y)

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = image_bgr[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, (image_size, image_size))


def quality_score(confidence, blur, brightness):
    return round(
        (confidence * 65)
        + min(blur / 100, 1) * 20
        + (1 - abs(brightness - 128) / 128) * 15,
        2,
    )


def save_rejected_preview(image_path, preview_root, source_root, reason):
    try:
        safe_reason = reason.replace(":", "_").replace("\\", "_").replace("/", "_")
        relative_path = image_path.relative_to(source_root)
        preview_path = preview_root / safe_reason / relative_path
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, preview_path)
        return str(preview_path)
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True, help="Source folder containing images")
    parser.add_argument("--output", required=True, help="Output folder for cleaned cropped images")
    parser.add_argument("--log", default="outputs/face_cleaning_report.csv")
    parser.add_argument("--dataset-name", default="custom_dataset")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=float, default=0.25)

    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--min-blur", type=float, default=25.0)
    parser.add_argument("--min-brightness", type=float, default=20.0)
    parser.add_argument("--max-brightness", type=float, default=240.0)

    parser.add_argument("--min-face-area-ratio", type=float, default=0.02)
    parser.add_argument("--max-face-area-ratio", type=float, default=0.90)

    parser.add_argument("--max-nose-offset", type=float, default=0.28)
    parser.add_argument("--max-mouth-offset", type=float, default=0.35)

    parser.add_argument("--limit", type=int, default=None, help="Process only first N images")
    parser.add_argument("--dry-run", action="store_true", help="Do not save cropped images, only create report")
    parser.add_argument("--allow-multiple", action="store_true", help="Use largest face if multiple faces are found")
    parser.add_argument("--skip-wait", action="store_true", help="Do not wait 60 seconds before full run")

    parser.add_argument(
        "--rejected-preview-folder",
        default="outputs/rejected_preview",
        help="Folder where rejected original images are copied for review",
    )

    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    log_path = Path(args.log)
    rejected_preview_root = Path(args.rejected_preview_folder) / args.dataset_name

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_preview_root.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = sorted(image_paths)

    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    print("\nFace cleaning configuration")
    print("---------------------------")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Log: {log_path}")
    print(f"Rejected preview folder: {rejected_preview_root}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Limit: {args.limit}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Min blur: {args.min_blur}")
    print(f"Brightness range: {args.min_brightness} - {args.max_brightness}")
    print(f"Face area ratio range: {args.min_face_area_ratio} - {args.max_face_area_ratio}")
    print(f"Max nose offset: {args.max_nose_offset}")
    print(f"Max mouth offset: {args.max_mouth_offset}")
    print(f"Allow multiple faces: {args.allow_multiple}")

    if args.limit is None and not args.dry_run and not args.skip_wait:
        timed_continue_prompt(seconds=60)

    detector = MTCNN()

    rows = []

    for i, image_path in enumerate(image_paths, start=1):
        status = "rejected"
        reason = ""
        output_path = ""
        rejected_preview_path = ""

        face_count = 0
        detection_confidence = 0.0
        blur = 0.0
        brightness = 0.0
        area_ratio = 0.0
        q_score = 0.0

        try:
            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:
                reason = "cannot_read_image"

            else:
                blur = blur_score(image_bgr)
                brightness = brightness_score(image_bgr)

                if blur < args.min_blur:
                    reason = "too_blurry"

                elif brightness < args.min_brightness:
                    reason = "too_dark"

                elif brightness > args.max_brightness:
                    reason = "too_bright"

                else:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(image_rgb)
                    face_count = len(faces)

                    if face_count == 0:
                        reason = "no_face_detected"

                    else:
                        good_faces = [
                            f for f in faces
                            if f.get("confidence", 0.0) >= args.min_confidence
                        ]

                        if len(good_faces) == 0:
                            best_conf = max([f.get("confidence", 0.0) for f in faces])
                            detection_confidence = round(float(best_conf), 4)
                            reason = "low_confidence_face"

                        elif len(good_faces) > 1 and not args.allow_multiple:
                            detection_confidence = round(
                                float(max([f.get("confidence", 0.0) for f in good_faces])),
                                4,
                            )
                            reason = "multiple_faces"

                        else:
                            selected_face = max(
                                good_faces,
                                key=lambda f: f["box"][2] * f["box"][3],
                            )

                            detection_confidence = round(float(selected_face.get("confidence", 0.0)), 4)
                            area_ratio = face_area_ratio(selected_face, image_bgr)

                            if area_ratio < args.min_face_area_ratio:
                                reason = "face_too_small"

                            elif area_ratio > args.max_face_area_ratio:
                                reason = "face_too_large"

                            else:
                                bad_pose, pose_reason = is_bad_pose(
                                    selected_face,
                                    max_nose_offset=args.max_nose_offset,
                                    max_mouth_offset=args.max_mouth_offset,
                                )

                                if bad_pose:
                                    reason = pose_reason

                                else:
                                    cropped = crop_face(
                                        image_bgr,
                                        selected_face["box"],
                                        image_size=args.image_size,
                                        margin=args.margin,
                                    )

                                    if cropped is None:
                                        reason = "crop_failed"

                                    else:
                                        relative_path = image_path.relative_to(source)
                                        save_path = output / relative_path
                                        save_path.parent.mkdir(parents=True, exist_ok=True)

                                        if not args.dry_run:
                                            cv2.imwrite(str(save_path), cropped)

                                        status = "accepted"
                                        reason = "clean_face_saved"
                                        output_path = str(save_path)

                q_score = quality_score(detection_confidence, blur, brightness)

        except Exception as e:
            reason = f"error_{str(e)}"

        if status == "rejected":
            rejected_preview_path = save_rejected_preview(
                image_path=image_path,
                preview_root=rejected_preview_root,
                source_root=source,
                reason=reason,
            )

        rows.append({
            "dataset": args.dataset_name,
            "original_path": str(image_path),
            "filename": image_path.name,
            "status": status,
            "reason": reason,
            "face_count": face_count,
            "detection_confidence": detection_confidence,
            "blur_score": round(blur, 2),
            "brightness_score": round(brightness, 2),
            "face_area_ratio": area_ratio,
            "quality_score": q_score,
            "output_path": output_path,
            "rejected_preview_path": rejected_preview_path,
        })

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(log_path, index=False)

    print("\nCleaning complete")
    print("-----------------")
    print(f"Dataset: {args.dataset_name}")
    print(f"Processed: {len(df)}")
    print(f"Accepted: {(df['status'] == 'accepted').sum()}")
    print(f"Rejected: {(df['status'] == 'rejected').sum()}")
    print(f"Output folder: {output}")
    print(f"Rejected preview folder: {rejected_preview_root}")
    print(f"Report CSV: {log_path}")

    print("\nRejection reasons:")
    print(df["reason"].value_counts())

    print("\nQuality summary:")
    print(df["quality_score"].describe())


if __name__ == "__main__":
    main()