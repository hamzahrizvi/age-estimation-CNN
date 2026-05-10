import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")


def age_to_group(age):
    age = int(age)

    if age <= 2:
        return 0
    elif age <= 5:
        return 1
    elif age <= 13:
        return 2
    elif age <= 18:
        return 3
    elif age <= 24:
        return 4
    elif age <= 33:
        return 5
    elif age <= 48:
        return 6
    elif age <= 64:
        return 7
    else:
        return 8


def find_existing_image_path(row, image_root):
    old_path = str(row.get("full_path", "")).strip()

    if old_path and os.path.exists(old_path):
        return old_path

    filename = str(row.get("filename", "")).strip()

    candidates = [
        os.path.join(image_root, filename),
        os.path.join(image_root, filename + ".jpg"),
        os.path.join(image_root, filename + ".JPG"),
        os.path.join(image_root, filename + ".png"),
        os.path.join(image_root, filename + ".jpeg"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def prepare_dataframe(csv_path, image_root):
    df = pd.read_csv(csv_path)

    print("CSV columns:")
    print(df.columns.tolist())

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age", "filename"])

    df["age_group"] = df["age"].apply(age_to_group)

    print("\nResolving image paths from:")
    print(image_root)

    df["resolved_path"] = df.apply(
        lambda row: find_existing_image_path(row, image_root),
        axis=1
    )

    before = len(df)
    df = df.dropna(subset=["resolved_path"])
    print(f"Valid image files found: {len(df)} / {before}")

    if len(df) == 0:
        raise ValueError(
            "No FG-Net images were found. Check that --image-root points to the actual FG-Net image folder."
        )

    print("\nFG-Net age-group distribution:")
    print(df["age_group"].value_counts().sort_index())

    return df


def make_generator(df, batch_size=16, image_size=224):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="resolved_path",
        y_col=None,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    return generator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-root", default="data/cleaned_faces/fgnet")
    parser.add_argument("--output-dir", default="outputs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    enable_gpu_memory_growth()

    df = prepare_dataframe(args.csv, args.image_root)

    generator = make_generator(
        df,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    model = tf.keras.models.load_model(args.model)

    print("\nGenerating predictions on FG-Net...")
    probs = model.predict(generator)

    pred_18class = np.argmax(probs, axis=1)
    pred_age_group = pred_18class // 2
    true_age_group = df["age_group"].values

    age_group_accuracy = np.mean(pred_age_group == true_age_group)
    near_accuracy = np.mean(np.abs(pred_age_group - true_age_group) <= 1)

    print("\n========== FG-Net RESULTS ==========")
    print(f"Age-group accuracy: {age_group_accuracy:.4f}")
    print(f"Near age-group accuracy (+/-1 group): {near_accuracy:.4f}")

    report = classification_report(
        true_age_group,
        pred_age_group,
        labels=list(range(9)),
        target_names=[str(i) for i in range(9)],
        zero_division=0
    )

    cm = confusion_matrix(
        true_age_group,
        pred_age_group,
        labels=list(range(9))
    )

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm)

    with open(os.path.join(args.output_dir, "fgnet_agegroup_report.txt"), "w") as f:
        f.write(report)

    pd.DataFrame(cm).to_csv(
        os.path.join(args.output_dir, "fgnet_agegroup_confusion_matrix.csv"),
        index=False
    )

    results = pd.DataFrame({
        "filename": df["filename"].values,
        "resolved_path": df["resolved_path"].values,
        "true_age_group": true_age_group,
        "predicted_18class": pred_18class,
        "predicted_age_group": pred_age_group,
        "correct": pred_age_group == true_age_group,
        "near_correct": np.abs(pred_age_group - true_age_group) <= 1
    })

    results.to_csv(
        os.path.join(args.output_dir, "fgnet_predictions.csv"),
        index=False
    )

    print("\nSaved outputs to outputs/")


if __name__ == "__main__":
    main()