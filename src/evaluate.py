import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


FINE_AGE_LABELS = {
    0: "0-2",
    1: "3-5",
    2: "6-13",
    3: "14-18",
    4: "19-24",
    5: "25-33",
    6: "34-48",
    7: "49-64",
    8: "65+",
}

COARSE_AGE_LABELS = {
    0: "0-13",
    1: "14-48",
    2: "49+",
}

GENDER_LABELS = {
    0: "male",
    1: "female",
}


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")


def age_to_fine_group(age):
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


def fine_to_coarse_group(fine_group):
    fine_group = int(fine_group)

    if fine_group <= 2:
        return 0
    elif fine_group <= 6:
        return 1
    else:
        return 2


def clean_gender(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if value in ["0", "0.0", "m", "male"]:
        return 0

    if value in ["1", "1.0", "f", "female"]:
        return 1

    return np.nan


def prepare_dataframe(csv_path, max_samples=None):
    df = pd.read_csv(csv_path)

    print("CSV columns:")
    print(list(df.columns))

    if "full_path" not in df.columns:
        raise ValueError("CSV must contain full_path column.")

    if "fine_age_group" not in df.columns:
        if "age_group" in df.columns:
            df["fine_age_group"] = pd.to_numeric(df["age_group"], errors="coerce")
        elif "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df = df.dropna(subset=["age"])
            df["fine_age_group"] = df["age"].apply(age_to_fine_group)
        else:
            raise ValueError("CSV must contain fine_age_group, age_group, or age.")

    if "coarse_age_group" not in df.columns:
        df["coarse_age_group"] = df["fine_age_group"].apply(fine_to_coarse_group)

    if "gender" not in df.columns:
        print("No gender column found. Gender metrics will be skipped.")
        df["gender"] = np.nan
    else:
        df["gender"] = df["gender"].apply(clean_gender)

    df["fine_age_group"] = pd.to_numeric(df["fine_age_group"], errors="coerce")
    df["coarse_age_group"] = pd.to_numeric(df["coarse_age_group"], errors="coerce")

    df = df.dropna(subset=["full_path", "fine_age_group", "coarse_age_group"])

    df["fine_age_group"] = df["fine_age_group"].astype(int)
    df["coarse_age_group"] = df["coarse_age_group"].astype(int)

    df = df[df["fine_age_group"].between(0, 8)]
    df = df[df["coarse_age_group"].between(0, 2)]

    before_paths = len(df)
    df = df[df["full_path"].apply(lambda x: os.path.exists(str(x)))]
    print(f"Valid image paths: {len(df)} / {before_paths}")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows remain after cleaning.")

    print("\nFine age distribution:")
    print(df["fine_age_group"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(df["coarse_age_group"].value_counts().sort_index())

    if df["gender"].notna().any():
        print("\nGender distribution:")
        print(df["gender"].value_counts().sort_index())

    return df.reset_index(drop=True)


def make_dataset(df, batch_size=16, image_size=224):
    paths = df["full_path"].astype(str).values

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_model_structure(image_size=224):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    shared = tf.keras.layers.Dense(512, activation="relu")(x)
    shared = tf.keras.layers.BatchNormalization()(shared)
    shared = tf.keras.layers.Dropout(0.4)(shared)

    gender_branch = tf.keras.layers.Dense(128, activation="relu")(shared)
    gender_branch = tf.keras.layers.Dropout(0.25)(gender_branch)
    gender_output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="gender_output"
    )(gender_branch)

    coarse_branch = tf.keras.layers.Dense(128, activation="relu")(shared)
    coarse_branch = tf.keras.layers.Dropout(0.25)(coarse_branch)
    coarse_age_output = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="coarse_age_output"
    )(coarse_branch)

    fine_branch = tf.keras.layers.Dense(256, activation="relu")(shared)
    fine_branch = tf.keras.layers.Dropout(0.35)(fine_branch)
    fine_age_output = tf.keras.layers.Dense(
        9,
        activation="softmax",
        name="fine_age_output"
    )(fine_branch)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            gender_output,
            coarse_age_output,
            fine_age_output,
        ]
    )

    return model


def save_confusion_matrix(cm, labels, path):
    df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    df.to_csv(path, index=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    enable_gpu_memory_growth()

    df = prepare_dataframe(
        csv_path=args.csv,
        max_samples=args.max_samples
    )

    ds = make_dataset(
        df,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    model = build_model_structure(image_size=args.image_size)

    print(f"\nLoading weights from: {args.weights}")
    model.load_weights(args.weights)

    print("\nGenerating predictions...")
    preds = model.predict(ds)

    gender_probs = preds[0]
    coarse_probs = preds[1]
    fine_probs = preds[2]

    pred_gender = np.argmax(gender_probs, axis=1)
    pred_coarse = np.argmax(coarse_probs, axis=1)
    pred_fine = np.argmax(fine_probs, axis=1)

    true_fine = df["fine_age_group"].values.astype(int)
    true_coarse = df["coarse_age_group"].values.astype(int)

    fine_exact_accuracy = accuracy_score(true_fine, pred_fine)
    fine_near_accuracy = np.mean(np.abs(true_fine - pred_fine) <= 1)

    coarse_accuracy = accuracy_score(true_coarse, pred_coarse)

    print("\n========== HIERARCHICAL EVALUATION ==========")
    print(f"Fine age exact accuracy: {fine_exact_accuracy:.4f}")
    print(f"Fine age near-group accuracy (+/-1): {fine_near_accuracy:.4f}")
    print(f"Coarse age accuracy: {coarse_accuracy:.4f}")

    results = pd.DataFrame({
        "full_path": df["full_path"].values,
        "true_fine_age_group": true_fine,
        "pred_fine_age_group": pred_fine,
        "true_coarse_age_group": true_coarse,
        "pred_coarse_age_group": pred_coarse,
        "fine_correct": true_fine == pred_fine,
        "fine_near_correct": np.abs(true_fine - pred_fine) <= 1,
        "coarse_correct": true_coarse == pred_coarse,
        "fine_confidence": np.max(fine_probs, axis=1),
        "coarse_confidence": np.max(coarse_probs, axis=1),
    })

    if df["gender"].notna().any():
        gender_df = df.dropna(subset=["gender"]).copy()
        gender_indices = gender_df.index.values

        true_gender = gender_df["gender"].values.astype(int)
        pred_gender_valid = pred_gender[gender_indices]

        gender_accuracy = accuracy_score(true_gender, pred_gender_valid)

        print(f"Gender accuracy: {gender_accuracy:.4f}")

        results["true_gender"] = df["gender"].values
        results["pred_gender"] = pred_gender
        results["gender_confidence"] = np.max(gender_probs, axis=1)

        gender_report = classification_report(
            true_gender,
            pred_gender_valid,
            labels=[0, 1],
            target_names=[GENDER_LABELS[0], GENDER_LABELS[1]],
            zero_division=0
        )

        gender_cm = confusion_matrix(
            true_gender,
            pred_gender_valid,
            labels=[0, 1]
        )

        with open(os.path.join(args.output_dir, "gender_report.txt"), "w") as f:
            f.write(gender_report)

        save_confusion_matrix(
            gender_cm,
            [GENDER_LABELS[0], GENDER_LABELS[1]],
            os.path.join(args.output_dir, "gender_confusion_matrix.csv")
        )

    fine_report = classification_report(
        true_fine,
        pred_fine,
        labels=list(range(9)),
        target_names=[FINE_AGE_LABELS[i] for i in range(9)],
        zero_division=0
    )

    coarse_report = classification_report(
        true_coarse,
        pred_coarse,
        labels=list(range(3)),
        target_names=[COARSE_AGE_LABELS[i] for i in range(3)],
        zero_division=0
    )

    fine_cm = confusion_matrix(
        true_fine,
        pred_fine,
        labels=list(range(9))
    )

    coarse_cm = confusion_matrix(
        true_coarse,
        pred_coarse,
        labels=list(range(3))
    )

    print("\nFine age classification report:")
    print(fine_report)

    print("\nCoarse age classification report:")
    print(coarse_report)

    with open(os.path.join(args.output_dir, "fine_age_report.txt"), "w") as f:
        f.write(fine_report)

    with open(os.path.join(args.output_dir, "coarse_age_report.txt"), "w") as f:
        f.write(coarse_report)

    save_confusion_matrix(
        fine_cm,
        [FINE_AGE_LABELS[i] for i in range(9)],
        os.path.join(args.output_dir, "fine_age_confusion_matrix.csv")
    )

    save_confusion_matrix(
        coarse_cm,
        [COARSE_AGE_LABELS[i] for i in range(3)],
        os.path.join(args.output_dir, "coarse_age_confusion_matrix.csv")
    )

    per_class_rows = []

    for class_id in range(9):
        mask = true_fine == class_id

        if np.sum(mask) == 0:
            acc = np.nan
            near_acc = np.nan
            support = 0
        else:
            acc = np.mean(pred_fine[mask] == true_fine[mask])
            near_acc = np.mean(np.abs(pred_fine[mask] - true_fine[mask]) <= 1)
            support = int(np.sum(mask))

        per_class_rows.append({
            "fine_age_group": class_id,
            "label": FINE_AGE_LABELS[class_id],
            "support": support,
            "exact_accuracy": acc,
            "near_accuracy": near_acc,
        })

    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(args.output_dir, "fine_age_per_class_accuracy.csv"),
        index=False
    )

    results.to_csv(
        os.path.join(args.output_dir, "predictions.csv"),
        index=False
    )

    summary = {
        "fine_exact_accuracy": fine_exact_accuracy,
        "fine_near_accuracy": fine_near_accuracy,
        "coarse_accuracy": coarse_accuracy,
    }

    if df["gender"].notna().any():
        summary["gender_accuracy"] = gender_accuracy

    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, "summary_metrics.csv"),
        index=False
    )

    print(f"\nSaved evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()