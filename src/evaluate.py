import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


IMG_SIZE = 224
NUM_FINE_CLASSES = 7


FINE_AGE_LABELS = {
    0: "0-13",
    1: "14-17",
    2: "18-24",
    3: "25-33",
    4: "34-48",
    5: "49-64",
    6: "65+",
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
    else:
        print("No GPU found. Evaluation will use CPU.")


def prepare_dataframe(csv_path, max_samples=None):
    df = pd.read_csv(csv_path)

    print("CSV columns:")
    print(list(df.columns))

    required = [
        "full_path",
        "gender",
        "coarse_age_group",
        "fine_age_group",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required)

    df["gender"] = df["gender"].astype(int)
    df["coarse_age_group"] = df["coarse_age_group"].astype(int)
    df["fine_age_group"] = df["fine_age_group"].astype(int)

    df = df[df["gender"].between(0, 1)]
    df = df[df["coarse_age_group"].between(0, 2)]
    df = df[df["fine_age_group"].between(0, NUM_FINE_CLASSES - 1)]

    before = len(df)
    df = df[df["full_path"].apply(lambda x: os.path.exists(str(x)))]
    print(f"Valid image paths: {len(df)} / {before}")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows remain after cleaning.")

    print("\nFine age distribution:")
    print(df["fine_age_group"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(df["coarse_age_group"].value_counts().sort_index())

    print("\nGender distribution:")
    print(df["gender"].value_counts().sort_index())

    return df.reset_index(drop=True)


def make_dataset(df, batch_size=16, image_size=224):
    paths = df["full_path"].astype(str).values

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def load_image(path):
        image = tf.io.read_file(path)

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image = tf.image.resize(
            image,
            (image_size, image_size)
        )

        image = tf.cast(image, tf.float32)

        image = tf.keras.applications.efficientnet.preprocess_input(image)

        return image

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

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
        NUM_FINE_CLASSES,
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

    model = build_model_structure(
        image_size=args.image_size
    )

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

    true_gender = df["gender"].values.astype(int)
    true_coarse = df["coarse_age_group"].values.astype(int)
    true_fine = df["fine_age_group"].values.astype(int)

    gender_accuracy = accuracy_score(true_gender, pred_gender)
    coarse_accuracy = accuracy_score(true_coarse, pred_coarse)
    fine_exact_accuracy = accuracy_score(true_fine, pred_fine)
    fine_near_accuracy = np.mean(np.abs(true_fine - pred_fine) <= 1)

    print("\n========== HIERARCHICAL 7-CLASS EVALUATION ==========")
    print(f"Fine age exact accuracy: {fine_exact_accuracy:.4f}")
    print(f"Fine age near-group accuracy (+/-1): {fine_near_accuracy:.4f}")
    print(f"Coarse age accuracy: {coarse_accuracy:.4f}")
    print(f"Gender accuracy: {gender_accuracy:.4f}")

    fine_report = classification_report(
        true_fine,
        pred_fine,
        labels=list(range(NUM_FINE_CLASSES)),
        target_names=[FINE_AGE_LABELS[i] for i in range(NUM_FINE_CLASSES)],
        zero_division=0
    )

    coarse_report = classification_report(
        true_coarse,
        pred_coarse,
        labels=[0, 1, 2],
        target_names=[COARSE_AGE_LABELS[i] for i in range(3)],
        zero_division=0
    )

    gender_report = classification_report(
        true_gender,
        pred_gender,
        labels=[0, 1],
        target_names=[GENDER_LABELS[i] for i in range(2)],
        zero_division=0
    )

    fine_cm = confusion_matrix(
        true_fine,
        pred_fine,
        labels=list(range(NUM_FINE_CLASSES))
    )

    coarse_cm = confusion_matrix(
        true_coarse,
        pred_coarse,
        labels=[0, 1, 2]
    )

    gender_cm = confusion_matrix(
        true_gender,
        pred_gender,
        labels=[0, 1]
    )

    print("\nFine age classification report:")
    print(fine_report)

    print("\nCoarse age classification report:")
    print(coarse_report)

    print("\nGender classification report:")
    print(gender_report)

    with open(os.path.join(args.output_dir, "fine_age_report.txt"), "w") as f:
        f.write(fine_report)

    with open(os.path.join(args.output_dir, "coarse_age_report.txt"), "w") as f:
        f.write(coarse_report)

    with open(os.path.join(args.output_dir, "gender_report.txt"), "w") as f:
        f.write(gender_report)

    save_confusion_matrix(
        fine_cm,
        [FINE_AGE_LABELS[i] for i in range(NUM_FINE_CLASSES)],
        os.path.join(args.output_dir, "fine_age_confusion_matrix.csv")
    )

    save_confusion_matrix(
        coarse_cm,
        [COARSE_AGE_LABELS[i] for i in range(3)],
        os.path.join(args.output_dir, "coarse_age_confusion_matrix.csv")
    )

    save_confusion_matrix(
        gender_cm,
        [GENDER_LABELS[i] for i in range(2)],
        os.path.join(args.output_dir, "gender_confusion_matrix.csv")
    )

    results = pd.DataFrame({
        "full_path": df["full_path"].values,
        "true_gender": true_gender,
        "pred_gender": pred_gender,
        "true_coarse_age_group": true_coarse,
        "pred_coarse_age_group": pred_coarse,
        "true_fine_age_group": true_fine,
        "pred_fine_age_group": pred_fine,
        "gender_correct": true_gender == pred_gender,
        "coarse_correct": true_coarse == pred_coarse,
        "fine_correct": true_fine == pred_fine,
        "fine_near_correct": np.abs(true_fine - pred_fine) <= 1,
        "gender_confidence": np.max(gender_probs, axis=1),
        "coarse_confidence": np.max(coarse_probs, axis=1),
        "fine_confidence": np.max(fine_probs, axis=1),
    })

    results.to_csv(
        os.path.join(args.output_dir, "predictions.csv"),
        index=False
    )

    per_class_rows = []

    for class_id in range(NUM_FINE_CLASSES):
        mask = true_fine == class_id

        if np.sum(mask) == 0:
            exact_acc = np.nan
            near_acc = np.nan
            support = 0
        else:
            exact_acc = np.mean(pred_fine[mask] == true_fine[mask])
            near_acc = np.mean(np.abs(pred_fine[mask] - true_fine[mask]) <= 1)
            support = int(np.sum(mask))

        per_class_rows.append({
            "fine_age_group": class_id,
            "label": FINE_AGE_LABELS[class_id],
            "support": support,
            "exact_accuracy": exact_acc,
            "near_accuracy": near_acc,
        })

    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(args.output_dir, "fine_age_per_class_accuracy.csv"),
        index=False
    )

    summary = {
        "fine_exact_accuracy": fine_exact_accuracy,
        "fine_near_accuracy": fine_near_accuracy,
        "coarse_accuracy": coarse_accuracy,
        "gender_accuracy": gender_accuracy,
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, "summary_metrics.csv"),
        index=False
    )

    print(f"\nSaved evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()