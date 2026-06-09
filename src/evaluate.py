import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

<<<<<<< Updated upstream
from sklearn.metrics import classification_report, confusion_matrix
=======
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


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
>>>>>>> Stashed changes


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


<<<<<<< Updated upstream
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
=======
def age_to_fine_group(age):
    age = int(age)

    if age <= 13:
        return 0
    elif age <= 17:
        return 1
    elif age <= 24:
        return 2
    elif age <= 33:
        return 3
    elif age <= 48:
        return 4
    elif age <= 64:
        return 5
    else:
        return 6


def fine_to_coarse_group(fine_group):
    fine_group = int(fine_group)

    if fine_group == 0:
        return 0
    elif fine_group in [1, 2, 3, 4]:
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
>>>>>>> Stashed changes
    df = pd.read_csv(csv_path)

    print("CSV columns:")
    print(df.columns.tolist())

<<<<<<< Updated upstream
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
=======
    if "full_path" not in df.columns:
        raise ValueError("CSV must contain full_path column.")

    if "age" not in df.columns:
        raise ValueError("CSV must contain age column for age regression evaluation.")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "fine_age_group" not in df.columns:
        if "age_group" in df.columns:
            df["fine_age_group"] = pd.to_numeric(df["age_group"], errors="coerce")
        else:
            df["fine_age_group"] = df["age"].apply(
                lambda x: age_to_fine_group(x) if not pd.isna(x) else np.nan
            )

    if "coarse_age_group" not in df.columns:
        df["coarse_age_group"] = df["fine_age_group"].apply(
            lambda x: fine_to_coarse_group(x) if not pd.isna(x) else np.nan
        )

    if "gender" not in df.columns:
        print("No gender column found. Gender metrics will be skipped.")
        df["gender"] = np.nan
    else:
        df["gender"] = df["gender"].apply(clean_gender)

    df["fine_age_group"] = pd.to_numeric(df["fine_age_group"], errors="coerce")
    df["coarse_age_group"] = pd.to_numeric(df["coarse_age_group"], errors="coerce")

    df = df.dropna(
        subset=[
            "full_path",
            "age",
            "fine_age_group",
            "coarse_age_group",
        ]
    )

    df["age"] = df["age"].astype(float)
    df["fine_age_group"] = df["fine_age_group"].astype(int)
    df["coarse_age_group"] = df["coarse_age_group"].astype(int)

    df = df[df["age"].between(0, 100)]
    df = df[df["fine_age_group"].between(0, NUM_FINE_CLASSES - 1)]
    df = df[df["coarse_age_group"].between(0, 2)]

    before_paths = len(df)
    df = df[df["full_path"].apply(lambda x: os.path.exists(str(x)))]
    print(f"Valid image paths: {len(df)} / {before_paths}")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(
            n=max_samples,
            random_state=42
        ).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows remain after cleaning.")

    print("\nFine age distribution:")
    print(df["fine_age_group"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(df["coarse_age_group"].value_counts().sort_index())

    if df["gender"].notna().any():
        print("\nGender distribution:")
        print(df["gender"].value_counts().sort_index())

    print("\nAge summary:")
    print(df["age"].describe())

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
>>>>>>> Stashed changes
        )

    print("\nFG-Net age-group distribution:")
    print(df["age_group"].value_counts().sort_index())

<<<<<<< Updated upstream
    return df

=======
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    return generator
=======
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

    age_branch = tf.keras.layers.Dense(256, activation="relu")(shared)
    age_branch = tf.keras.layers.Dropout(0.35)(age_branch)
    age_output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="age_output"
    )(age_branch)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            gender_output,
            coarse_age_output,
            fine_age_output,
            age_output,
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
>>>>>>> Stashed changes


def prediction_age_to_group(age):
    return age_to_fine_group(age)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
<<<<<<< Updated upstream
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-root", default="data/cleaned_faces/fgnet")
    parser.add_argument("--output-dir", default="outputs")
=======
    parser.add_argument("--weights", required=True)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="outputs/evaluation_age_regression")
    parser.add_argument("--max-samples", type=int, default=None)
>>>>>>> Stashed changes

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    enable_gpu_memory_growth()

    df = prepare_dataframe(args.csv, args.image_root)

    generator = make_generator(
        df,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

<<<<<<< Updated upstream
    model = tf.keras.models.load_model(args.model)
=======
    model = build_model_structure(image_size=args.image_size)
>>>>>>> Stashed changes

    print("\nGenerating predictions on FG-Net...")
    probs = model.predict(generator)

    pred_18class = np.argmax(probs, axis=1)
    pred_age_group = pred_18class // 2
    true_age_group = df["age_group"].values

<<<<<<< Updated upstream
    age_group_accuracy = np.mean(pred_age_group == true_age_group)
    near_accuracy = np.mean(np.abs(pred_age_group - true_age_group) <= 1)
=======
    gender_probs = preds[0]
    coarse_probs = preds[1]
    fine_probs = preds[2]
    age_scaled_pred = preds[3].reshape(-1)
>>>>>>> Stashed changes

    print("\n========== FG-Net RESULTS ==========")
    print(f"Age-group accuracy: {age_group_accuracy:.4f}")
    print(f"Near age-group accuracy (+/-1 group): {near_accuracy:.4f}")

<<<<<<< Updated upstream
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
=======
    pred_age = np.clip(age_scaled_pred * 100.0, 0, 100)

    true_age = df["age"].values.astype(float)
    true_fine = df["fine_age_group"].values.astype(int)
    true_coarse = df["coarse_age_group"].values.astype(int)

    pred_fine_from_age = np.array([
        prediction_age_to_group(age)
        for age in pred_age
    ])

    fine_exact_accuracy = accuracy_score(true_fine, pred_fine)
    fine_near_accuracy = np.mean(np.abs(true_fine - pred_fine) <= 1)

    fine_from_age_accuracy = accuracy_score(true_fine, pred_fine_from_age)
    fine_from_age_near_accuracy = np.mean(
        np.abs(true_fine - pred_fine_from_age) <= 1
    )

    coarse_accuracy = accuracy_score(true_coarse, pred_coarse)

    pred_coarse_from_age = np.array([
        fine_to_coarse_group(group)
        for group in pred_fine_from_age
    ])

    coarse_from_age_accuracy = accuracy_score(
        true_coarse,
        pred_coarse_from_age
    )

    age_mae = mean_absolute_error(true_age, pred_age)

    print("\n========== HIERARCHICAL + AGE REGRESSION EVALUATION ==========")
    print(f"Fine age exact accuracy: {fine_exact_accuracy:.4f}")
    print(f"Fine age near-group accuracy (+/-1): {fine_near_accuracy:.4f}")
    print(f"Fine from predicted-age accuracy: {fine_from_age_accuracy:.4f}")
    print(f"Fine from predicted-age near-group accuracy (+/-1): {fine_from_age_near_accuracy:.4f}")
    print(f"Coarse age accuracy: {coarse_accuracy:.4f}")
    print(f"Coarse from predicted-age accuracy: {coarse_from_age_accuracy:.4f}")
    print(f"Age MAE: {age_mae:.2f} years")

    results = pd.DataFrame({
        "full_path": df["full_path"].values,
        "true_age": true_age,
        "pred_age": pred_age,
        "age_error": np.abs(true_age - pred_age),
        "true_fine_age_group": true_fine,
        "pred_fine_age_group": pred_fine,
        "pred_fine_from_age": pred_fine_from_age,
        "true_coarse_age_group": true_coarse,
        "pred_coarse_age_group": pred_coarse,
        "pred_coarse_from_age": pred_coarse_from_age,
        "fine_correct": true_fine == pred_fine,
        "fine_near_correct": np.abs(true_fine - pred_fine) <= 1,
        "fine_from_age_correct": true_fine == pred_fine_from_age,
        "fine_from_age_near_correct": np.abs(true_fine - pred_fine_from_age) <= 1,
        "coarse_correct": true_coarse == pred_coarse,
        "coarse_from_age_correct": true_coarse == pred_coarse_from_age,
        "fine_confidence": np.max(fine_probs, axis=1),
        "coarse_confidence": np.max(coarse_probs, axis=1),
    })

    if df["gender"].notna().any():
        gender_df = df.dropna(subset=["gender"]).copy()
        gender_indices = gender_df.index.values

        true_gender = gender_df["gender"].values.astype(int)
        pred_gender_valid = pred_gender[gender_indices]

        gender_accuracy = accuracy_score(
            true_gender,
            pred_gender_valid
        )

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
        labels=list(range(NUM_FINE_CLASSES)),
        target_names=[FINE_AGE_LABELS[i] for i in range(NUM_FINE_CLASSES)],
        zero_division=0
    )

    fine_from_age_report = classification_report(
        true_fine,
        pred_fine_from_age,
        labels=list(range(NUM_FINE_CLASSES)),
        target_names=[FINE_AGE_LABELS[i] for i in range(NUM_FINE_CLASSES)],
        zero_division=0
    )

    coarse_report = classification_report(
        true_coarse,
        pred_coarse,
        labels=list(range(3)),
        target_names=[COARSE_AGE_LABELS[i] for i in range(3)],
        zero_division=0
    )
>>>>>>> Stashed changes

    print("\nConfusion matrix:")
    print(cm)

<<<<<<< Updated upstream
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
=======
    fine_from_age_cm = confusion_matrix(
        true_fine,
        pred_fine_from_age,
        labels=list(range(NUM_FINE_CLASSES))
    )

    coarse_cm = confusion_matrix(
        true_coarse,
        pred_coarse,
        labels=list(range(3))
    )

    print("\nFine age classification report:")
    print(fine_report)

    print("\nFine from predicted-age classification report:")
    print(fine_from_age_report)

    print("\nCoarse age classification report:")
    print(coarse_report)

    with open(os.path.join(args.output_dir, "fine_age_report.txt"), "w") as f:
        f.write(fine_report)

    with open(os.path.join(args.output_dir, "fine_from_predicted_age_report.txt"), "w") as f:
        f.write(fine_from_age_report)

    with open(os.path.join(args.output_dir, "coarse_age_report.txt"), "w") as f:
        f.write(coarse_report)

    save_confusion_matrix(
        fine_cm,
        [FINE_AGE_LABELS[i] for i in range(NUM_FINE_CLASSES)],
        os.path.join(args.output_dir, "fine_age_confusion_matrix.csv")
    )

    save_confusion_matrix(
        fine_from_age_cm,
        [FINE_AGE_LABELS[i] for i in range(NUM_FINE_CLASSES)],
        os.path.join(args.output_dir, "fine_from_predicted_age_confusion_matrix.csv")
    )

    save_confusion_matrix(
        coarse_cm,
        [COARSE_AGE_LABELS[i] for i in range(3)],
        os.path.join(args.output_dir, "coarse_age_confusion_matrix.csv")
    )

    per_class_rows = []

    for class_id in range(NUM_FINE_CLASSES):
        mask = true_fine == class_id

        if np.sum(mask) == 0:
            exact_acc = np.nan
            near_acc = np.nan
            from_age_acc = np.nan
            from_age_near_acc = np.nan
            support = 0
            mae = np.nan
        else:
            exact_acc = np.mean(pred_fine[mask] == true_fine[mask])
            near_acc = np.mean(np.abs(pred_fine[mask] - true_fine[mask]) <= 1)
            from_age_acc = np.mean(pred_fine_from_age[mask] == true_fine[mask])
            from_age_near_acc = np.mean(
                np.abs(pred_fine_from_age[mask] - true_fine[mask]) <= 1
            )
            support = int(np.sum(mask))
            mae = mean_absolute_error(true_age[mask], pred_age[mask])

        per_class_rows.append({
            "fine_age_group": class_id,
            "label": FINE_AGE_LABELS[class_id],
            "support": support,
            "classification_exact_accuracy": exact_acc,
            "classification_near_accuracy": near_acc,
            "predicted_age_group_accuracy": from_age_acc,
            "predicted_age_group_near_accuracy": from_age_near_acc,
            "age_mae": mae,
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
        "fine_from_predicted_age_accuracy": fine_from_age_accuracy,
        "fine_from_predicted_age_near_accuracy": fine_from_age_near_accuracy,
        "coarse_accuracy": coarse_accuracy,
        "coarse_from_predicted_age_accuracy": coarse_from_age_accuracy,
        "age_mae_years": age_mae,
    }

    if df["gender"].notna().any():
        summary["gender_accuracy"] = gender_accuracy

    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, "summary_metrics.csv"),
        index=False
    )

    print(f"\nSaved evaluation outputs to: {args.output_dir}")
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()