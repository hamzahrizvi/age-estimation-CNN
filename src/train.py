import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


IMG_SIZE = 224
NUM_FINE_CLASSES = 7


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)

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

    print("\nFine age distribution:")
    print(df["fine_age_group"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(df["coarse_age_group"].value_counts().sort_index())

    print("\nGender distribution:")
    print(df["gender"].value_counts().sort_index())

    return df.reset_index(drop=True)


def compute_sample_weights(df):
    fine_classes = np.arange(NUM_FINE_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=fine_classes,
        y=df["fine_age_group"].values
    )

    fine_weight_map = {
        int(cls): float(weight)
        for cls, weight in zip(fine_classes, weights)
    }

    print("\nFine age class weights:")
    print(fine_weight_map)

    gender_weights = np.ones(len(df), dtype=np.float32)
    coarse_weights = np.ones(len(df), dtype=np.float32)

    fine_weights = df["fine_age_group"].map(fine_weight_map).values.astype(np.float32)

    return {
        "gender_output": gender_weights,
        "coarse_age_output": coarse_weights,
        "fine_age_output": fine_weights,
    }


def make_dataset(df, batch_size, image_size, training, use_sample_weights=False):
    paths = df["full_path"].astype(str).values

    gender = tf.keras.utils.to_categorical(
        df["gender"].values,
        num_classes=2
    )

    coarse = tf.keras.utils.to_categorical(
        df["coarse_age_group"].values,
        num_classes=3
    )

    fine = tf.keras.utils.to_categorical(
        df["fine_age_group"].values,
        num_classes=NUM_FINE_CLASSES
    )

    labels = {
        "gender_output": gender,
        "coarse_age_output": coarse,
        "fine_age_output": fine,
    }

    if use_sample_weights:
        sample_weights = compute_sample_weights(df)

        ds = tf.data.Dataset.from_tensor_slices(
            (
                paths,
                labels,
                sample_weights,
            )
        )

        def load_image(path, labels, sample_weights):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(
                image,
                channels=3,
                expand_animations=False
            )
            image = tf.image.resize(image, (image_size, image_size))
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.efficientnet.preprocess_input(image)

            return image, labels, sample_weights

    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (
                paths,
                labels,
            )
        )

        def load_image(path, labels):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(
                image,
                channels=3,
                expand_animations=False
            )
            image = tf.image.resize(image, (image_size, image_size))
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.efficientnet.preprocess_input(image)

            return image, labels

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=4096)

        if use_sample_weights:
            def augment(image, labels, sample_weights):
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=0.08)
                image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

                return image, labels, sample_weights
        else:
            def augment(image, labels):
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, max_delta=0.08)
                image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

                return image, labels

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_model(image_size=224, learning_rate=1e-4):
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

    compile_model(model, learning_rate)

    return model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "gender_output": "categorical_crossentropy",
            "coarse_age_output": "categorical_crossentropy",
            "fine_age_output": "categorical_crossentropy",
        },
        loss_weights={
            "gender_output": 0.4,
            "coarse_age_output": 0.7,
            "fine_age_output": 1.5,
        },
        metrics={
            "gender_output": ["accuracy"],
            "coarse_age_output": ["accuracy"],
            "fine_age_output": ["accuracy"],
        }
    )


def unfreeze_top_efficientnet(model, learning_rate=3e-6, unfreeze_last_n=35):
    print("\nStarting EfficientNet fine-tuning phase...")
    print(f"Unfreezing only last {unfreeze_last_n} non-BatchNorm EfficientNet layers.")

    for layer in model.layers:
        layer.trainable = False

    head_keywords = [
        "global_average_pooling",
        "batch_normalization",
        "dropout",
        "dense",
        "gender_output",
        "coarse_age_output",
        "fine_age_output",
    ]

    for layer in model.layers:
        if any(keyword in layer.name for keyword in head_keywords):
            layer.trainable = True

    backbone_layers = []

    for layer in model.layers:
        name = layer.name

        is_backbone_layer = (
            name.startswith("stem_")
            or name.startswith("block")
            or name.startswith("top_")
            or name in ["rescaling", "normalization", "rescaling_1"]
        )

        if is_backbone_layer:
            backbone_layers.append(layer)

    if len(backbone_layers) == 0:
        raise ValueError("Could not find EfficientNet backbone layers by name.")

    selected = []

    for layer in reversed(backbone_layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue

        selected.append(layer)

        if len(selected) >= unfreeze_last_n:
            break

    selected = list(reversed(selected))

    for layer in selected:
        layer.trainable = True

    print("\nTrainable backbone layers:")
    for layer in selected:
        print("  " + layer.name)

    print("\nTrainable head layers:")
    for layer in model.layers:
        if layer.trainable and layer not in selected:
            print("  " + layer.name)

    compile_model(model, learning_rate)

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--model-in", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--initial-epoch", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--use-fine-class-weights", action="store_true")

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    enable_gpu_memory_growth()

    df = load_dataframe(args.csv)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["fine_age_group"]
    )

    print("\nTrain rows:", len(train_df))
    print("Validation rows:", len(val_df))

    train_ds = make_dataset(
        train_df,
        batch_size=args.batch_size,
        image_size=IMG_SIZE,
        training=True,
        use_sample_weights=args.use_fine_class_weights
    )

    val_ds = make_dataset(
        val_df,
        batch_size=args.batch_size,
        image_size=IMG_SIZE,
        training=False,
        use_sample_weights=False
    )

    if args.model_in:
        print(f"Loading full model: {args.model_in}")
        model = tf.keras.models.load_model(args.model_in)
    else:
        print("Building new EfficientNet hierarchical model...")
        model = build_model(
            image_size=IMG_SIZE,
            learning_rate=args.learning_rate
        )

    if args.resume_from:
        print(f"Loading weights from: {args.resume_from}")
        model.load_weights(args.resume_from)

    if args.fine_tune:
        model = unfreeze_top_efficientnet(
            model,
            learning_rate=args.learning_rate
        )
    else:
        compile_model(model, args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output,
            monitor="val_fine_age_output_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_fine_age_output_accuracy",
            patience=7,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_fine_age_output_accuracy",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            "outputs/hierarchical_weighted_training_log.csv",
            append=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        callbacks=callbacks
    )

    model.save_weights(args.output)
    print(f"\nSaved model weights to: {args.output}")


if __name__ == "__main__":
    main()