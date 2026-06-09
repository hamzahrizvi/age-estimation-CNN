from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from model import build_vgg16_model, build_efficientnet_model


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
<<<<<<< Updated upstream
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
=======

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    else:
        print("No GPU found. Training will use CPU.")
>>>>>>> Stashed changes


def make_generators(csv_path, image_size, batch_size):
    df = pd.read_csv(csv_path)

<<<<<<< Updated upstream
    df["final_label"] = pd.to_numeric(df["final_label"], errors="coerce")
    df = df.dropna(subset=["final_label"])
    df["final_label"] = df["final_label"].astype(int)
=======
    required = [
        "full_path",
        "age",
        "gender",
        "coarse_age_group",
        "fine_age_group",
    ]
>>>>>>> Stashed changes

    df = df[(df["final_label"] >= 0) & (df["final_label"] <= 17)]

    df["final_label"] = df["final_label"].astype(str)

<<<<<<< Updated upstream
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.25,
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
)

    valid_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.25,
)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="full_path",
        y_col="final_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
=======
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
    df["coarse_age_group"] = pd.to_numeric(df["coarse_age_group"], errors="coerce")
    df["fine_age_group"] = pd.to_numeric(df["fine_age_group"], errors="coerce")

    df = df.dropna(subset=[
        "age",
        "gender",
        "coarse_age_group",
        "fine_age_group",
    ])

    df["age"] = df["age"].astype(float)
    df["gender"] = df["gender"].astype(int)
    df["coarse_age_group"] = df["coarse_age_group"].astype(int)
    df["fine_age_group"] = df["fine_age_group"].astype(int)

    df = df[df["age"].between(0, 100)]
    df = df[df["gender"].between(0, 1)]
    df = df[df["coarse_age_group"].between(0, 2)]
    df = df[df["fine_age_group"].between(0, NUM_FINE_CLASSES - 1)]

    df["age_scaled"] = df["age"] / 100.0

    before = len(df)
    df = df[df["full_path"].apply(lambda x: os.path.exists(str(x)))]
    print(f"Valid image paths: {len(df)} / {before}")

    print("\nFine age distribution:")
    print(df["fine_age_group"].value_counts().sort_index())

    print("\nCoarse age distribution:")
    print(df["coarse_age_group"].value_counts().sort_index())

    print("\nGender distribution:")
    print(df["gender"].value_counts().sort_index())

    print("\nAge summary:")
    print(df["age"].describe())

    return df.reset_index(drop=True)


def balance_dataframe(df, balance_col=None, max_per_class=None):
    if balance_col is None or max_per_class is None:
        return df

    if balance_col not in df.columns:
        raise ValueError(f"Balance column not found: {balance_col}")

    print(f"\nBalancing dataset by: {balance_col}")
    print(f"Max samples per class: {max_per_class}")

    print("\nBefore balancing:")
    print(df[balance_col].value_counts().sort_index())

    balanced_parts = []

    for class_value, class_df in df.groupby(balance_col):
        if len(class_df) > max_per_class:
            class_df = class_df.sample(
                n=max_per_class,
                random_state=42
            )

        balanced_parts.append(class_df)

    balanced_df = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nAfter balancing:")
    print(balanced_df[balance_col].value_counts().sort_index())

    print(f"\nRows before balancing: {len(df)}")
    print(f"Rows after balancing: {len(balanced_df)}")

    return balanced_df


def make_dataset(df, batch_size, image_size, training):
    paths = df["full_path"].astype(str).values

    gender = tf.keras.utils.to_categorical(
        df["gender"].values,
        num_classes=2
>>>>>>> Stashed changes
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="full_path",
        y_col="final_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return df, train_generator, valid_generator


def get_class_weights(df):
    labels = pd.to_numeric(df["final_label"], errors="coerce")
    labels = labels.dropna().astype(int).values

    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

<<<<<<< Updated upstream
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}
=======
    age_scaled = df["age_scaled"].values.astype("float32")

    labels = {
        "gender_output": gender,
        "coarse_age_output": coarse,
        "fine_age_output": fine,
        "age_output": age_scaled,
    }

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, labels):
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

        return image, labels

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        ds = ds.shuffle(buffer_size=4096)

        def augment(image, labels):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            return image, labels

        ds = ds.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "gender_output": "categorical_crossentropy",
            "coarse_age_output": "categorical_crossentropy",
            "fine_age_output": "categorical_crossentropy",
            "age_output": "mae",
        },
        loss_weights={
            "gender_output": 0.4,
            "coarse_age_output": 0.7,
            "fine_age_output": 1.0,
            "age_output": 0.8,
        },
        metrics={
            "gender_output": ["accuracy"],
            "coarse_age_output": ["accuracy"],
            "fine_age_output": ["accuracy"],
            "age_output": ["mae"],
        }
    )


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

    compile_model(model, learning_rate)

    return model


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
        "age_output",
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
>>>>>>> Stashed changes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", choices=["vgg16", "efficientnet"], default="vgg16")
    parser.add_argument("--output", default="models/best_model.keras")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

<<<<<<< Updated upstream
=======
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--balance", default=None, choices=["fine_age_group", "coarse_age_group", "gender"])
    parser.add_argument("--max-per-class", type=int, default=None)

>>>>>>> Stashed changes
    args = parser.parse_args()

    setup_gpu()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

<<<<<<< Updated upstream
    df, train_gen, valid_gen = make_generators(
        csv_path=args.csv,
        image_size=args.image_size,
=======
    df = load_dataframe(args.csv)

    df = balance_dataframe(
        df,
        balance_col=args.balance,
        max_per_class=args.max_per_class
    )

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
>>>>>>> Stashed changes
        batch_size=args.batch_size,
    )

    class_weights = get_class_weights(df)

    if args.model == "vgg16":
        model = build_vgg16_model(num_classes=18, image_size=args.image_size)
    else:
        model = build_efficientnet_model(num_classes=18, image_size=args.image_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

<<<<<<< Updated upstream
    callbacks = [
        ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        CSVLogger("outputs/training_log.csv"),
=======
    if args.model_in:
        print(f"Loading full model: {args.model_in}")
        model = tf.keras.models.load_model(args.model_in)
    else:
        print("Building new EfficientNet hierarchical age-regression model...")
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
            "outputs/hierarchical_age_regression_training_log.csv",
            append=True
        )
>>>>>>> Stashed changes
    ]

    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
#        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()