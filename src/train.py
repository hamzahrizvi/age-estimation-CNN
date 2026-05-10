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
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)


def make_generators(csv_path, image_size, batch_size):
    df = pd.read_csv(csv_path)

    df["final_label"] = pd.to_numeric(df["final_label"], errors="coerce")
    df = df.dropna(subset=["final_label"])
    df["final_label"] = df["final_label"].astype(int)

    df = df[(df["final_label"] >= 0) & (df["final_label"] <= 17)]

    df["final_label"] = df["final_label"].astype(str)

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

    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", choices=["vgg16", "efficientnet"], default="vgg16")
    parser.add_argument("--output", default="models/best_model.keras")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    args = parser.parse_args()

    setup_gpu()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    df, train_gen, valid_gen = make_generators(
        csv_path=args.csv,
        image_size=args.image_size,
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

    callbacks = [
        ModelCheckpoint(args.output, monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        CSVLogger("outputs/training_log.csv"),
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