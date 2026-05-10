import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


NUM_CLASSES = 18


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")


def clean_labels(df):
    df = df.copy()

    df["final_label"] = pd.to_numeric(df["final_label"], errors="coerce")
    df = df.dropna(subset=["final_label"])

    df["final_label"] = df["final_label"].astype(int)
    df = df[df["final_label"].between(0, NUM_CLASSES - 1)]

    df["final_label"] = df["final_label"].astype(str)

    return df


def make_generators(csv_path, batch_size=16, image_size=224):
    df = pd.read_csv(csv_path)
    df = clean_labels(df)

    print("Final label distribution:")
    print(df["final_label"].value_counts().sort_index())

    train_df, val_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["final_label"]
    )

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    class_names = sorted([str(i) for i in range(NUM_CLASSES)])

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="full_path",
        y_col="final_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=class_names,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="full_path",
        y_col="final_label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=class_names,
        shuffle=False
    )

    labels = train_generator.classes
    classes = np.arange(NUM_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(classes, weights)
    }

    print("Class indices:")
    print(train_generator.class_indices)

    return train_generator, val_generator, class_weights


def fine_tune_setup(model):
    """
    Correct fine-tuning:
    - freeze most VGG16 layers
    - unfreeze VGG16 block5
    - keep classifier head trainable
    """

    model.trainable = True

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True

            for sublayer in layer.layers:
                sublayer.trainable = False

            for sublayer in layer.layers:
                if "block5" in sublayer.name:
                    sublayer.trainable = True

        else:
            # Keep classifier head trainable
            layer.trainable = True

    print("\nTrainable layers:")
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            print(layer.name)
            for sublayer in layer.layers:
                if sublayer.trainable:
                    print("  " + sublayer.name)
        else:
            if layer.trainable:
                print(layer.name)

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--model-in", required=True)
    parser.add_argument("--model-out", required=True)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    enable_gpu_memory_growth()

    train_generator, val_generator, class_weights = make_generators(
        csv_path=args.csv,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    model = tf.keras.models.load_model(args.model_in)

    model = fine_tune_setup(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nEvaluating before fine-tuning:")
    model.evaluate(val_generator)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            "outputs/finetune_log.csv",
            append=True
        )
    ]

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save(args.model_out)
    print(f"Saved fine-tuned model to {args.model_out}")


if __name__ == "__main__":
    main()