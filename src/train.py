"""Train the WIKI-IMDB transfer-learning model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_vgg16_classifier, compile_classifier
from utils import save_history, set_seed


def make_generators(csv_path: str | Path, image_col: str, label_col: str, batch_size: int, image_size: int):
    df = pd.read_csv(csv_path)
    df[label_col] = df[label_col].astype(str)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.25)
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col=image_col,
        y_col=label_col,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    valid_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col=image_col,
        y_col=label_col,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    return train_generator, valid_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VGG16 age/gender classifier.")
    parser.add_argument("--csv", required=True, help="Training CSV.")
    parser.add_argument("--image-col", default="full_path")
    parser.add_argument("--label-col", default="final_label")
    parser.add_argument("--model-out", default="models/wiki_model.keras")
    parser.add_argument("--history-out", default="outputs/results/wiki_history.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)
    train_generator, valid_generator = make_generators(args.csv, args.image_col, args.label_col, args.batch_size, args.image_size)

    model = build_vgg16_classifier(input_shape=(args.image_size, args.image_size, 3), num_classes=train_generator.num_classes)
    model = compile_classifier(model, learning_rate=args.learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6),
        ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(train_generator, validation_data=valid_generator, epochs=args.epochs, callbacks=callbacks)
    save_history(history, args.history_out)
    model.save(args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
