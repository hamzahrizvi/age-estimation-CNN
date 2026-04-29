"""Evaluate the age/gender classifier on FG-Net or another test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_test_generator(csv_path: str | Path, image_col: str, label_col: str, batch_size: int, image_size: int):
    df = pd.read_csv(csv_path)
    df[label_col] = df[label_col].astype(str)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col=image_col,
        y_col=label_col,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on a held-out dataset.")
    parser.add_argument("--csv", required=True, help="Test CSV.")
    parser.add_argument("--model", default="models/utk_finetuned_model.keras")
    parser.add_argument("--image-col", default="full_path")
    parser.add_argument("--label-col", default="final_label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--results-out", default="outputs/results/evaluation_report.txt")
    parser.add_argument("--confusion-out", default="outputs/results/confusion_matrix.csv")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    generator = make_test_generator(args.csv, args.image_col, args.label_col, args.batch_size, args.image_size)

    loss, accuracy, *rest = model.evaluate(generator, verbose=1)
    probabilities = model.predict(generator, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = generator.classes

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    plain_accuracy = accuracy_score(y_true, y_pred)

    results_text = (
        f"Loss: {loss:.6f}\n"
        f"Accuracy: {accuracy:.6f}\n"
        f"Accuracy sklearn: {plain_accuracy:.6f}\n"
        f"Balanced accuracy: {balanced:.6f}\n\n"
        f"Classification report:\n{report}\n"
    )

    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_out).write_text(results_text)
    pd.DataFrame(cm).to_csv(args.confusion_out, index=False)
    print(results_text)
    print(f"Saved report to {args.results_out}")
    print(f"Saved confusion matrix to {args.confusion_out}")


if __name__ == "__main__":
    main()
