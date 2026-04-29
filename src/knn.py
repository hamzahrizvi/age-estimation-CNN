"""Experimental KNN classifier on extracted CNN features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from evaluate import make_test_generator


def extract_features(model: keras.Model, generator, layer_name: str = "features") -> np.ndarray:
    """Extract intermediate features from a named model layer."""
    feature_model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return feature_model.predict(generator, verbose=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KNN on CNN feature embeddings.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model", default="models/utk_finetuned_model.keras")
    parser.add_argument("--image-col", default="full_path")
    parser.add_argument("--label-col", default="final_label")
    parser.add_argument("--layer-name", default="features")
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    train_gen = make_test_generator(args.train_csv, args.image_col, args.label_col, args.batch_size, args.image_size)
    test_gen = make_test_generator(args.test_csv, args.image_col, args.label_col, args.batch_size, args.image_size)

    x_train = extract_features(model, train_gen, args.layer_name)
    x_test = extract_features(model, test_gen, args.layer_name)
    y_train = train_gen.classes
    y_test = test_gen.classes

    classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=args.neighbors))
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    print(f"KNN accuracy: {score:.4f}")


if __name__ == "__main__":
    main()
