"""Fine-tune the saved WIKI-IMDB model on UTKFace."""

from __future__ import annotations

import argparse

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import unfreeze_last_vgg_block
from train import make_generators
from utils import save_history, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune model on UTKFace.")
    parser.add_argument("--csv", required=True, help="UTKFace CSV.")
    parser.add_argument("--model-in", default="models/wiki_model.keras")
    parser.add_argument("--model-out", default="models/utk_finetuned_model.keras")
    parser.add_argument("--history-out", default="outputs/results/utk_history.json")
    parser.add_argument("--image-col", default="full_path")
    parser.add_argument("--label-col", default="final_label")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(42)
    train_generator, valid_generator = make_generators(args.csv, args.image_col, args.label_col, args.batch_size, args.image_size)

    model = keras.models.load_model(args.model_in)
    model = unfreeze_last_vgg_block(model, learning_rate=args.learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7),
        ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(train_generator, validation_data=valid_generator, epochs=args.epochs, callbacks=callbacks)
    save_history(history, args.history_out)
    model.save(args.model_out)
    print(f"Saved fine-tuned model to {args.model_out}")


if __name__ == "__main__":
    main()
