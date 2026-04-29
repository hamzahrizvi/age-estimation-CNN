"""Model definitions for facial age estimation."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16


def build_vgg16_classifier(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 18,
    train_base: bool = False,
    dense_units: int = 512,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Build an improved VGG16 transfer-learning classifier.

    This replaces the original Flatten + Dense(4000) head with
    GlobalAveragePooling2D to reduce parameters and overfitting risk.
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = train_base

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(dense_units, activation="relu", name="features")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="vgg16_age_gender_classifier")
    return model


def compile_classifier(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Compile the classifier with Adam and categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
    )
    return model


def unfreeze_last_vgg_block(model: tf.keras.Model, block_name: str = "block5", learning_rate: float = 1e-5) -> tf.keras.Model:
    """Unfreeze only the final VGG block for safer fine-tuning."""
    base_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "vgg16" in layer.name), None)
    if base_model is None:
        raise ValueError("Could not find VGG16 base model inside the supplied model.")

    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = layer.name.startswith(block_name)

    return compile_classifier(model, learning_rate=learning_rate)
