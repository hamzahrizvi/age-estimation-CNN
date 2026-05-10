import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, EfficientNetB0


def build_vgg16_model(num_classes=18, image_size=224, train_base=False):
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )

    base_model.trainable = train_base

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


def build_efficientnet_model(num_classes=18, image_size=224, train_base=False):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )

    base_model.trainable = train_base

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


def unfreeze_last_vgg_block(model):
    for layer in model.layers:
        layer.trainable = False

    base_model = model.get_layer("vgg16")

    for layer in base_model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True

    return model