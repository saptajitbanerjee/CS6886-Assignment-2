"""
model.py -- Q1 baseline model construction: MobileNetV2 backbone (frozen,
ImageNet weights) + classification head, with L2 regularization and
dropout (Q1b: "Describe your MobileNet-v2 configuration ... and training
strategy ... regularization").
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from data import build_augmentation_layer

L2_LAMBDA = 1e-4  # gentle L2 on conv kernels -- tune if you see over/under-regularizing


def add_l2_regularizer(model: tf.keras.Model, l2_lambda: float) -> tf.keras.Model:
    """
    Adds an L2 kernel regularizer to every Conv2D / DepthwiseConv2D layer in
    `model`, via clone_model + a config override.

    WHY clone_model instead of just `layer.kernel_regularizer = ...`: Keras
    registers a layer's regularization loss via add_loss() at build() time,
    when add_weight() is called with a `regularizer=` argument. Setting the
    attribute on an already-built layer (like every layer in a loaded
    MobileNetV2()) does NOT retroactively register that loss -- it silently
    does nothing on a pretrained model. Rebuilding through clone_model
    (config -> fresh layer -> build()) is what actually makes it stick.

    DepthwiseConv2D subclasses Conv2D in Keras but uses a DIFFERENT
    regularizer attribute (depthwise_regularizer, since its weight is named
    depthwise_kernel, not kernel) -- handled as its own case, checked BEFORE
    the general Conv2D case (the more specific subclass has to be checked
    first, or it's never reached -- the same trap compress.py's
    _is_pointwise_conv has to avoid).

    Applies uniformly across the whole backbone -- doesn't distinguish
    pointwise/depthwise/stem the way Q2's quantization buckets do, since
    this is a Q1 baseline regularization choice, not part of the
    quantization scheme.
    """
    def clone_fn(layer):
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            config = layer.get_config()
            config["depthwise_regularizer"] = tf.keras.regularizers.l2(l2_lambda)
            return layer.__class__.from_config(config)
        if isinstance(layer, tf.keras.layers.Conv2D):
            config = layer.get_config()
            config["kernel_regularizer"] = tf.keras.regularizers.l2(l2_lambda)
            return layer.__class__.from_config(config)
        return layer.__class__.from_config(layer.get_config())

    regularized = tf.keras.models.clone_model(model, clone_function=clone_fn)
    assert len(regularized.layers) == len(model.layers), (
        "clone_model produced a different layer count while adding L2 "
        "regularization -- stop and check before trusting the weight copy below"
    )
    # every layer keeps the same weight shapes/order here (only a
    # regularizer got added, nothing structural changed), so a flat
    # set_weights is safe
    regularized.set_weights(model.get_weights())
    return regularized


def build_baseline_model(
    l2_lambda: float = L2_LAMBDA,
    dropout_pre: float = 0.2,
    dropout_head: float = 0.5,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    """
    Builds the Q1 baseline: MobileNetV2 (ImageNet weights, frozen) ->
    GlobalAveragePooling2D -> Dropout(dropout_pre) -> Dense(128, relu, L2)
    -> Dropout(dropout_head) -> Dense(10, softmax).

    use_augmentation=False by default: matches the actual checkpoint this
    codebase has been trained/debugged against, where the augmentation
    layer is defined (see data.build_augmentation_layer) but was not
    wired into the trained Sequential. Set True to include it.
    """
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model = add_l2_regularizer(base_model, l2_lambda)
    base_model.trainable = False

    seq_layers = []
    if use_augmentation:
        seq_layers.append(build_augmentation_layer())
    seq_layers += [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_pre),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_head),
        layers.Dense(10, activation="softmax"),
        # Dense(10) left unregularized on purpose -- L2 on the final softmax
        # layer is less standard and can distort probability calibration
    ]
    return keras.Sequential(seq_layers)
