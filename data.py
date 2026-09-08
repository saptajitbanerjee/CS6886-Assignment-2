"""
data.py -- CIFAR-10 loading, preprocessing, and the tf.data pipeline
(Q1a: "Prepare CIFAR-10 with proper normalization and data augmentation;
specify transforms").

Also owns the global seed, since reproducible data order/augmentation is
part of what a fixed seed needs to cover (Q5b: "include seed configuration").
"""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def set_seed(seed: int = 42) -> None:
    """Seeds Python's random, NumPy, and TensorFlow, plus PYTHONHASHSEED.
    Call this once, before building datasets or models, for reproducibility.
    Note: this makes data ORDER and augmentation reproducible; it does not
    by itself make GPU training bit-exact (some cuDNN ops are nondeterministic
    regardless of seed -- see README for the op-determinism flag if you need
    that too)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_cifar10():
    """Returns (x_train, y_train), (x_test, y_test) exactly as
    keras.datasets.cifar10.load_data() provides them (labels shaped
    (N, 1), not yet squeezed -- preprocess() below does that)."""
    return keras.datasets.cifar10.load_data()


def preprocess(image, label):
    """Resize 32x32 -> 224x224 (MobileNetV2's expected input) inside the
    tf.data pipeline (not eagerly upfront), and scale to [-1, 1] via
    MobileNetV2's own preprocess_input. label[0] unwraps CIFAR-10's
    (1,)-shaped per-sample label into a scalar for
    sparse_categorical_crossentropy."""
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)
    return image, label[0]


VAL_SPLIT = 0.1  # fraction of the CIFAR-10 train split held out for validation


def build_datasets(batch_size: int = BATCH_SIZE, val_split: float = VAL_SPLIT):
    """Builds train_ds/val_ds/test_ds tf.data pipelines from CIFAR-10.

    CIFAR-10's own train/test split (50k/10k) only gives you two sets, and
    the test set is meant to be a final, untouched holdout -- it must never
    be used for early stopping, model selection, or LR/config decisions
    during training (that's what leaks test-set information into the
    "trained" model and inflates its reported accuracy). So a validation
    set is carved out of the 50k TRAIN images here (`val_split` fraction,
    default 10% = 5k images), and the original 10k TEST images are kept
    completely separate, only ever touched by evaluate.py /
    weight_storage_report-style final reporting.

    Split is done by shuffling indices with NumPy's global RNG -- call
    set_seed() before this if you need that split to be reproducible.

    Returns (train_ds, val_ds, test_ds). Pass train_ds/val_ds to
    model.fit(..., validation_data=val_ds); only ever call
    model.evaluate(test_ds) once you're done tuning/training, for the
    final reported number.
    """
    (x_train_full, y_train_full), (x_test, y_test) = load_cifar10()

    n_total = x_train_full.shape[0]
    n_val = int(n_total * val_split)

    # shuffle indices before splitting off validation, so val isn't just
    # "the last N images in CIFAR-10's stored order" (which is grouped by
    # class in the raw dataset)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds


def build_augmentation_layer():
    """The augmentation pipeline used for Q1a. NOTE: as of the last working
    notebook, this is defined but NOT wired into the trained model
    (model.py's build_baseline_model defaults to use_augmentation=False,
    matching the actual checkpoint you've been training against). Pass
    use_augmentation=True to build_baseline_model to include this."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            # layers.RandomRotation(factor=0.05),
            # layers.RandomTranslation(0.1, 0.1),
            # layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]