"""
evaluate.py -- Q1(c) accuracy/curves/failure-mode reporting, and Q4's
full compression analysis (weight/activation compression ratio, accuracy,
final model size).
"""

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from data import CLASS_NAMES
from compress import (
    QuantPointwiseConv2D, QuantDense, weight_storage_report, _find_backbone_and_head,
)


def evaluate_model(model, test_ds, verbose=1):
    """Returns (loss, accuracy)."""
    return model.evaluate(test_ds, verbose=verbose)


def confusion_report(model, test_ds, class_names=CLASS_NAMES, top_k=5):
    """Q1(c): 'briefly discuss failure modes'. Returns the top_k most
    common (true -> predicted) confusions as a list of
    ((true_label, pred_label), count) tuples, and prints them."""
    preds = model.predict(test_ds, verbose=0).argmax(axis=1)
    true_labels = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    wrong = np.where(preds != true_labels)[0]
    confusions = Counter(zip(true_labels[wrong], preds[wrong]))
    top = confusions.most_common(top_k)

    print("Most common confusions (true -> predicted):")
    for (true_idx, pred_idx), count in top:
        print(f"  {class_names[true_idx]} -> {class_names[pred_idx]}: {count} times")
    return top


def plot_curves(history, out_path="loss_accuracy_curves.png"):
    """Q1(c): 'include loss/accuracy curves'. `history` is a Keras
    History.history dict (or any dict with the same keys)."""
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_range, acc, label="train")
    axes[0].plot(epochs_range, val_acc, label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs_range, loss, label="train")
    axes[1].plot(epochs_range, val_loss, label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _count_activation_quantized_layers(quantized_model):
    """How many layers actually have activation quantization applied
    (activation_bits is not None) -- for reporting alongside the
    activation compression ratio, located robustly via the backbone
    finder rather than a guessed layer index."""
    base_model, dense128, _dense10 = _find_backbone_and_head(list(quantized_model.layers))
    count = 0
    for l in base_model.layers:
        if isinstance(l, QuantPointwiseConv2D) and l.activation_bits is not None:
            count += 1
    if isinstance(dense128, QuantDense) and dense128.activation_bits is not None:
        count += 1
    return count


def compression_report(baseline_model, quantized_checkpoint, test_ds, activation_bits):
    """
    Q4: single chosen config, full report --
      (a) weight compression ratio
      (b) activation compression ratio (states how it was measured)
      (c) accuracy at this config
      (d) final approximate model size (MB)

    baseline_model: the original fp32 Q1 model (already in memory).
    quantized_checkpoint: PATH to a saved .keras checkpoint of the
      QAT-trained quantized model (e.g. produced by
      `train.py --mode qat --out checkpoints/quantized.keras`), NOT an
      in-memory model object. Loaded fresh from disk here rather than
      accepted directly, so this report can only ever be run against a
      model that has actually finished training and been saved -- a model
      built via build_quantized_model() but never fit()/saved has no
      compiled state and no QAT adaptation, and would silently report a
      meaningless number (or fail on .evaluate() with "must call
      compile()") if passed in directly instead.
    activation_bits: the activation_bits used to build quantized_checkpoint
      -- must match, since this isn't stored on the model itself.
    """
    if not isinstance(quantized_checkpoint, (str, os.PathLike)):
        raise TypeError(
            "compression_report's quantized_checkpoint argument must be a "
            "file path (str), e.g. 'checkpoints/quantized_w4a8.keras' -- "
            f"got a {type(quantized_checkpoint).__name__} instead. If you "
            "have a model object already in memory, save it first with "
            "model.save('checkpoints/quantized_w4a8.keras') and pass that "
            "path string here, don't pass the model object directly."
        )
    quantized_model = keras.models.load_model(quantized_checkpoint)

    baseline_report = weight_storage_report(baseline_model)
    quantized_report = weight_storage_report(quantized_model)
    quantized_loss, quantized_acc = quantized_model.evaluate(test_ds, verbose=0)

    weight_compression_ratio = baseline_report["total_bytes"] / quantized_report["total_bytes"]

    # Activation compression ratio: activations are never materialized to
    # disk the way weights are, so there's no "activation file size" to
    # compare directly. Expressed as a bits-per-value ratio (fp32 vs.
    # activation_bits), restricted to the layers actually covered by the
    # scheme (INT4-bucket layers: pointwise convs + the 128-unit hidden
    # Dense). FP16-bucket layers (stem/depthwise) and the untouched
    # Dense(10) output keep full-precision activations and are excluded
    # from this ratio rather than folded in, so the number isn't inflated
    # by layers the scheme never touches.
    FP32_ACTIVATION_BITS = 32
    activation_compression_ratio = FP32_ACTIVATION_BITS / activation_bits
    n_quantized_activation_layers = _count_activation_quantized_layers(quantized_model)

    final_size_mb = quantized_report["total_mb"]

    report = {
        "weight_compression_ratio": weight_compression_ratio,
        "baseline_weight_bytes": baseline_report["total_bytes"],
        "quantized_weight_bytes": quantized_report["total_bytes"],
        "activation_compression_ratio": activation_compression_ratio,
        "activation_bits": activation_bits,
        "n_quantized_activation_layers": n_quantized_activation_layers,
        "accuracy": float(quantized_acc),
        "final_size_mb": final_size_mb,
        "storage_breakdown": quantized_report,
    }

    print("=" * 70)
    print("Q4 -- Compression Analysis (single chosen config)")
    print("=" * 70)
    print(f"(a) Weight compression ratio:      {weight_compression_ratio:.2f}x")
    print(f"    fp32 baseline weight bytes:     {baseline_report['total_bytes']:,.0f}")
    print(f"    quantized weight bytes:         {quantized_report['total_bytes']:,.0f}")
    print()
    print(f"(b) Activation compression ratio:  {activation_compression_ratio:.2f}x")
    print(f"    (applies to {n_quantized_activation_layers} layers)")
    print()
    print(f"(c) Accuracy at this config:       {quantized_acc:.4f}")
    print()
    print(f"(d) Final approximate model size:  {final_size_mb:.3f} MB")
    print(f"    Breakdown: INT4 weights {quantized_report['int4_weight_bytes']/1024**2:.3f} MB, "
          f"INT4 metadata (scale/zero-point) {quantized_report['int4_metadata_bytes']/1024**2:.3f} MB, "
          f"other (FP16/FP32) weights {quantized_report['other_weight_bytes']/1024**2:.3f} MB")
    print("=" * 70)

    return None