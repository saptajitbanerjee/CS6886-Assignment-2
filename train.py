"""
train.py -- CLI entry point for Q1 (baseline) and Q2 (QAT) training.

Examples:
    # Q1 baseline
    python train.py --mode baseline --epochs 30 --out checkpoints/baseline.keras

    # Q2 QAT, starting from a trained baseline checkpoint
    python train.py --mode qat --checkpoint-in checkpoints/baseline.keras \\
        --weight-bits 4 --activation-bits 8 --epochs 15 --lr 1e-5 \\
        --out checkpoints/quantized.keras
"""

import argparse
import json
import os

from tensorflow import keras

from data import build_datasets, set_seed
from model import build_baseline_model
from compress import build_quantized_model


def train_baseline(args):
    model = build_baseline_model()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )
    # train_ds/val_ds drive fitting + early stopping; test_ds is held out
    # untouched until the post-training evaluate() call below
    train_ds, val_ds, test_ds = build_datasets()
    early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    hist = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    return model, hist, (test_loss, test_acc)


def train_qat(args):
    if not args.checkpoint_in:
        raise ValueError("--checkpoint-in is required for --mode qat "
                          "(path to a trained Q1 baseline .keras file)")
    trained_model = keras.models.load_model(args.checkpoint_in)
    quantized_model = build_quantized_model(
        trained_model, weight_bits=args.weight_bits, activation_bits=args.activation_bits
    )
    quantized_model.compile(
        loss="sparse_categorical_crossentropy",
        # low LR: the backbone just went from frozen to unfrozen AND had
        # fake-quant noise injected in the same step -- a baseline-sized LR
        # here risks wrecking the pretrained features before QAT adapts them
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )
    # same split discipline as the baseline: fit/early-stop against
    # train_ds/val_ds only, test_ds stays untouched until evaluate() below
    train_ds, val_ds, test_ds = build_datasets()
    early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    hist = quantized_model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = quantized_model.evaluate(test_ds, verbose=1)
    return quantized_model, hist, (test_loss, test_acc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "qat"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None,
                         help="Defaults to 1e-4 for baseline, 1e-5 for qat if not set.")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--checkpoint-in", type=str, default=None,
                         help="Required for --mode qat: path to a trained baseline .keras file.")
    parser.add_argument("--out", type=str, required=True,
                         help="Path to save the trained .keras model to.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 1e-4 if args.mode == "baseline" else 1e-5

    set_seed(args.seed)

    if args.mode == "baseline":
        model, hist, (test_loss, test_acc) = train_baseline(args)
    else:
        model, hist, (test_loss, test_acc) = train_qat(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)

    history_path = os.path.splitext(args.out)[0] + "_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_run": len(hist.history["loss"]),
            "lr": args.lr,
            "weight_bits": args.weight_bits if args.mode == "qat" else None,
            "activation_bits": args.activation_bits if args.mode == "qat" else None,
            # train/val curves from fit(); test_loss/test_acc below are the
            # single post-training numbers from the held-out test set,
            # never used for early stopping or any training decision
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            **hist.history,
        }, f, indent=2)

    print(f"Saved model to {args.out}")
    print(f"Saved history to {history_path}")
    print(f"Test loss/accuracy (held-out, post-training): {test_loss:.4f} / {test_acc:.4f}")


if __name__ == "__main__":
    main()