"""
sweep.py -- Q3: sweep weight_bits/activation_bits configs, QAT-train and
export each one, log to Wandb.

Not one of the five files the assignment names explicitly, but Q3's sweep
is a genuinely different workflow from a single train.py run (many configs,
each needing its own fresh model + report), so it gets its own file rather
than being crammed into train.py's single-run CLI.

IMPORTANT: each sweep point is built from the ORIGINAL fp32 baseline model,
never from a previously-quantized model -- chaining configs off each other
was a real bug earlier in this project's history (silently degrades
accuracy in a way that's hard to distinguish from "that bit-width is just
bad").

Usage:
    python sweep.py --checkpoint-in checkpoints/baseline.keras \\
        --project cs6886-assignment2 --epochs 5
"""

import argparse

import wandb
from tensorflow import keras

from data import build_datasets, set_seed
from compress import build_quantized_model, weight_storage_report

SWEEP_CONFIGS = [  
    {"weight_bits": 4, "activation_bits": 4},
    {"weight_bits": 4, "activation_bits": 6},
    {"weight_bits": 4, "activation_bits": 8},
    {"weight_bits": 6, "activation_bits": 4},
    {"weight_bits": 6, "activation_bits": 6},
    {"weight_bits": 6, "activation_bits": 8},
    {"weight_bits": 8, "activation_bits": 4},
    {"weight_bits": 8, "activation_bits": 6},
    {"weight_bits": 8, "activation_bits": 8},

]

# Extreme low bit-widths (2-bit weight/activation) have collapsed to
# chance-level accuracy from epoch 1 in this project's own sweep runs, at
# the same LR used for 4/8/16-bit configs. That's very plausibly a real
# optimization-difficulty effect at 2-bit, not a bug -- QAT from a normal
# fp32 checkpoint at 2 bits is genuinely hard without extra tricks
# (progressive bit-width reduction, a much lower LR, longer warmup). This
# dict lets the 2-bit point use a gentler LR instead of silently reusing
# the same one as everything else; add other overrides here if other
# configs show the same pattern.
LR_OVERRIDES = {
    (2, 2): 2e-6,
}
DEFAULT_LR = 1e-4


def run_sweep(args):
    # val_ds drives early stopping for every sweep point; test_ds is only
    # ever touched once per config, after training, for the number that
    # actually gets logged to Wandb -- same split discipline as train.py
    train_ds, val_ds, test_ds = build_datasets()
    trained_model = keras.models.load_model(args.checkpoint_in)
    baseline_report = weight_storage_report(trained_model)
    baseline_bytes = baseline_report["total_bytes"]

    for cfg in args.configs:
        wb, ab = cfg["weight_bits"], cfg["activation_bits"]
        lr = LR_OVERRIDES.get((wb, ab), DEFAULT_LR)

        run = wandb.init(
            project=args.project, name=f"w{wb}_a{ab}", config={**cfg, "lr": lr},
            reinit=True,
        )

        q_model = build_quantized_model(trained_model, weight_bits=wb, activation_bits=ab)
        q_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"],
        )
        q_model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        )

        # test_ds only appears here -- once per sweep point, after training
        # is fully done (including early-stopping's restore_best_weights,
        # which already used val_ds) -- so this number is a clean,
        # untouched-holdout accuracy, not something the config was tuned on
        _, quantized_acc = q_model.evaluate(test_ds)

        # A near-chance accuracy (1/num_classes = 0.10 for CIFAR-10) after
        # QAT training almost always means a real bug, not just quantization
        # damage -- stop and fix rather than silently log a broken point
        # into the sweep chart. Extreme bit-widths (see LR_OVERRIDES note)
        # are the one legitimate exception; everything else at 0.10 needs
        # investigating before you trust it.
        if quantized_acc < 0.15 and (wb, ab) not in LR_OVERRIDES:
            raise RuntimeError(
                f"quantized_acc collapsed to near-chance ({quantized_acc:.4f}) for "
                f"weight_bits={wb}, activation_bits={ab} -- stop and check "
                f"build_quantized_model rather than logging this point"
            )

        report = weight_storage_report(q_model)
        compression_ratio = baseline_bytes / report["total_bytes"]

        wandb.log({
            "weight_quant_bits": wb,
            "activation_quant_bits": ab,
            "compression_ratio": compression_ratio,
            "model_size_mb": report["total_mb"],
            "quantized_acc": quantized_acc,
        })
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-in", type=str, required=True,
                         help="Path to the trained Q1 baseline .keras file.")
    parser.add_argument("--project", type=str, default="cs6886-assignment2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.configs = SWEEP_CONFIGS

    set_seed(args.seed)
    wandb.login()
    run_sweep(args)


if __name__ == "__main__":
    main()