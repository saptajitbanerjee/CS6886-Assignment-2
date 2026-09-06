# CS6886 Assignment 2 -- MobileNetV2 / CIFAR-10 Quantization

Train MobileNetV2 on CIFAR-10 (Q1), implement per-channel INT4 weight
quantization + EMA-calibrated activation quantization from scratch via QAT,
with a mixed-precision FP16/INT4 layer scheme (Q2), sweep bit-widths against
accuracy (Q3), and report the final chosen config (Q4).

## Layout

| File | Covers |
|---|---|
| `data.py` | CIFAR-10 loading, tf.data pipeline, augmentation, global seed |
| `model.py` | Q1 baseline model: MobileNetV2 (frozen) + regularized head |
| `compress.py` | Q2: fake-quant primitives, QAT layers, bucket rules, storage accounting |
| `train.py` | CLI: train the Q1 baseline, or QAT-fine-tune a quantized model |
| `evaluate.py` | Q1(c) accuracy/curves/confusions; Q4 full compression report |
| `sweep.py` | Q3: bit-width sweep, Wandb logging |

## Setup

```bash
pip install -r requirements.txt
```

Needs a Wandb account + API key (`wandb login`) for `sweep.py` only -- `train.py` alone has no Wandb dependency.

## Seed configuration

`data.set_seed(seed)` (default `42`) seeds Python's `random`, NumPy,
TensorFlow, and `PYTHONHASHSEED`. `train.py` and `sweep.py` both call this
first thing, and both take `--seed` if you want to vary it. This makes data
order and augmentation reproducible. It does **not** by itself make GPU
training bit-exact -- some cuDNN ops are nondeterministic regardless of
seed; if you need that too, additionally set
`os.environ["TF_DETERMINISTIC_OPS"] = "1"` before importing TensorFlow
(slower, so it's not on by default here).

## Reproducing each question

**Q1 -- baseline:**
```bash
python train.py --mode baseline --epochs 50 --out checkpoints/baseline.keras
```
Then, in a Python shell or notebook:
```python
import json
from data import build_datasets
from evaluate import evaluate_model, confusion_report, plot_curves
from tensorflow import keras

model = keras.models.load_model("checkpoints/baseline.keras")
_, test_ds = build_datasets()
evaluate_model(model, test_ds)
confusion_report(model, test_ds)
history = json.load(open("checkpoints/baseline_history.json"))
plot_curves(history, "loss_accuracy_curves.png")
```

**Q2 -- quantize + QAT fine-tune, single config:**
```bash
python train.py --mode qat --checkpoint-in checkpoints/baseline.keras \
    --weight-bits 4 --activation-bits 8 --epochs 50 \
    --out checkpoints/quantized.keras
```
Then check the storage improvements:
```python
from tensorflow import keras
from compress import weight_storage_report

q_model = keras.models.load_model("checkpoints/quantized.keras")
print(weight_storage_report(q_model))
```

**Q3 -- bit-width sweep:**
```bash
python sweep.py --checkpoint-in checkpoints/baseline.keras --project cs6886-assignment2

**Q4 -- final chosen-config report:**

First, actually QAT-train and save the chosen config as a checkpoint (this
is the step that must happen before reporting -- `compression_report`
below loads this file, it does not accept an in-memory, untrained model):
```bash
python train.py --mode qat --checkpoint-in checkpoints/baseline.keras \
    --weight-bits 4 --activation-bits 8 --epochs 15 \
    --out checkpoints/quantized_w4a8.keras
```
Then generate the report from the two saved checkpoints:
```python
from tensorflow import keras
from data import build_datasets
from evaluate import compression_report

baseline = keras.models.load_model("checkpoints/baseline.keras")
_, test_ds = build_datasets()

compression_report(baseline, "checkpoints/quantized.keras", test_ds, activation_bits=8)
```

**Q5 -- this repo.**

## Design notes worth knowing before reading the code

- **Bucket rules** (Q2b): stem conv, depthwise convs, and the final
  `Dense(10, softmax)` are FP16 (real float16 storage+compute). Every 1x1
  pointwise conv (expansion, projection, and the final `Conv_1` top conv)
  and the 128-unit hidden Dense layer are INT4, via QAT with a
  straight-through estimator. Every BatchNorm layer in the backbone is
  rebuilt as FP16, regardless of whether it follows an FP16-bucket conv or
  an INT4 pointwise conv.
- **`data_augmentation` is defined in `data.py` but not applied by default**
  (`model.build_baseline_model(use_augmentation=False)`) -- matches what
  was actually trained/validated in this project's history. Flip the flag
  if you want it included.
- **Always build a QAT model from the original fp32 baseline**, never from
  an already-quantized model -- `sweep.py` and `train.py --mode qat` both
  require an explicit `--checkpoint-in` for exactly this reason.
- Extreme low bit-widths (2-bit weight/activation) have been observed to
  collapse to chance-level accuracy during QAT at the same LR used for
  4/8/16-bit configs -- see `sweep.py`'s `LR_OVERRIDES` for where to adjust
  this per-config rather than assuming it's a bug every time.
