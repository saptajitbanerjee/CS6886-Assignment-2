"""
compress.py -- Q2 model compression: per-channel INT4 weight quantization
(QAT, with a straight-through estimator), EMA-calibrated activation
quantization, and mixed-precision FP16 bucketing.

Design (Q2b: "which layers compressed, any exceptions"):
  - Per-channel uniform affine quantization for weights: scale + zero-point
    per output channel, round-and-clamp, dequantize for the forward pass.
  - QAT, not PTQ: fake-quant (quantize-then-dequantize) is inserted into the
    forward pass so weights adapt to rounding noise via a straight-through
    estimator (STE) on round(). The backbone must be UNFROZEN for this
    (build_quantized_model does this for you) -- QAT only does anything if
    gradients actually reach the fake-quantized weights.
  - Bucket rules:
      * Stem conv, depthwise convs, Dense(10, softmax) [last layer]
        -> FP16 bucket: real float16 storage + compute (not just cast at
        export time).
      * Every 1x1 (pointwise) Conv2D -- expansion, projection, and the
        final Conv_1 top conv -- -> INT4 (weight_bits, default 4), wrapped
        in QuantPointwiseConv2D. This is where most of MobileNetV2's
        parameters live, so it's where the compression ratio comes from.
      * The 128-unit hidden Dense layer -> INT4, wrapped in QuantDense.
      * Activations: EMA-calibrated fake-quant (ActivationQuantMixin) on
        the INT4-bucket layers' outputs, at `activation_bits` (Q2a: "reducing
        both model weights and activations").
  - BatchNorm: every BatchNorm layer in the backbone is rebuilt as FP16,
    regardless of whether it follows an FP16-bucket conv (stem/depthwise)
    or an INT4 pointwise conv -- BN's own parameters are a tiny fraction
    of total model size, so keeping them at higher precision everywhere
    costs little and avoids compounding INT4 rounding error through BN's
    normalization statistics.

Keras-native equivalent of a "forward hook": tf.keras.models.clone_model
with a clone_function, since MobileNetV2's Keras application is a
Functional model and there's no PyTorch-style module hook API here.

Layer detection (build_quantized_model) works by TYPE/SHAPE,
not fixed layer position -- robust to whether a data_augmentation layer (or
anything else) is present, absent, or reordered in the outer Sequential.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, Dense, BatchNormalization,
)

try:
    _register = tf.keras.saving.register_keras_serializable
except AttributeError:  # older TF/Keras
    _register = tf.keras.utils.register_keras_serializable


# ---------------------------------------------------------------------------
# Core fake-quant primitives
# ---------------------------------------------------------------------------

@tf.custom_gradient
def _ste_round(x):
    """round(x) on the forward pass; identity gradient on the backward pass
    (straight-through estimator) so gradients can flow through the
    non-differentiable round op during QAT."""
    def grad(dy):
        return dy
    return tf.round(x), grad


def fake_quantize_per_channel(w, weight_bits, channel_axis=-1, eps=1e-8):
    """
    Per-channel uniform affine fake-quantization: quantize-then-dequantize w.

    w: weight tensor of any rank.
    weight_bits: bit-width of the quantized grid (4 for INT4).
    channel_axis: axis of w treated as "output channel" (own scale/zero
      point per index along this axis) -- -1 for both Conv2D kernels
      (kh, kw, in_ch, out_ch) and Dense kernels (in_features, out_features).

    Returns (w_dequantized, scale, zero_point) -- scale/zero_point are
    1-D (one value per channel), for the Q2c storage-overhead accounting.
    """
    rank = len(w.shape)
    axis = channel_axis % rank
    perm = [axis] + [a for a in range(rank) if a != axis]
    inv_perm = [0] * rank
    for i, p in enumerate(perm):
        inv_perm[p] = i

    qmin, qmax = 0.0, float(2 ** weight_bits - 1)

    w_t = tf.transpose(w, perm)
    channels = w_t.shape[0]
    w_flat = tf.reshape(w_t, [channels, -1])

    w_min = tf.reduce_min(w_flat, axis=1)
    w_max = tf.reduce_max(w_flat, axis=1)

    scale = tf.maximum((w_max - w_min) / (qmax - qmin), eps)
    zero_point = _ste_round(qmin - w_min / scale)
    zero_point = tf.clip_by_value(zero_point, qmin, qmax)

    scale_b = tf.reshape(scale, [-1, 1])
    zp_b = tf.reshape(zero_point, [-1, 1])

    w_q = _ste_round(w_flat / scale_b + zp_b)
    w_q = tf.clip_by_value(w_q, qmin, qmax)
    w_dq = (w_q - zp_b) * scale_b

    w_dq = tf.reshape(w_dq, w_t.shape)
    w_dq = tf.transpose(w_dq, inv_perm)
    return w_dq, scale, zero_point


def fake_quantize_activation(x, num_bits, min_val, max_val, eps=1e-8):
    """Per-tensor uniform affine fake-quantization for activations."""
    qmin, qmax = 0.0, float(2 ** num_bits - 1)
    scale = tf.maximum((max_val - min_val) / (qmax - qmin), eps)
    zero_point = _ste_round(qmin - min_val / scale)
    zero_point = tf.clip_by_value(zero_point, qmin, qmax)

    x_q = _ste_round(x / scale + zero_point)
    x_q = tf.clip_by_value(x_q, qmin, qmax)
    return (x_q - zero_point) * scale


class ActivationQuantMixin:
    """Adds EMA-calibrated activation fake-quant to a layer's output.
    Mix this into a layer alongside tf.keras.layers.Layer."""

    def _build_act_quant_state(self):
        self.act_min = self.add_weight(name="act_min", shape=(), trainable=False,
                                        initializer=tf.constant_initializer(0.0))
        self.act_max = self.add_weight(name="act_max", shape=(), trainable=False,
                                        initializer=tf.constant_initializer(1.0))

    def _quantize_activation(self, out, training):
        if self.activation_bits is None:
            return out
        batch_min = tf.reduce_min(out)
        batch_max = tf.reduce_max(out)
        if training:
            m = 0.9  # EMA momentum
            self.act_min.assign(m * self.act_min + (1 - m) * batch_min)
            self.act_max.assign(m * self.act_max + (1 - m) * batch_max)
            use_min, use_max = batch_min, batch_max  # live batch range while training
        else:
            use_min, use_max = self.act_min, self.act_max  # frozen calibrated range
        return fake_quantize_activation(out, self.activation_bits, use_min, use_max)


# ---------------------------------------------------------------------------
# Wrapper layers -- INT4 bucket (pointwise convs + the 128-unit hidden Dense)
# ---------------------------------------------------------------------------

@_register(package="cs6886")
class QuantPointwiseConv2D(ActivationQuantMixin, tf.keras.layers.Layer):
    """Drop-in replacement for a 1x1 Conv2D layer: fake-quantizes its
    kernel (per output channel, INT4 by default) and its output activation
    (EMA-calibrated, if activation_bits is set) on every forward pass."""

    def __init__(self, filters, weight_bits=4, activation_bits=None, strides=(1, 1),
                 padding="same", use_bias=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self._last_scale = None
        self._last_zero_point = None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.kernel = self.add_weight(name="kernel", shape=(1, 1, in_ch, self.filters),
                                       initializer="glorot_uniform", trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.filters,),
                                         initializer="zeros", trainable=True)
        self._build_act_quant_state()
        super().build(input_shape)

    def call(self, inputs, training=None):
        w_dq, scale, zero_point = fake_quantize_per_channel(
            self.kernel, self.weight_bits, channel_axis=-1
        )
        self._last_scale, self._last_zero_point = scale, zero_point
        x = tf.cast(inputs, w_dq.dtype)
        out = tf.nn.conv2d(x, w_dq, strides=[1, self.strides[0], self.strides[1], 1],
                            padding=self.padding)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        return self._quantize_activation(out, training)

    def storage_bytes(self):
        """(quantized_weight_bytes, metadata_bytes) for Q2c accounting."""
        n_params = int(tf.size(self.kernel).numpy())
        weight_bytes = n_params * self.weight_bits / 8.0
        meta_bytes = self.filters * (4 + 4)  # fp32 scale + fp32 zero_point / out-channel
        return weight_bytes, meta_bytes

    def load_from_conv2d(self, conv_layer):
        """Copy weights from the original (unquantized) 1x1 Conv2D layer.
        Only overwrites kernel (+ bias, if present) -- leaves act_min/act_max
        at their fresh initial values, since the source layer never had
        that state to begin with."""
        conv_weights = conv_layer.get_weights()
        kernel_and_bias = conv_weights if self.use_bias else conv_weights[:1]
        current = self.get_weights()
        new_weights = list(kernel_and_bias) + current[len(kernel_and_bias):]
        self.set_weights(new_weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "strides": self.strides,
            "padding": self.padding.lower(),
            "use_bias": self.use_bias,
        })
        return config


@_register(package="cs6886")
class QuantDense(ActivationQuantMixin, tf.keras.layers.Layer):
    """Drop-in replacement for a Dense layer: fake-quantizes its kernel
    (per output neuron, INT4 by default) and its output activation
    (EMA-calibrated, post-activation-function) on every forward pass."""

    def __init__(self, units, weight_bits=4, activation_bits=None, activation=None,
                 use_bias=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self._last_scale = None
        self._last_zero_point = None

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel", shape=(in_features, self.units),
            initializer="glorot_uniform", trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,),
                initializer="zeros", trainable=True,
            )
        self._build_act_quant_state()
        super().build(input_shape)

    def call(self, inputs, training=None):
        w_dq, scale, zero_point = fake_quantize_per_channel(
            self.kernel, self.weight_bits, channel_axis=-1
        )
        self._last_scale, self._last_zero_point = scale, zero_point
        x = tf.cast(inputs, w_dq.dtype)
        out = tf.matmul(x, w_dq)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            out = self.activation(out)
        # quantized AFTER the activation function -- unlike the pointwise
        # convs (where ReLU6 is a separate downstream layer), this
        # activation is intrinsic to the layer, so the real value flowing
        # to the next layer is the post-activation one
        return self._quantize_activation(out, training)

    def storage_bytes(self):
        n_params = int(tf.size(self.kernel).numpy())
        weight_bytes = n_params * self.weight_bits / 8.0
        meta_bytes = self.units * (4 + 4)
        return weight_bytes, meta_bytes

    def load_from_dense(self, dense_layer):
        dense_weights = dense_layer.get_weights()
        kernel_and_bias = dense_weights if self.use_bias else dense_weights[:1]
        current = self.get_weights()
        new_weights = list(kernel_and_bias) + current[len(kernel_and_bias):]
        self.set_weights(new_weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
        })
        return config


# ---------------------------------------------------------------------------
# Layer-bucket classification (by type/shape, not position)
# ---------------------------------------------------------------------------

def _is_pointwise_conv(layer):
    """True for a plain 1x1 Conv2D -- catches expansion convs, projection
    convs, and the final Conv_1 top conv automatically, regardless of how
    many inverted-residual blocks the model has. Explicitly excludes
    DepthwiseConv2D, which subclasses Conv2D in Keras and would otherwise
    match too."""
    return (
        isinstance(layer, Conv2D)
        and not isinstance(layer, DepthwiseConv2D)
        and tuple(layer.kernel_size) == (1, 1)
    )


def _clone_fn(layer, weight_bits, activation_bits):
    if _is_pointwise_conv(layer):
        return QuantPointwiseConv2D(
            filters=layer.filters, weight_bits=weight_bits, activation_bits=activation_bits,
            strides=layer.strides, padding=layer.padding,
            use_bias=layer.use_bias, name=layer.name + "_q",
        )
    if isinstance(layer, BatchNormalization):
        # every BatchNorm layer is FP16, regardless of whether it follows an
        # FP16-bucket conv (stem/depthwise) or an INT4 pointwise conv -- BN's
        # own parameters (gamma/beta/moving_mean/moving_var) are cheap to
        # keep at higher precision even downstream of an aggressively
        # quantized conv, since they're a tiny fraction of total size
        config = layer.get_config()
        config["dtype"] = "float16"
        return layer.__class__.from_config(config)
    if isinstance(layer, (Conv2D, DepthwiseConv2D)) and not _is_pointwise_conv(layer):
        config = layer.get_config()
        config["dtype"] = "float16"
        return layer.__class__.from_config(config)
    return layer.__class__.from_config(layer.get_config())


def _find_backbone_and_head(layers_list):
    """
    Finds the nested backbone model, the 128-unit hidden layer, and the
    10-unit output layer inside a Sequential's layer list, by type/shape
    rather than fixed position -- robust to whether a data_augmentation
    layer (or anything else) is present, absent, or reordered.

    Works on both the original Q1 model (backbone = plain Functional Model,
    hidden layer = plain Dense(128)) and an already-quantized model
    (backbone = the cloned quantized_base, still a Model; hidden layer =
    QuantDense with units==128 -- matched via a generic `.units` check,
    since QuantDense isn't a Dense subclass). The output layer is always
    matched as a plain Dense(10), since it's never wrapped.
    """
    backbone_candidates = [
        l for l in layers_list
        if isinstance(l, tf.keras.Model) and l.name != "data_augmentation"
    ]
    assert len(backbone_candidates) == 1, (
        f"expected exactly one nested Model (the MobileNetV2 backbone) after "
        f"excluding data_augmentation, found {len(backbone_candidates)}: "
        f"{[l.name for l in backbone_candidates]} -- print the outer "
        f"model.summary() and adjust before trusting anything below"
    )
    backbone = backbone_candidates[0]

    hidden_candidates = [
        l for l in layers_list
        if l is not backbone and getattr(l, "units", None) == 128
    ]
    out_candidates = [
        l for l in layers_list if isinstance(l, Dense) and l.units == 10
    ]
    assert len(hidden_candidates) == 1, (
        f"expected exactly one 128-unit hidden layer, found "
        f"{[(l.name, getattr(l, 'units', None)) for l in hidden_candidates]} -- "
        f"print the outer model.summary() and adjust before trusting anything below"
    )
    assert len(out_candidates) == 1, (
        f"expected exactly one Dense(10) output layer, found "
        f"{[(l.name, l.units) for l in out_candidates]} -- "
        f"print the outer model.summary() and adjust before trusting anything below"
    )
    return backbone, hidden_candidates[0], out_candidates[0]


# ---------------------------------------------------------------------------
# Model surgery: build the training-time quantized model
# ---------------------------------------------------------------------------

def build_quantized_model(trained_model, weight_bits=4, activation_bits=8):
    """
    trained_model: your trained Q1 model (a Sequential containing, in any
    order/position: the MobileNetV2 backbone, a 128-unit hidden Dense
    layer, a 10-unit output Dense layer, and whatever else is there).

    Returns a new Sequential model, same layer order, where:
      - every 1x1 pointwise Conv2D in the backbone and the 128-unit hidden
        layer are replaced with fake-quant INT4 (+ activation-quant)
        wrappers, weights copied over from trained_model;
      - the stem conv and depthwise convs are rebuilt as real float16
        layers (storage + compute);
      - every BatchNorm layer in the backbone is rebuilt as float16,
        whether it follows an FP16-bucket conv or an INT4 pointwise conv;
      - the 10-unit output layer is left completely untouched (original
        float32);
      - every other layer is carried through unchanged, in its original
        position.

    The returned model's backbone is set trainable=True (unlike Q1) --
    QAT only does anything if gradients reach the fake-quantized weights.

    IMPORTANT: always pass the ORIGINAL fp32 trained_model here, never an
    already-quantized model -- each quantization config should start fresh
    from the Q1 checkpoint, not chain off a previous quantization run.
    """
    layers_list = list(trained_model.layers)
    base_model, dense128, dense10 = _find_backbone_and_head(layers_list)

    quantized_base = tf.keras.models.clone_model(
        base_model,
        clone_function=lambda l: _clone_fn(l, weight_bits, activation_bits),
    )
    assert len(quantized_base.layers) == len(base_model.layers), (
        "clone_model produced a different layer count than the original "
        "backbone -- stop and investigate rather than proceed past this"
    )

    for orig_layer, new_layer in zip(base_model.layers, quantized_base.layers):
        if isinstance(new_layer, QuantPointwiseConv2D):
            new_layer.load_from_conv2d(orig_layer)
        elif new_layer.weights:
            new_layer.set_weights(orig_layer.get_weights())

    new_dense128 = QuantDense(
        units=dense128.units, weight_bits=weight_bits, activation_bits=activation_bits,
        activation=dense128.activation, use_bias=dense128.use_bias,
        name=dense128.name + "_q",
    )

    new_sequence = []
    for l in layers_list:
        if l is base_model:
            new_sequence.append(quantized_base)
        elif l is dense128:
            new_sequence.append(new_dense128)
        else:
            new_sequence.append(l)
    quantized_model = tf.keras.Sequential(new_sequence)
    quantized_model.build(input_shape=(None, 224, 224, 3))
    new_dense128.load_from_dense(dense128)

    quantized_base.trainable = True
    return quantized_model


# ---------------------------------------------------------------------------
# Q2c storage accounting
# ---------------------------------------------------------------------------

def weight_storage_report(model):
    """
    Walks the model and sums up storage bytes for Q2c:
      - INT4 bucket (QuantPointwiseConv2D / QuantDense): quantized weight
        bytes (weight_bits/8 per param) + per-channel scale/zero-point
        metadata bytes (fp32 each).
      - Every other layer with weights: counted at that weight's ACTUAL
        dtype size (real float16 layers count as 2 bytes/param
        automatically; anything left float32 counts as 4).
    Returns a dict with the breakdown and totals, in bytes and MB.
    """
    int4_weight_bytes = int4_meta_bytes = other_weight_bytes = 0

    def walk(model_or_layer):
        nonlocal int4_weight_bytes, int4_meta_bytes, other_weight_bytes
        sublayers = getattr(model_or_layer, "layers", None)
        if sublayers is not None:
            for l in sublayers:
                walk(l)
            return
        layer = model_or_layer
        if isinstance(layer, (QuantPointwiseConv2D, QuantDense)):
            wb, mb = layer.storage_bytes()
            int4_weight_bytes += wb
            int4_meta_bytes += mb
        elif layer.weights:
            for w in layer.weights:
                n_params = int(tf.size(w).numpy())
                other_weight_bytes += n_params * tf.as_dtype(w.dtype).size

    walk(model)

    total_bytes = int4_weight_bytes + int4_meta_bytes + other_weight_bytes
    return {
        "int4_weight_bytes": int4_weight_bytes,
        "int4_metadata_bytes": int4_meta_bytes,
        "other_weight_bytes": other_weight_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
    }