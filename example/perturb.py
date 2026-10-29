"""Input-sensitivity gate: perturbations defined by input name, for every model.

A fixture is useful only if its outputs move when its inputs change. Instead of
hand-written faults per family, each perturbation is defined once for an input
name (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, ...) and
applied to whatever inputs the contract passes to the model. The gate compares
the model's logits before and after each perturbation; labels are never
perturbed, and the model's own loss is not compared.
"""

import inspect

import torch


def roll_batch(x):
    """Give each row the next row's input: catches outputs that ignore their own input."""
    return x.roll(1, dims=0)


def reverse(x):
    return x.flip(-1)


def all_ones(x):
    return torch.ones_like(x)


def swap_channels(x):
    return x.flip(1)


def halve(x):
    return x * 0.5


PERTURBATIONS = {
    "input_ids": (roll_batch, reverse),
    "decoder_input_ids": (roll_batch, reverse),
    "attention_mask": (all_ones,),
    "decoder_attention_mask": (all_ones,),
    "token_type_ids": (all_ones,),
    "pixel_values": (roll_batch, swap_channels, reverse, halve),
    "input_features": (roll_batch, reverse, halve),
    "input_values": (roll_batch, reverse, halve),
    "state": (roll_batch, reverse, halve),
    # Time-series forecasters (TimesFM): one context series per row, possibly a
    # list of 1-D tensors of different lengths.
    "past_values": (roll_batch, reverse, halve),
}
# Training targets are never perturbed; future_values is the forecasting target.
NEVER = {"labels", "actions", "return_loss", "future_values"}


def per_row(perturb, rows):
    """Perturb a list of per-row tensors (variable-length contexts) like a batch."""
    if perturb is roll_batch:
        return rows[-1:] + rows[:-1]
    return [perturb(row) for row in rows]


def tensor_list(value):
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(v, torch.Tensor) for v in value)
    )


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return len(a) == len(b) and all(
        x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
    )


def logits_of(output):
    """The prediction tensors of a model output: every floating field named *logits*."""
    tensors = {k: v for k, v in output.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}
    picked = {k: v for k, v in tensors.items() if "logits" in k}
    return picked or {k: v for k, v in tensors.items() if k != "loss"}


def max_delta(a, b):
    """Largest change across output tensors. Matching infinities (e.g. -inf logits
    at masked text positions) are unchanged; an infinity that moves is an infinite
    change; a NaN anywhere stays NaN, so it fails the gate."""
    deltas = []
    for k in a:
        x, y = a[k].float(), b[k].float()
        if x.isnan().any() or y.isnan().any():
            return float("nan")
        delta = (x - y).abs().masked_fill(x == y, 0)
        deltas.append(float(torch.nan_to_num(delta, nan=float("inf"), posinf=float("inf")).max()))
    return max(deltas)


def sensitivity(model, batch, threshold, accepted=None):
    """Run every applicable perturbation; pass if each moves the logits by >= threshold.

    `accepted` maps perturbations the fixture is known to be blind to (by design
    of the architecture at fixture size) to the reviewed reason; they are still
    measured and reported, but do not fail the gate.
    """
    accepted = accepted or {}
    model.eval()
    parameter = next(model.parameters())
    with torch.no_grad(), torch.autocast(parameter.device.type, dtype=parameter.dtype, enabled=parameter.dtype in (torch.float16, torch.bfloat16)):
        base = logits_of(model(**batch))
        rows, undefined, ignored = {}, [], []
        # A tokenizer can return an input the model only absorbs in **kwargs
        # (DistilBERT's token_type_ids): it cannot move the logits, so do not gate on it.
        declared = inspect.signature(model.forward).parameters
        for name, value in batch.items():
            if name in NEVER or not (isinstance(value, torch.Tensor) or tensor_list(value)):
                continue
            if name not in declared:
                ignored.append(name)
                continue
            if name not in PERTURBATIONS:
                undefined.append(name)
                continue
            for perturb in PERTURBATIONS[name]:
                changed = perturb(value) if isinstance(value, torch.Tensor) else per_row(perturb, list(value))
                key = f"{name}:{perturb.__name__}"
                if same(changed, value):
                    rows[key] = None  # not applicable to these inputs (e.g. no padding)
                    continue
                rows[key] = max_delta(logits_of(model(**{**batch, name: changed})), base)
    applied = {k: v for k, v in rows.items() if v is not None and k not in accepted}
    weakest = min(applied, key=applied.get) if applied else None
    return {
        "pass": bool(applied) and all(v >= threshold for v in applied.values()),
        "accepted_insensitive": {k: {"delta": rows.get(k), "reason": r} for k, r in accepted.items()},
        "threshold": threshold,
        "deltas": rows,
        "weakest": weakest,
        "no_perturbation_defined": undefined,
        "not_a_forward_parameter": ignored,
    }
