"""Input-sensitivity gate: perturbations defined by input name, for every model.

A fixture is useful only if its outputs move when its inputs change. Instead of
hand-written faults per family, each perturbation is defined once for an input
name (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, ...) and
applied to whatever inputs the contract passes to the model. The gate compares
the model's logits before and after each perturbation; labels are never
perturbed, and the model's own loss is not compared.
"""

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
}
NEVER = {"labels", "actions", "return_loss"}


def logits_of(output):
    """The prediction tensors of a model output: every floating field named *logits*."""
    tensors = {k: v for k, v in output.items() if isinstance(v, torch.Tensor) and v.is_floating_point()}
    picked = {k: v for k, v in tensors.items() if "logits" in k}
    return picked or {k: v for k, v in tensors.items() if k != "loss"}


def max_delta(a, b):
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


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
        rows, undefined = {}, []
        for name, value in batch.items():
            if name in NEVER or not isinstance(value, torch.Tensor):
                continue
            if name not in PERTURBATIONS:
                undefined.append(name)
                continue
            for perturb in PERTURBATIONS[name]:
                changed = perturb(value)
                key = f"{name}:{perturb.__name__}"
                if torch.equal(changed, value):
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
    }
