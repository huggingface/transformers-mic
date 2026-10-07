"""Local workarounds for known Transformers bugs, applied by the runner before contracts run.

A framework's fleet lock lists the patches it needs under `local_patches`, with
the reason. The runner applies them in its own process and in every isolated
contract process, and records their names in each result's environment, so a
baseline recorded with a workaround says so. Remove a patch, and re-record the
affected baselines, once the fix is in the pinned Transformers build.
"""

import importlib
import inspect

# Audio models whose `masked_spec_embed` (SpecAugment's mask vector) has no
# `_init_weights` branch. Checkpoints that lack the tensor get uninitialized
# memory on load; HuBERT, SEW-D, SpeechT5, and Wav2Vec2-BERT already handle it.
MASKED_SPEC_EMBED_MODELS = {
    "wav2vec2": "Wav2Vec2PreTrainedModel",
    "wav2vec2_conformer": "Wav2Vec2ConformerPreTrainedModel",
    "data2vec": "Data2VecAudioPreTrainedModel",
    "wavlm": "WavLMPreTrainedModel",
    "unispeech": "UniSpeechPreTrainedModel",
    "unispeech_sat": "UniSpeechSatPreTrainedModel",
    "sew": "SEWPreTrainedModel",
}


def masked_spec_embed_init():
    """Initialize masked_spec_embed like HuBERT does: init.uniform_ in _init_weights."""
    import torch
    from transformers import initialization as init

    patched = []
    for model_type, class_name in MASKED_SPEC_EMBED_MODELS.items():
        module_name = f"transformers.models.{model_type}.modeling_{'data2vec_audio' if model_type == 'data2vec' else model_type}"
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        original = cls._init_weights
        if getattr(original, "_mic_patched", False):
            continue
        if "masked_spec_embed" in inspect.getsource(original):
            continue  # fixed upstream (#49386): a second init.uniform_ would draw different values

        def _init_weights(self, module, original=original):
            original(self, module)
            if isinstance(getattr(module, "masked_spec_embed", None), torch.nn.Parameter):
                init.uniform_(module.masked_spec_embed)

        _init_weights._mic_patched = True
        cls._init_weights = _init_weights
        patched.append(class_name)
    return patched


PATCHES = {"masked-spec-embed-init": masked_spec_embed_init}


def apply(names):
    """Apply the named patches; return {name: what was patched}."""
    unknown = sorted(set(names) - set(PATCHES))
    if unknown:
        raise SystemExit(f"unknown local patches: {unknown}")
    return {name: PATCHES[name]() for name in names}
