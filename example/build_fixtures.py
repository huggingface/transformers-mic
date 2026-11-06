"""Build fixtures for the contracts in a fleet lock, with the same recipe for every model.

    python example/build_fixtures.py                                   # every fixture with a build recipe
    python example/build_fixtures.py --fixture mic-whisper-tiny-slim
    python example/build_fixtures.py --keys-only                       # check fixture keys, build nothing

For each fixture in a contract's manifest that has a `build` recipe:

1. Fixture key. Every checkpoint the fixture covers must have the same key:
   its processor files byte for byte, its config after shrinking, and the
   parameters its weights lack. The build fails on a mismatch instead of
   testing a checkpoint through the wrong fixture.
2. Shrink. The real config goes through the generic rule from
   experiments/sweep.py (widths by field name in every sub-config, depth by
   layer pattern, MoE to 4 experts); nothing else changes, so dropout and
   SpecAugment stay as configured (contracts seed them). Seeded random weights,
   saved in the source checkpoint's dtype (float32 when its config names none),
   so the fixture loads in the dtype users of the checkpoint get by default.
   A source of SMALL_SOURCE parameters or fewer is not shrunk: the fixture is
   the checkpoint itself, config and trained weights unchanged (`weights:
   real`), since it costs no more than a fixture and tests trained behaviour.
3. Copy. Every non-weight file of the source checkpoint (tokenizer,
   processor, generation config) is copied unchanged. A recipe can explicitly
   include a processor asset with a weight-like extension and preserve selected
   frozen parameters or persistent buffers from the pinned checkpoint. Other
   weights are new. The weights mirror the real checkpoint's tensor names:
   parameters the real checkpoint lacks are dropped from the fixture too, so
   loading initializes them as it does for users, and the build checks that
   the fixture reports the same missing parameters.
4. Capture. The contract's own runnable markdown runs against the random
   fixture, and the runner records the arguments of each top-level model call
   by name. The training step's call is the batch used below.
5. Input-sensitivity gate (perturb.py). Perturbations are defined by input
   name, not per model. Random weights are kept if every perturbation moves
   the logits by at least 10x the strictest tolerance the fixture must hold.
   Otherwise the fixture is overfit on the contract's training batch and must
   then pass the gate; BatchNorm running statistics are recalibrated on the captured
   batch before evaluation, with dropout disabled. A fixture that does neither
   is a build failure. Real weights get the same gate and are never trained;
   failing it means the contract's inputs do not reach the model. A
   recipe can ask for overfitting anyway (`overfit`, with the reason), e.g.
   when random weights generate the same text for every prompt.
6. Numeric health. The saved fixture is reloaded in float32, bfloat16, and
   float16; outputs, loss, and gradients must be finite.

build.json records the key, the dtype, the gate results for random weights (and for the
overfit weights when training was needed), the training run, the health
checks, the copied files, and every file's sha256. The fleet lock pins that
hash map; the runner refuses fixtures whose files differ from it. Fixtures
without a recipe (mic-SmolLM-1.7B-slim, built by an earlier script) are left
alone.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys

import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Fixtures are built on the CPU, including the contract run that captures their
# inputs: some GPU kernels have no deterministic backward (CTC loss).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig, PreTrainedModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import run  # noqa: E402
from perturb import logits_of, sensitivity  # noqa: E402
from sweep import fit_rope_everywhere, shrink  # noqa: E402

WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".ot", ".onnx", ".tflite", ".ckpt", ".gguf", ".npz", ".pdparams", ".pkl", ".nemo")  # .nemo: NVIDIA NeMo archives beside the HF weights (Parakeet)
# Processor files are a few MB; anything this large is weights in a format WEIGHTS
# misses (gemma-2b ships a 9.5 GB .gguf, pegasus-xsum a 2.3 GB .pkl, which inflated the fixture and every
# CUDA baseline artifact carrying it).
MAX_COPIED_BYTES = 128 * 1024**2
NOT_COPIED = {"config.json", "README.md", ".gitattributes"}
# Config fields that name or date the source rather than describe the model.
VOLATILE = {"_name_or_path", "transformers_version", "dtype", "torch_dtype", "_commit_hash"}
GATE_FACTOR = 10
# Sources at or under this size are kept whole with their trained weights (the median fixture is ~9M).
SMALL_SOURCE = 20_000_000
DTYPES = ("float32", "bfloat16", "float16")


def config_insensitive(config, batch):
    """Perturbations the config makes inert by construction, with the reason.

    A config without token types (DeBERTa-v3: type_vocab_size 0) has no embedding
    for them, so a tokenizer's token_type_ids cannot move the logits.
    """
    inert = {}
    if "token_type_ids" in batch and getattr(config.get_text_config(), "type_vocab_size", None) == 0:
        inert["token_type_ids:all_ones"] = "type_vocab_size is 0: the model has no token type embedding"
    vision = getattr(config, "vision_config", None) or config
    if "pixel_values" in batch and getattr(vision, "model_type", None) == "pixtral":
        # Mistral-Large-4: delta 0.0 random and trained. Same shape as Wav2Vec2's input_values:halve.
        inert["pixel_values:halve"] = (
            "Pixtral's bias-free patch convolution feeds an RMSNorm (ln_pre), so scaling the pixel "
            "values cannot move the output"
        )
    return inert

# Two uint8 images of different colours differ by units after normalization; float images
# in [0, 1] that the processor rescales by 1/255 again differ by at most 1/255/std (~0.017:
# VideoMAE's torch.full(0.2) and torch.full(0.8) clips became -1.983 and -1.972).
MIN_ROW_DIFFERENCE = 0.05


def input_diagnosis(config, batch):
    """Why the contract's own inputs cannot move the outputs, when the gate fails.

    Training cannot fix these: they are contract bugs, so the build stops with the
    reason instead of overfitting. Diagnoses only; the gate thresholds are unchanged.
    """
    problems = []
    for name, value in batch.items():
        if not (name.startswith("pixel_values") and isinstance(value, torch.Tensor)
                and value.is_floating_point() and value.dim() > 1 and len(value) > 1):
            continue
        rows = value.float()
        difference = float((rows[1:] - rows[:1]).abs().max())
        if difference < MIN_ROW_DIFFERENCE:
            problems.append(
                f"the rows of {name} differ by at most {difference:.3g} after preprocessing, so the "
                "examples are nearly the same image: images or video frames given as floats in [0, 1] are "
                "rescaled by 1/255 again. Build each image or frame as an (H, W, 3) uint8 array with "
                "texture and a different colour per example"
            )
    token = next((getattr(config, k) for k in ("image_token_index", "image_token_id")
                  if isinstance(getattr(config, k, None), int)), None)
    ids = batch.get("input_ids")
    if token is not None and isinstance(ids, torch.Tensor) and isinstance(batch.get("pixel_values"), torch.Tensor):
        if not (ids == token).any():
            problems.append(
                f"input_ids carry no image token (id {token}), so the model never inserts the image "
                "features and pixel_values cannot move the logits (BLIP-2). Tokenize the text together "
                "with the images, processor(images=images, text=texts), and mask the image tokens in labels"
            )
    return problems


def report_phase(phase, detail=""):
    print("MIC_PROGRESS " + json.dumps({"phase": phase, "detail": detail}), flush=True)


def source_snapshot(repo, revision, recipe=None):
    """Config and processor files of a pinned checkpoint, without its weights."""
    try:
        ignore = [f"*{suffix}" for suffix in WEIGHTS] + run.WEIGHT_PATTERNS
        source = Path(snapshot_download(repo, revision=revision, ignore_patterns=ignore))
        if json.loads((source / "config.json").read_text()).get("model_type") == "qwen2_5_omni":
            # The full Omni loader needs this small speaker asset, even for text.
            from huggingface_hub import HfApi, hf_hub_download
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            asset = next(file for file in info.siblings if file.rfilename == "spk_dict.pt")
            if asset.size is None or asset.size > 1024**2:
                raise ValueError("Omni speaker asset exceeds the 1 MiB copy limit")
            hf_hub_download(repo, "spk_dict.pt", revision=revision)
    except Exception:
        cached = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        if (cached / "config.json").exists():
            source = cached
        else:
            raise
    for name in (recipe or {}).get("copy_assets", []):
        if Path(name).name != name or name in {".", "..", *NOT_COPIED} or name.startswith(("model.", "pytorch_model")):
            raise ValueError(f"Invalid processor asset name: {name}")
        if not (source / name).exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo, name, revision=revision)
        if not (source / name).is_file():
            raise ValueError(f"Pinned processor asset is unavailable: {name}")
    return source


def copied_files(source, recipe=None):
    assets = set((recipe or {}).get("copy_assets", []))
    names = sorted(
        f.relative_to(source).as_posix()
        for f in set(source.iterdir()) | set(run.sentence_transformer_files(source))
        if f.is_file()
        and f.relative_to(source).as_posix() not in NOT_COPIED
        and (not f.name.endswith(WEIGHTS) or f.name == "spk_dict.pt" or f.name in assets)
        and not f.name.endswith(".index.json")
    )
    for name in names:
        if (source / name).stat().st_size > MAX_COPIED_BYTES:
            raise ValueError(f"{name} is too large to be a processor file; add its suffix to WEIGHTS")
    return names


def model_class(source, recipe):
    if "model_class" in recipe:
        return getattr(transformers, recipe["model_class"])
    config = AutoConfig.from_pretrained(source)
    for name in config.architectures or []:
        cls = getattr(transformers, name, None)
        if cls is not None and isinstance(config, cls.config_class):
            return cls
    # Legacy cards can name removed classes (e.g. PI0Policy) or a head for a
    # different config family (Idefics3 on SmolVLM). Resolve the actual config.
    from sweep import primary_head
    _, name = primary_head(config)
    if not name:
        raise ValueError(f"No supported model head for {config.model_type}")
    return getattr(transformers, name)


def parameter_count(cls, config):
    with torch.device("meta"):
        return sum(p.numel() for p in cls._from_config(config).parameters())


def keeps_source(source, cls, config):
    """A small source is the fixture itself: same config, trained weights."""
    full = cls.config_class.from_pretrained(source)
    return parameter_count(cls, full) <= SMALL_SOURCE and parameter_count(cls, config) == parameter_count(cls, full)


# Widths taken from the model's tester, by source; and the sources whose build failed
# with them (a meta-device build does not run a forward: RT-DETR's tester
# encoder_hidden_dim built, then broke a convolution expecting 256 channels).
TESTER_SIZES, TESTER_OFF = {}, {}


def shrunk_config(source, cls, recipe=None):
    recorded = (recipe or {}).get("recorded_config")
    if recorded is not None:  # a rebuild reproduces the published config, not today's rule
        return cls.config_class.from_dict(recorded)
    config = cls.config_class.from_pretrained(source)
    original = config.to_dict()
    shrunk = parameter_count(cls, config) > SMALL_SOURCE
    if shrunk:
        config = shrink(config)
        if str(source) not in TESTER_OFF:
            config = adopt_tester_sizes(cls, original, config)
            # Provenance goes to build.json, never into the fixture's config.json.
            sizes = config.__dict__.pop("_mic_tester_sizes", None)
            if sizes:
                TESTER_SIZES[str(source)] = sizes
    overrides = (recipe or {}).get("config_overrides", {}).get("fields", {})
    if not overrides:
        return config
    values = config.to_dict()
    for dotted, value in overrides.items():
        *parents, field = dotted.split(".")
        target = values
        for key in parents:
            target = target[key]
        if field not in target:
            if value is None:
                continue  # dropping a field the shrunk config already lacks
            raise SystemExit(f"config_overrides: the shrunk config has no field {dotted}")
        target[field] = value
    if shrunk:
        # Overrides that resize a head (GLM-4.1V's text_config.hidden_size 128) leave the
        # rotary tables the rule fitted to its head; fit them again, as the rule would. A
        # head the overrides leave alone keeps its tables: the fit is the rule's own.
        fit_rope_everywhere(values, original)
    return type(config).from_dict(values)


def adopt_tester_sizes(cls, original, config):
    """Shrink the widths the generic rule does not know by name, as the model's tester does.

    The tester in tests/models/<model>/test_modeling_<model>.py names the size fields
    that matter for its model (DeepSeek-V4-Flash's index_head_dim, o_lora_rank;
    Qwen3-Next's linear_key_head_dim). Only fields the rule left at the checkpoint's value
    are taken, layer kinds stay with the rule, and the result must still build, else the
    rule's config is kept. Reviewed config_overrides apply afterwards and win.
    """
    from transformers_mic import tester

    url, found = tester.tester_values(original.get("model_type"))
    adopted = tester.unshrunk_sizes(found, original, config.to_dict()) if found else {}
    if not adopted:
        return config
    values = config.to_dict()
    for path, value in adopted.items():
        node = values
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
    try:
        candidate = type(config).from_dict(fit_rope_everywhere(values, original))
        with torch.device("meta"):
            cls._from_config(candidate)
    except Exception as error:
        print(f"tmic: model-tester sizes {sorted('.'.join(p) for p in adopted)} do not build ({type(error).__name__}); keeping the rule's", flush=True)
        return config
    candidate._mic_tester_sizes = {"source": url, "fields": {".".join(p): v for p, v in adopted.items()}}
    return candidate


def missing_parameters(cls, path):
    """Parameters a checkpoint's weights lack, in the model's own names, as loading reports them."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        if cls.__name__ == "Qwen2_5OmniForConditionalGeneration":
            # Its override assumes a model return and cannot unpack loading info.
            # Inspect tensor loading through the base method; the contract and
            # health checks still exercise the full loader and speaker asset.
            model, info = PreTrainedModel.from_pretrained.__func__(cls, path, output_loading_info=True)
        else:
            model, info = cls.from_pretrained(path, output_loading_info=True)
    del model
    return sorted(info["missing_keys"])


def source_dtype(source):
    """The dtype the checkpoint loads in by default (dtype="auto"): its config's, else float32."""
    config = json.loads((source / "config.json").read_text())
    return config.get("dtype") or config.get("torch_dtype") or "float32"


def builder_revision():
    """The transformers-mic commit whose shrink rule built the fixture ("+dirty" if edited)."""
    try:
        head = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", "."], stderr=subprocess.DEVNULL).returncode
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


def recorded_config(out, fixture):
    """The shrunk config a previous build of this exact recipe and source recorded, if any."""
    path = Path(out) / "build.json"
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    if previous.get("recipe") != fixture["build"] or previous.get("built_from", {}).get("repo") != fixture["built_from"]:
        return None  # an edited recipe (or a pre-provenance record) re-shrinks with the current rule
    return previous.get("key_fields")


def fixture_key(source, cls, missing, recipe=None, frozen_hashes=None):
    config = {k: v for k, v in shrunk_config(source, cls, recipe).to_dict().items() if k not in VOLATILE}
    processor = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in copied_files(source, recipe)}
    fields = {"config": config, "processor": processor, "missing": missing}
    if frozen_hashes is not None:
        fields["frozen_parameters"] = frozen_hashes
    dtype = source_dtype(source)
    if dtype != "float32":  # float32 is implied, so the keys of float32 fixtures stay as built
        fields["dtype"] = config["dtype"] = dtype
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest(), config


def tensor_tree(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {key: tensor_tree(item, function) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(item, function) for item in value)
    return value


def as_dtype(batch, dtype):
    return tensor_tree(batch, lambda value: value.to(dtype) if value.is_floating_point() else value)


def restore_frozen(model, source, repo, revision, recipe):
    """Copy selected frozen parameters and persistent buffers from the pinned checkpoint."""
    prefixes = recipe.get("preserve_frozen_prefixes", [])
    if not prefixes:
        return None
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    selected = {}
    for name, parameter in model.state_dict(keep_vars=True).items():
        if any(name.startswith(prefix) for prefix in prefixes):
            if parameter.requires_grad:
                raise ValueError(f"Requested fixed tensor is trainable: {name}")
            selected[name] = parameter
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in selected):
            raise ValueError(f"No frozen parameters or persistent buffers match {prefix!r}")
    from transformers_mic.inspection import explicit_weights

    explicit = explicit_weights(source)
    sharded = explicit is None or explicit.endswith(".index.json")
    single = "model.safetensors" if sharded else explicit
    index = source / (explicit if explicit and sharded else "model.safetensors.index.json")
    weight_map = json.loads(index.read_text())["weight_map"] if sharded and index.exists() else {}
    files = {weight_map.get(name, single) for name in selected}
    hashes = {}
    for filename in sorted(files):
        path = hf_hub_download(repo, filename, revision=revision)
        with safe_open(path, framework="pt", device="cpu") as checkpoint, torch.no_grad():
            for name, parameter in selected.items():
                if weight_map.get(name, single) != filename:
                    continue
                if name not in checkpoint.keys():
                    raise ValueError(f"Pinned checkpoint lacks fixed tensor {name}")
                tensor = checkpoint.get_tensor(name)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"Fixed tensor {name} has shape {tuple(tensor.shape)}, expected {tuple(parameter.shape)}")
                parameter.copy_(tensor)
                hashes[name] = hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return hashes


def checkpoint_conversions(model, keys):
    """The weight transforms from_pretrained uses for these checkpoint names.

    Loading keeps exactly the transforms some checkpoint key matched, and
    save_pretrained reverts them; attached to the fixture, they make it keep the
    checkpoint's names (LLaVA's legacy vision_tower.vision_model, Qwen3.5-MoE's
    fused experts), so loading the fixture runs the same converters.
    """
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    mapping = get_model_conversion_mapping(model)
    renamings = [t for t in mapping if isinstance(t, WeightRenaming)]
    converters = [t for t in mapping if isinstance(t, WeightConverter)]
    state = model.state_dict()
    for key in sorted(keys):
        rename_source_key(key, renamings, converters, model.base_model_prefix, state)
    return [t for t in mapping if t.was_used()]


def in_fixture(missing, names):
    """The real checkpoint's missing parameters that the shrunk fixture has.

    A shrunk fixture keeps fewer layers: the real checkpoint's missing names in
    the layers it dropped (A.X-K1's e_score_correction_bias in layers 2-60) are
    left out, as long as the same name is missing in a layer the fixture keeps, or
    the fixture holds that tensor in its kept layers (Kimi K2.5's vision tower lacks
    layer 2's fc2, which the two kept layers have). Any other absent name means the
    fixture cannot mirror the checkpoint.
    """
    def pattern(name):
        return re.sub(r"\.\d+(?=\.)", ".*", name)

    kept = [name for name in missing if name in names]
    patterns = {pattern(name) for name in kept} | {pattern(name) for name in names}
    absent = [name for name in missing if name not in names and pattern(name) not in patterns]
    if absent:
        raise SystemExit(f"cannot mirror the real checkpoint: the fixture has no tensor named {absent}")
    return kept


def save(model, out, source, missing, recipe=None, checkpoint=None):
    """Write the fixture: new config and weights without the parameters the real
    checkpoint lacks, every processor file copied unchanged.

    `checkpoint` ({"keys", "binary"}) is the real checkpoint's serialization: the
    fixture keeps its tensor names, base-model prefix and expert layout. The file is
    always model.safetensors, even for a pytorch_model.bin source: the names are what
    the loader's conversions read, and a .bin container tests nothing more.
    """
    from safetensors.torch import load_file, save_file
    from transformers_mic.fixture_validation import sync_dtype, validate_model, validate_saved_config

    sync_dtype(model)
    validate_model(model)
    out.mkdir(parents=True, exist_ok=True)
    for file in out.iterdir():
        if file.is_file() and file.name != "README.md":
            file.unlink()
    state = model.state_dict()
    missing = in_fixture(missing, state)
    if checkpoint:
        model._weight_conversions = checkpoint_conversions(model, checkpoint["keys"])
    # A tied alias (BERT-style cls.predictions.decoder.bias) is never serialized.
    model.save_pretrained(out, state_dict={k: v for k, v in state.items() if k not in missing})
    weights = out / "model.safetensors"
    tensors = load_file(weights)
    prefix = f"{model.base_model_prefix}." if model.base_model_prefix else None
    if (checkpoint and prefix and checkpoint["keys"]
            and all(k.startswith(prefix) for k in checkpoint["keys"])
            and not any(k.startswith(prefix) for k in tensors)):
        # A base model loaded from a head checkpoint (Jina's roberta.*): the loader strips the prefix.
        tensors = {prefix + k: v for k, v in tensors.items()}
    tied = getattr(model, "all_tied_weights_keys", None) or {}
    if checkpoint and checkpoint["keys"] and isinstance(tied, dict):
        for alias, kept in tied.items():
            # A tied pair is serialized once; keep the name the source uses
            # (OPT's .bin stores lm_head.weight, save_pretrained keeps embed_tokens).
            if kept in tensors and kept not in checkpoint["keys"] and alias in checkpoint["keys"] and alias not in tensors:
                tensors[alias] = tensors.pop(kept)
    if checkpoint:
        from transformers_mic.fixture_quantization import quantize

        tensors, quantization = quantize(tensors, checkpoint.get("quantization_config"), checkpoint["keys"])
        if quantization:
            config_path = out / "config.json"
            saved_config = json.loads(config_path.read_text())
            saved_config["quantization_config"] = quantization
            config_path.write_text(json.dumps(saved_config, indent=2) + "\n")
    validate_saved_config(model, out / "config.json")
    save_file(tensors, weights, metadata={"format": "pt"})
    copied = copied_files(source, recipe)
    for name in copied:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, out / name)  # overwrites the generation config save_pretrained wrote
    return copied


@contextlib.contextmanager
def captured_calls(model_class=PreTrainedModel, method="__call__"):
    """Capture calls to the selected model class method."""
    from functools import wraps

    calls, depth = [], [0]
    call = getattr(model_class, method)

    @wraps(call)
    def recording(self, *args, **kwargs):
        if depth[0] == 0:
            named = dict(kwargs)
            if args:  # model(pixel_values) passes inputs positionally; replay needs their names
                try:
                    bound = inspect.signature(self.forward).bind_partial(*args, **kwargs).arguments
                except TypeError:
                    bound = {}
                extra = {k for k, p in inspect.signature(self.forward).parameters.items() if p.kind == p.VAR_KEYWORD}
                named = {k: v for k, v in bound.items() if k not in extra}
                for key in extra:
                    named.update(bound.get(key, {}))
            calls.append(
                {
                    "training": self.training,
                    "class": type(self).__name__,
                    "kwargs": tensor_tree(named, lambda value: value.detach().clone()),
                }
            )
        depth[0] += 1
        try:
            return call(self, *args, **kwargs)
        finally:
            depth[0] -= 1

    with patch.object(model_class, method, recording):
        yield calls


def contract_lines(stderr, code, entry):
    """The contract lines a failing traceback passed through, as the code ran them.

    The traceback's tail is all library frames when the error is raised deep in a model
    (moonshine-ai/moonshine-streaming-tiny: "You must specify exactly one of input_ids or
    inputs_embeds" from the decoder), so neither a reader nor a contract repair could tell
    which call was wrong; the repair returned the usage unchanged. The code is compiled
    under the entry's file name, so a frame's line number indexes the executed code, not
    the Markdown file.
    """
    lines = code.splitlines()
    found = []
    for match in re.finditer(r'File "' + re.escape(str(entry)) + r'", line (\d+)', stderr):
        number = int(match.group(1))
        if 0 < number <= len(lines):
            found.append(f"contract line {number}: {lines[number - 1].strip()}")
    return "".join(f"{line}\n" for line in dict.fromkeys(found))


def contract_batch(entry, fixture_dir, *, model_class=PreTrainedModel, method="__call__"):
    """Run the contract on the fixture; return the training step's model inputs by name."""
    code = run.contract_code(entry)
    env = {"MIC_RUNNER": "1", "MIC_MODEL": str(fixture_dir), "MIC_MODEL_REVISION": "", "MIC_MODEL_KWARGS": "{}"}
    with captured_calls(model_class, method) as calls, run.chdir(entry.parent):
        ok, _, stderr = run.execute_inline(code, str(entry), env)
    if not ok:
        raise SystemExit(f"the contract fails on the random fixture:\n{contract_lines(stderr, code, entry)}{stderr[-1500:]}")
    training = [c for c in calls if c["training"]]
    chosen = training[0] if training else calls[0]
    # Fixtures are built on the CPU; the contract may have put its inputs on a GPU.
    batch = {k: tensor_tree(v, lambda tensor: tensor.detach().cpu().clone()) for k, v in chosen["kwargs"].items()
             if isinstance(v, (torch.Tensor, bool, list, tuple, dict))}
    return batch, {"call": "training step" if training else "first inference call", "inputs": sorted(batch), "class": chosen["class"]}


def overfit(model, batch, max_steps, target_loss=0.05):
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    stable = 0
    targets = None
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = getattr(output, "loss", None)
        if loss is None:
            # Models without a built-in loss (SAM2): fit the predictions the gate
            # measures to fixed, distinct targets per example, as labels would.
            # A non-negative output (VitMatte's sigmoid alphas, DPT's ReLU depth)
            # cannot reach normal targets.
            predictions = logits_of(output)
            if targets is None:
                generator = torch.Generator().manual_seed(17)
                targets = {
                    k: (torch.rand if v.min() >= 0 else torch.randn)(v.shape, generator=generator).to(v.device, v.dtype)
                    for k, v in predictions.items()
                }
            loss = sum(torch.nn.functional.mse_loss(v, targets[k]) for k, v in predictions.items())
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            report_phase("Fit fixture weights", f"step {step}/{max_steps} max · loss {loss.item():.4f}")
            stable = stable + 1 if loss.item() < target_loss else 0
            if stable == 2:
                break
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"seconds": round(time.perf_counter() - start, 2), "steps": step, "loss": loss.item(), "converged": stable == 2}


def unit_batch_norm(model):
    """Reset BatchNorm scales drawn near zero to PyTorch's unit default; return how many.

    VitMatte initializes every BatchNorm weight from N(0, 0.02), so each conv block
    scales activations and gradients by ~0.02: outputs sit at sigmoid(0) and 500
    overfit steps cannot move them. Used only before overfitting a failing fixture.
    """
    norms = [module for module in model.modules()
             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
             and module.affine and module.weight.abs().mean() < 0.1]
    with torch.no_grad():
        for module in norms:
            module.weight.fill_(1.0)
            module.bias.zero_()
    return len(norms)


def calibrate_batch_norm(model, batch):
    """Set inference statistics from the captured batch after fixture overfitting.

    Training loss can converge before the default exponential moving averages
    approach the tiny activations of a slim randomly initialized CNN. Use the
    batch's measured statistics, with dropout disabled, before evaluating gates.
    This updates buffers only; all sensitivity thresholds remain unchanged.
    """
    norms = [module for module in model.modules()
             if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
             and module.track_running_stats]
    if not norms:
        return 0
    model.eval()
    momenta = [module.momentum for module in norms]
    try:
        for module in norms:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        with torch.no_grad():
            model(**batch)
    finally:
        for module, momentum in zip(norms, momenta):
            module.momentum = momentum
            module.eval()
    return len(norms)


def called_module(model, captured):
    """Probe the module called by the contract, including composite heads."""
    name = captured.get("class", type(model).__name__)
    if type(model).__name__ == name:
        return model
    matches = [module for module in model.modules() if type(module).__name__ == name]
    if len(matches) != 1:
        raise ValueError(f"Captured {name}: expected one matching module, found {len(matches)}")
    return matches[0]


def masked_like(mask, reference, tolerance=0.01):
    return mask.shape == reference.shape and float((mask != reference).float().mean()) <= tolerance


def numeric_health(cls, folder, batch, captured=None):
    """Reload the saved fixture per dtype; forward and backward must stay finite.

    Some models put infinities in their outputs on purpose (-inf logits at padded
    text positions, +inf coordinates for masked proposals). Non-finite values are
    healthy only where float32 has them too, up to 1% of elements (a train-mode
    forward is not bit-identical across dtypes, so a value-dependent mask can shift
    slightly; an overflow is not that small); NaN never is. Unscaled float16
    gradients are reported but do not gate: float16 training needs a loss scaler.
    """
    config_path = Path(folder) / "config.json"
    source_quantization = json.loads(config_path.read_text()).get("quantization_config") if config_path.exists() else None
    rows, structural = {}, {}
    for name in DTYPES:
        dtype = getattr(torch, name)
        row = {}
        try:
            loaded = cls.from_pretrained(folder, dtype=dtype)
            quantizer = getattr(loaded, "hf_quantizer", None)
            if (source_quantization and not getattr(loaded, "is_quantized", False)) or (
                quantizer and getattr(quantizer.quantization_config, "dequantize", False)
            ):
                # FP8 dequantization may retain the checkpoint dtype on converted
                # projections. Test the requested compute dtype explicitly, and
                # record this distinction from raw from_pretrained behavior.
                row["loaded_parameter_dtypes"] = sorted({str(p.dtype) for p in loaded.parameters()})
                loaded = loaded.to(dtype=dtype)
                row["cast_after_dequantization"] = True
            model = called_module(loaded, captured or {}).train()
            with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
                output = model(**as_dtype(batch, dtype))
            logits = logits_of(output)
            masks = {k: ~torch.isfinite(t) for k, t in logits.items()}
            if dtype == torch.float32:
                structural = masks
            row["outputs_finite"] = all(
                not bool(t.isnan().any())
                and bool(torch.isfinite(t).any())
                and (k not in structural or masked_like(masks[k], structural[k]))
                for k, t in logits.items()
            )
            if output.get("loss") is not None:
                output.loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                row["loss_finite"] = bool(torch.isfinite(output.loss))
                finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
                row["gradients_finite" if dtype != torch.float16 else "unscaled_gradients_finite"] = finite
            row["finite"] = all(v for k, v in row.items() if k.endswith("_finite") and not k.startswith("unscaled"))
        except Exception as exc:  # noqa: BLE001 - recorded, then fails the build below
            row = {"finite": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows[name] = row
    return rows


def depth(config, prefix=""):
    """Layer counts of a shrunk config dict and its sub-configs, for the README."""
    keys = ("num_hidden_layers", "num_layers", "num_decoder_layers", "encoder_layers", "decoder_layers")
    parts = [f"{prefix}{k} {config[k]}" for k in keys if isinstance(config.get(k), int)]
    for key, value in config.items():
        if isinstance(value, dict) and key.endswith("_config"):
            parts.append(depth(value, f"{key.removesuffix('_config')} "))
    return ", ".join(p for p in parts if p)


def readme(name, contract, build, cls, config, contract_repo=None):
    contract_repo = run.canonical_repo(contract_repo or f"hf-internal-testing/{contract}-integration")
    lacks = build.get("missing_parameters", [])
    mirror = (
        f"The real checkpoint lacks {', '.join(f'`{n}`' for n in lacks)}, so the fixture omits it too, and loading initializes it as it does for users."
        if lacks
        else "The real checkpoint has every parameter, and so does the fixture."
    )
    covers = ", ".join(f"[`{r}`](https://huggingface.co/{r})" for r in build["covers"])
    gate = build["gate"].get("random") or build["gate"].get("real")
    if build["weights"] == "real":
        weights = (
            f"**The checkpoint's own trained weights**, unchanged: at {build['parameters'] / 1e6:.1f}M parameters it is "
            f"no bigger than a slim fixture. They passed the input-sensitivity gate (weakest: `{gate['weakest']}`, "
            f"{gate['deltas'][gate['weakest']]:.3g})."
        )
    elif build["weights"] == "random":
        weights = (
            f"**Random weights (seed {build['seed']}).** They passed the input-sensitivity gate: every perturbation "
            f"of the contract's inputs moved the logits by at least {gate['threshold']:g} (weakest: `{gate['weakest']}`, "
            f"{gate['deltas'][gate['weakest']]:.3g}), so no training was needed."
        )
    elif gate["pass"]:
        weights = (
            f"**Overfit to the contract's training step** ({build['training']['steps']} steps), although random weights "
            f"passed the input-sensitivity gate: {recipe_reason(build)}"
        )
    else:
        weights = (
            f"**Overfit to the contract's training step** ({build['training']['steps']} steps). Random weights failed "
            f"the input-sensitivity gate: `{gate['weakest']}` moved the logits by only {gate['deltas'][gate['weakest']]:.2g} "
            f"(threshold {gate['threshold']:g})."
        )
    shape = (
        f"**The checkpoint's own config, not shrunk:** `{cls.__name__}` at {build['parameters'] / 1e6:.1f}M parameters is at or under the {SMALL_SOURCE // 10**6}M below which a source is kept whole."
        if build.get("shrink") == "none"
        else f"**Same architecture, smaller config:** `{cls.__name__}`, shrunk by the generic rule in [`sweep.py`](https://github.com/huggingface/model-integration-contracts/blob/main/experiments/sweep.py) ({depth(config)})."
    )
    preserved = build.get("preserved_frozen_sha256", {})
    fixed = (
        f"- **Fixed tensors preserved** from the pinned checkpoint: {', '.join(f'`{n}`' for n in preserved)}.\n"
        if preserved else ""
    )
    return f"""---
tags:
  - mic
  - test-artifact
base_model: {build['built_from']['repo']}
---

# Slim fixture: {short(name)}

A {build['parameters'] / 1e6:.1f}M-parameter fixture for the [{contract} MIC](https://huggingface.co/{contract_repo}), covering {covers}. It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

- **Fixture key `{build['key'][:12]}`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- {shape}
- **Processor files copied byte for byte** from `{build['built_from']['repo']}`: {', '.join(f'`{f}`' for f in build['copied_unchanged'])}.
- {weights}
- **Same tensor names as the real checkpoint.** {mirror}
{fixed}- **Saved in {build['dtype']}**, the dtype `{build['built_from']['repo']}` loads in by default.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](https://github.com/huggingface/transformers-mic/blob/main/example/build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. The source repository commits only `README.md` and `build.json`; rebuild the other files with `python example/build_fixtures.py --fixture {short(name)}`.
"""


def recipe_reason(build):
    reason = build["overfit_reason"]
    return reason if reason.endswith(".") else reason + "."


def short(repo):
    return repo.split("/")[-1]


def environment():
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


MISSING_PACKAGE = re.compile(
    r"^(?:ModuleNotFoundError: .*|ImportError:.*)$|requires the \S+ library but it was not found", re.MULTILINE
)


def missing_package(error):
    """The sentence naming a package this environment lacks, or None.

    Contract failures arrive as SystemExit carrying the child's traceback text, so the
    exception chain alone does not say it: read the text too.
    """
    seen = error
    while seen is not None and not isinstance(seen, ImportError):
        seen = seen.__cause__ or seen.__context__
    text = str(seen if seen is not None else error)
    found = re.search(r"\S[^\n]*requires the \S+ library but it was not found[^.\n]*\.?", text)
    if found:
        return found.group(0).strip()
    if seen is None and not MISSING_PACKAGE.search(text):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return next((line for line in reversed(lines) if "No module named" in line), lines[-1] if lines else type(error).__name__)


def build_fixture(*args, **kwargs):
    """Build; if it fails with model-tester sizes, rebuild once with the shrink rule's."""
    try:
        return _build_fixture(*args, **kwargs)
    except (SystemExit, Exception) as error:
        adopted = [source for source in TESTER_SIZES if source not in TESTER_OFF]
        missing = missing_package(error)
        if missing:
            # Other sizes would fail the same way: say what to install instead of retrying.
            print(error, file=sys.stderr, flush=True)
            raise SystemExit(f"ImportError: {missing} Install it and rerun; this is the environment, not the fixture.") from None
        if not adopted:
            raise
        reason = (str(error).strip().splitlines() or [type(error).__name__])[-1][:300]
        print(f"tmic: the build failed with model-tester sizes ({reason}); rebuilding with the shrink rule's", flush=True)
        for source in adopted:
            TESTER_OFF[source] = {"fields": TESTER_SIZES.pop(source)["fields"], "build_failed": reason}
        return _build_fixture(*args, **kwargs)


def source_weights_file(source, binary):
    """The weights file (or index) the source checkpoint loads from."""
    from transformers_mic.inspection import explicit_weights

    explicit = explicit_weights(source)
    if explicit:
        return explicit
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (Path(source) / name).exists():
            return name
    return "pytorch_model.bin" if binary else "model.safetensors"


def saved_keys(folder):
    """Tensor names as the fixture serializes them (the file a loader reads)."""
    folder = Path(folder)
    if (folder / "model.safetensors").exists():
        from safetensors import safe_open

        with safe_open(folder / "model.safetensors", framework="pt") as weights:
            return sorted(weights.keys()), "model.safetensors"
    state = torch.load(folder / "pytorch_model.bin", map_location="meta", weights_only=True)
    return sorted(state), "pytorch_model.bin"


def review_fixture(name, fixture, revisions, out, artifact=None, inspect_checkpoint=None):
    """Write the shrunk config and seeded random weights for review, running no contract code.

    The contract (usage.md) has not been reviewed yet, so this stops where the
    build starts running it: the config, tensor names and file format are final
    here; only the weight values change once the build gates them. With
    `artifact`, describe that already-built fixture instead of shrinking again.
    Returns what a reviewer compares against the source checkpoint.
    """
    recipe = fixture["build"]
    source_repo = fixture["built_from"]
    revision = revisions[source_repo]
    report_phase("Inspect checkpoint", source_repo)
    source = source_snapshot(source_repo, revision, recipe)
    cls = model_class(source, recipe)
    from transformers_mic import inspection

    source_keys = sorted(inspection.checkpoint_keys(source_repo, revision, source))
    if artifact is None:
        lacks = (inspect_checkpoint(cls, source, source_repo, revision) if inspect_checkpoint
                 else missing_parameters(cls, run.checkpoint_path(source_repo, revision)))
        checkpoint = {
            "keys": set(source_keys),
            "binary": inspection.loads_binary(source_repo, revision, source),
            "quantization_config": getattr(cls.config_class.from_pretrained(source), "quantization_config", None),
        }
        source_file = source_weights_file(source, checkpoint["binary"])
        dtype = getattr(torch, source_dtype(source))
        real = keeps_source(source, cls, shrunk_config(source, cls, recipe))
        if real:
            report_phase("Load the small source checkpoint unchanged")
            model = cls.from_pretrained(run.checkpoint_path(source_repo, revision), dtype=dtype).eval()
        else:
            report_phase("Initialize fixture")
            torch.manual_seed(recipe["seed"])
            model = cls._from_config(shrunk_config(source, cls, recipe), dtype=torch.float32).eval().to(dtype)
            restore_frozen(model, source, source_repo, revision, recipe)
        save(model, out, source, lacks, recipe, checkpoint)
        del model
        weights = "real" if real else "random (seeded; the build checks and may fit them)"
        build = {"model_class": cls.__name__, "shrink": "none" if real else "rule", "recipe": recipe}
    else:
        build = json.loads((Path(artifact) / "build.json").read_text())
        lacks = build.get("missing_parameters", [])
        source_file = (build.get("serialization") or {}).get("source_file") or source_weights_file(
            source, inspection.loads_binary(source_repo, revision, source)
        )
        weights = build.get("weights")
    report_phase("Compare with the source checkpoint")
    from transformers_mic import shrink_view

    report = shrink_view.analyze({
        "fixture": name,
        "source": f"{source_repo}@{revision}",
        "build": build,
        "original": shrink_view.config_dict(source),
        "shrunk": shrink_view.config_dict(out, cls.config_class.model_type),
    })
    keys, weights_file = saved_keys(out)
    return {
        "fixture": name,
        "source": {"repo": source_repo, "revision": revision},
        "model_class": cls.__name__,
        "weights": weights,
        "weights_file": weights_file,
        "source_weights_file": source_file,
        "files": sorted(f.name for f in Path(out).iterdir() if f.is_file()),
        "config": json.loads((Path(out) / "config.json").read_text()),
        "source_keys": source_keys,
        "fixture_keys": keys,
        "missing_parameters": lacks,
        "parameters": report.get("parameters"),
        "config_changes": report["config_changes"],
        "config_unchanged": report["config_unchanged"],
        "findings": shrink_view.verdict(report),
        "tester_notes": (report.get("tester") or {}).get("findings") or [],
        "page": shrink_view.render(report),
    }


def _build_fixture(name, fixture, contract_name, manifest, root, revisions, out, keep_random=False, inspect_checkpoint=None, contract_repo=None, reshrink=False):
    if fixture["build"].get("sentence_transformers_class"):
        from transformers_mic import sentence_transformers

        return sentence_transformers.build(
            sys.modules[__name__], name, fixture, contract_name, manifest, root, revisions, out,
            keep_random=keep_random,
        )
    recorded = None if reshrink else recorded_config(out, fixture)
    previous_tester = None  # save() clears `out`, so read the old record now
    if recorded is not None and (Path(out) / "build.json").exists():
        previous_tester = json.loads((Path(out) / "build.json").read_text()).get("tester_sizes")
    recipe = {**fixture["build"], "recorded_config": recorded} if recorded is not None else fixture["build"]
    keys, missing = {}, {}
    for repo in fixture["covers"]:
        report_phase("Inspect checkpoint", repo)
        source = source_snapshot(repo, revisions[repo], recipe)
        cls = model_class(source, recipe)
        missing[repo] = (inspect_checkpoint(cls, source, repo, revisions[repo]) if inspect_checkpoint
                         else missing_parameters(cls, run.checkpoint_path(repo, revisions[repo])))
        frozen_hashes = None
        if recipe.get("preserve_frozen_prefixes"):
            frozen_model = cls._from_config(shrunk_config(source, cls, recipe))
            frozen_hashes = restore_frozen(frozen_model, source, repo, revisions[repo], recipe)
            del frozen_model
        keys[repo] = fixture_key(source, cls, missing[repo], recipe, frozen_hashes)
    if len({key for key, _ in keys.values()}) != 1:
        raise SystemExit(f"{name}: covered checkpoints have different fixture keys: { {r: k[:12] for r, (k, _) in keys.items()} }")
    key, config = next(iter(keys.values()))
    source_repo = fixture["built_from"]
    lacks = missing[source_repo]
    source = source_snapshot(source_repo, revisions[source_repo], recipe)
    from transformers_mic import inspection
    checkpoint = {
        "keys": inspection.checkpoint_keys(source_repo, revisions[source_repo], source),
        "binary": inspection.loads_binary(source_repo, revisions[source_repo], source),
        "quantization_config": getattr(cls.config_class.from_pretrained(source), "quantization_config", None),
    }
    cls = model_class(source, recipe)
    tolerance = recipe.get("gate_tolerance") or max(
        r["atol"] for r in manifest["outputs"].values() if r["compare"] == "allclose"
    )
    threshold = GATE_FACTOR * tolerance
    dtype = getattr(torch, source_dtype(source))

    real = keeps_source(source, cls, shrunk_config(source, cls, recipe))
    if real:
        report_phase("Load the small source checkpoint unchanged")
        model = cls.from_pretrained(run.checkpoint_path(source_repo, revisions[source_repo]), dtype=dtype).eval()
        frozen_hashes = None
    else:
        report_phase("Initialize fixture")
        torch.manual_seed(recipe["seed"])
        # Initialized in float32 like every other fixture, then cast: the seed gives the same weights in any dtype.
        model = cls._from_config(shrunk_config(source, cls, recipe), dtype=torch.float32).eval().to(dtype)
        frozen_hashes = restore_frozen(model, source, source_repo, revisions[source_repo], recipe)
    copied = save(model, out, source, lacks, recipe, checkpoint)
    entry = root / manifest["integrations"]["transformers"]["entry_point"]
    report_phase("Run contract and capture training inputs")
    batch, captured = contract_batch(entry, out)
    accepted = {**config_insensitive(model.config, batch), **recipe.get("accept_insensitive", {})}
    weights, training = ("real" if real else "random"), None
    report_phase(f"Check {weights}-weight input sensitivity")
    gate = {weights: sensitivity(called_module(model, captured), as_dtype(batch, dtype), threshold, accepted)}
    unmeasured = gate[weights]["no_perturbation_defined"]
    if unmeasured and not keep_random and not any(v is not None for v in gate[weights]["deltas"].values()):
        # Overfitting cannot help a gate that perturbs nothing, nor can a usage edit.
        raise SystemExit(
            f"{name}: no input perturbation is defined for the contract's inputs {unmeasured}; "
            "add them to PERTURBATIONS in example/perturb.py (a tmic gap, not a contract error)"
        )
    diagnosis = [] if gate[weights]["pass"] else input_diagnosis(model.config, batch)
    if diagnosis:
        raise SystemExit(
            f"{name}: the contract's inputs cannot move the outputs, and overfitting cannot fix that: "
            + " ".join(f"({n}) {p}." for n, p in enumerate(diagnosis, 1)) + f" Gate: {gate[weights]}"
        )
    if real and not gate["real"]["pass"]:
        raise SystemExit(f"{name}: the trained weights are insensitive to the contract's inputs: {gate['real']}")
    if not real and not keep_random and (not gate["random"]["pass"] or recipe.get("overfit")):
        torch.manual_seed(recipe["seed"])  # dropout and SpecAugment stay on while training
        model = model.float()  # trained in float32, saved in the source dtype
        unit_scales = unit_batch_norm(model)
        training = overfit(called_module(model, captured), as_dtype(batch, torch.float32), recipe.get("max_steps", 500), recipe.get("target_loss", 0.05))
        report_phase("Calibrate BatchNorm statistics")
        calibrated = calibrate_batch_norm(called_module(model, captured), as_dtype(batch, torch.float32))
        if calibrated:
            training["batch_norm_calibration"] = {"layers": calibrated, "batches": 1}
        if unit_scales:
            training["batch_norm_unit_scale"] = unit_scales
        model = model.to(dtype)
        report_phase("Verify trained fixture sensitivity")
        gate["overfit"] = sensitivity(called_module(model, captured), as_dtype(batch, dtype), threshold, accepted)
        if not training["converged"] or not gate["overfit"]["pass"]:
            # Zero-initialized gates (Mllama) give 0.0 at random weights; still 0.0 after
            # training means the contract's call never routes that input to the outputs.
            unrouted = sorted(k for k, v in gate["overfit"]["deltas"].items() if v == 0.0 and k not in accepted)
            hint = (f" {', '.join(unrouted)} moved nothing even after training: the contract's model call does "
                    "not route that input to the outputs." if unrouted else "")
            raise SystemExit(f"{name}: random weights fail the gate and overfitting did not fix it:{hint} {training}, {gate}")
        weights = "overfit"
        save(model, out, source, lacks, recipe, checkpoint)
    mirrored = missing_parameters(cls, out)
    if mirrored != in_fixture(lacks, model.state_dict()):
        raise SystemExit(f"{name}: fixture lacks {mirrored}, the real checkpoint lacks {lacks}")
    report_phase("Check float32, bfloat16 and float16 numeric health")
    health = numeric_health(cls, out, batch, captured)
    # A model that cannot run a half dtype (DeBERTa's mask fill overflows, FNet's FFT
    # rejects it on CPU) declares it; the row is still recorded in build.json.
    unhealthy = recipe.get("accept_unhealthy", {})
    if not all(row["finite"] for dtype, row in health.items() if dtype not in unhealthy):
        raise SystemExit(f"{name}: numeric health check failed: {health}")
    # Widths taken from the model's tester; a rebuild from the recorded config keeps the record.
    shrunk_config(source, cls, recipe)  # records TESTER_SIZES for this source
    tester_sizes = (TESTER_SIZES.get(str(source))
                    or ({"rejected": TESTER_OFF[str(source)]} if str(source) in TESTER_OFF else None)
                    or previous_tester)
    build = {
        "fixture": name,
        "key": key,
        "key_fields": config,
        "covers": {repo: revisions[repo] for repo in fixture["covers"]},
        "built_from": {"repo": source_repo, "revision": revisions[source_repo]},
        "seed": recipe["seed"],
        "dtype": str(dtype).removeprefix("torch."),
        "builder": "example/build_fixtures.py",
        "builder_revision": builder_revision(),
        "recipe": fixture["build"],
        "shrink": "none" if real else "recorded" if recorded is not None else "rule",
        "model_class": cls.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "missing_parameters": lacks,
        "serialization": {
            "file": "model.safetensors",
            "source_file": source_weights_file(source, checkpoint["binary"]),
            "reverted_conversions": [
                f"{type(t).__name__}: {t.source_patterns} -> {t.target_patterns}"
                for t in getattr(model, "_weight_conversions", None) or []
            ],
        },
        **({"preserved_frozen_sha256": frozen_hashes} if frozen_hashes is not None else {}),
        **({"tester_sizes": tester_sizes} if tester_sizes else {}),
        "checkpoint_inspection": "parameter headers/indexes; bounded binary fallback" if inspect_checkpoint else "full checkpoint load",
        "numeric_health_scope": "CPU float32/bfloat16/float16; autocast for low-precision losses",
        "captured": captured,
        "weights": weights,
        "gate": gate,
        **({"training": training} if training else {}),
        **({"overfit_reason": recipe["overfit"]} if recipe.get("overfit") else {}),
        "numeric_health": health,
        "copied_unchanged": copied,
        "environment": environment(),
        "sha256": run.file_hashes(out),
    }
    run.validate(build, "fixture-build", f"{name} build.json")
    (out / "build.json").write_text(json.dumps(build, indent=1, default=str) + "\n")
    (out / "README.md").write_text(readme(name, contract_name, build, cls, config, contract_repo))
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", default="transformers", help="whose fleet lock pins the sources")
    parser.add_argument("--fixture", action="append", help="build only this fixture (short name, repeatable)")
    parser.add_argument("--keys-only", action="store_true", help="check that covered checkpoints share a key")
    parser.add_argument("--out", type=Path, help="build into this directory instead, leaving the fixtures and lock untouched (reproducibility checks)")
    parser.add_argument("--keep-random", action="store_true", help="with --out: keep the random weights even where the recipe overfits (random-vs-overfit experiments)")
    parser.add_argument("--reshrink", action="store_true", help="apply today's shrink rule instead of the config a matching build.json recorded")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.keep_random and not args.out:
        parser.error("--keep-random needs --out: it never replaces a published fixture")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    transformers.logging.set_verbosity_error()

    lock_dir = HERE / args.framework / "tests/contracts"
    lock_path = lock_dir / "fleet.lock"
    lock = run.load(lock_path, "fleet-lock")
    if lock.get("local_patches"):
        import patches

        patches.apply(sorted(lock["local_patches"]))
    for contract_name, contract in lock["contracts"].items():
        root = (lock_dir / contract["integration"]["path"]).resolve()
        manifest = run.load(root / "integration.json", "integration")
        for name, fixture in manifest["fixtures"].items():
            if "build" not in fixture or (args.fixture and short(name) not in args.fixture):
                continue
            pin = contract["fixtures"][name]
            if args.keys_only:
                keys = {}
                for repo in fixture["covers"]:
                    source = source_snapshot(repo, contract["checkpoints"][repo], fixture["build"])
                    cls = model_class(source, fixture["build"])
                    lacks = missing_parameters(cls, run.checkpoint_path(repo, contract["checkpoints"][repo]))
                    frozen_hashes = None
                    if fixture["build"].get("preserve_frozen_prefixes"):
                        model = cls._from_config(shrunk_config(source, cls, fixture["build"]))
                        frozen_hashes = restore_frozen(model, source, repo, contract["checkpoints"][repo], fixture["build"])
                        del model
                    keys[repo] = fixture_key(source, cls, lacks, fixture["build"], frozen_hashes)[0][:12]
                print(json.dumps({"fixture": name, "keys": keys, "shared": len(set(keys.values())) == 1}))
                continue
            start = time.perf_counter()
            out = args.out / short(name) if args.out else (lock_dir / pin["path"]).resolve()
            build = build_fixture(name, fixture, contract_name, manifest, root, contract["checkpoints"], out, args.keep_random, contract_repo=contract["integration"]["repo"], reshrink=args.reshrink)
            if args.out:
                same = run.digest(build["sha256"]) == pin.get("files_sha256")
                print(json.dumps({"fixture": name, "matches_pin": same, "differs": [] if same else sorted(
                    f for f, h in build["sha256"].items()
                    if h != json.loads(((lock_dir / pin["path"]) / "build.json").read_text())["sha256"].get(f))}), flush=True)
                continue
            pin["files_sha256"] = run.digest(build["sha256"])
            lock_path.write_text(json.dumps(lock, indent=2) + "\n")
            gate = build["gate"]
            print(
                json.dumps(
                    {
                        "built": name,
                        "key": build["key"][:12],
                        "weights": build["weights"],
                        "parameters": build["parameters"],
                        "random_weakest": [gate["random"]["weakest"], gate["random"]["deltas"].get(gate["random"]["weakest"])],
                        "overfit_weakest": [gate["overfit"]["weakest"], gate["overfit"]["deltas"].get(gate["overfit"]["weakest"])] if "overfit" in gate else None,
                        "steps": build.get("training", {}).get("steps"),
                        "seconds": round(time.perf_counter() - start, 1),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
