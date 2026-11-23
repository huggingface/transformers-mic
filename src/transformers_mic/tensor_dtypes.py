"""The dtype each tensor of a checkpoint is stored in, read from safetensors headers.

Real checkpoints are not stored in one dtype: MoE routers keep `mlp.gate.weight`
or `e_score_correction_bias` in float32 beside bfloat16 weights, and quantized
experts are F8. A fixture saved in one dtype hides changes that depend on it (a
router kept in float32 on load is a lossless upcast of a bfloat16 fixture, and a
false difference for a float32 fixture of a bfloat16 router).

So a fixture mirrors the checkpoint's dtypes relative to each other: tensors stored
in the checkpoint's main float dtype (the one most parameters use) take the
fixture's default dtype, the dtype users load the checkpoint in, and tensors stored
in another float dtype keep it. Names match by pattern, so a shrunk fixture (fewer
layers, fewer experts) still maps onto the checkpoint. A uniformly stored checkpoint
therefore changes nothing (SmolLM-135M stores float32 under a bfloat16 config; its
fixture stays bfloat16), nor does a float32 fixture of a bfloat16 checkpoint whose
router is float32.

Only headers are read: the index plus each shard's 8-byte length and header JSON,
by HTTP range request or from local files, cached per revision.
"""

import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Stored float dtypes a fixture keeps; anything else (F8_*, I8, U8, packed
# quantized blocks, F64) is storage the dense fixture does not mirror.
FLOAT = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}


def pattern(name):
    """A tensor name with its numeric path components wildcarded: layers.45.mlp -> layers.*.mlp."""
    return re.sub(r"(?<=\.)\d+(?=\.|$)|^\d+(?=\.)", "*", name)


def _dtype(value):
    return value if isinstance(value, str) else value["dtype"]


def _numel(value):
    return 1 if isinstance(value, str) else value["numel"]


def pattern_dtypes(stored):
    """{pattern: torch dtype name} for patterns whose source tensors are unanimous in a float dtype.

    `stored` maps checkpoint tensor names to safetensors dtypes ("BF16", "F8_E4M3"),
    or to {"dtype", "numel"} as checkpoint_dtypes returns them. A pattern whose
    tensors disagree (o_proj BF16 in some layers, F8 in others), or that is stored
    quantized or as integers, has no entry: the fixture default applies.
    """
    seen = collections.defaultdict(set)
    for name, value in stored.items():
        seen[pattern(name)].add(_dtype(value))
    return {
        key: FLOAT[next(iter(dtypes))] for key, dtypes in seen.items() if len(dtypes) == 1 and dtypes <= FLOAT.keys()
    }


def main_dtype(stored):
    """The float dtype most of the checkpoint's float parameters are stored in (tensors when sizes are unknown)."""
    totals = collections.Counter()
    for value in stored.values():
        if _dtype(value) in FLOAT:
            totals[FLOAT[_dtype(value)]] += _numel(value)
    return max(sorted(totals), key=totals.get) if totals else None


def mixed(stored):
    """Whether the checkpoint stores its floating tensors in more than one float dtype."""
    return len({_dtype(value) for value in stored.values() if _dtype(value) in FLOAT}) > 1


def kept_dtypes(stored):
    """{pattern: torch dtype name} the fixture keeps: the patterns stored in another float dtype than the main one."""
    main = main_dtype(stored)
    return {key: dtype for key, dtype in pattern_dtypes(stored).items() if dtype != main}


def overrides(stored, default):
    """The kept patterns whose dtype differs from the fixture default: what a fixture key records."""
    return {key: dtype for key, dtype in sorted(kept_dtypes(stored).items()) if dtype != default}


def target_dtypes(names, stored, default):
    """{fixture tensor name: torch dtype name} each floating fixture tensor is saved in.

    Names stored in the checkpoint's main float dtype take the default, as do names
    whose pattern the checkpoint does not have (renamed or fused by a conversion).
    """
    patterns = kept_dtypes(stored)
    return {name: patterns.get(pattern(name), default) for name in names}


def config_dtype(config):
    """The dtype a config names: its own, else its sub-configs' (text_config first, else unanimous)."""
    own = config.get("dtype") or config.get("torch_dtype")
    if own:
        return own
    nested = {}
    for key, value in config.items():
        if isinstance(value, dict) and ("model_type" in value or key.endswith("_config")):
            found = config_dtype(value)
            if found:
                nested[key] = found
    if "text_config" in nested:
        return nested["text_config"]
    if len(set(nested.values())) == 1:
        return next(iter(nested.values()))
    return None


def _header(path):
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(length))


def _local_dtypes(source, files):
    stored = {}
    for name in files:
        for tensor, info in _header(Path(source) / name).items():
            if tensor != "__metadata__":
                numel = 1
                for size in info["shape"]:
                    numel *= size
                stored[tensor] = {"dtype": info["dtype"], "numel": numel}
    return stored


def weight_files(source):
    """The safetensors files a checkpoint loads from, from its local config/index, or None if it has none."""
    from .inspection import explicit_weights

    source = Path(source)
    explicit = explicit_weights(source)
    index = source / (explicit if explicit and explicit.endswith(".index.json") else "model.safetensors.index.json")
    if explicit and not explicit.endswith(".index.json"):
        return [explicit]
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return ["model.safetensors"]


def cache_path(repo, revision):
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE).parent / "tmic-tensor-headers" / repo.replace("/", "--") / f"{revision}.json"


def checkpoint_dtypes(repo, revision, source=None):
    """{tensor name: {"dtype": safetensors dtype, "numel"}} of a pinned checkpoint, from headers only.

    Empty for a checkpoint with no safetensors file (a pytorch_model.bin source):
    its fixture keeps the default dtype. Cached per revision, since a commit's
    headers never change.
    """
    cached = cache_path(repo, revision) if re.fullmatch(r"[0-9a-f]{40}", revision or "") else None
    if cached and cached.exists():
        return json.loads(cached.read_text())
    from huggingface_hub import HfApi, snapshot_download

    if source is None:  # the config and indexes only: they name the shards
        source = snapshot_download(repo, revision=revision, allow_patterns=["config.json", "*.index.json"])
    files = weight_files(source)
    if all((Path(source) / name).exists() for name in files):
        stored = _local_dtypes(source, files)
    else:
        api = HfApi()
        if files == ["model.safetensors"] and not api.file_exists(repo, "model.safetensors", revision=revision):
            stored = {}
        else:
            with ThreadPoolExecutor(8) as pool:
                headers = list(
                    pool.map(
                        lambda name: api.parse_safetensors_file_metadata(repo, name, revision=revision, timeout=60),
                        files,
                    )
                )
            stored = {
                name: {"dtype": info.dtype, "numel": info.parameter_count}
                for header in headers
                for name, info in header.tensors.items()
            }
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(stored, sort_keys=True))
    return stored
