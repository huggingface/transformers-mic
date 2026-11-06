"""Compare a fixture's config with the model's own Transformers test config.

Each model's `ModelTester.__init__` in `tests/models/<model>/test_modeling_<model>.py`
usually sets every field a small but representative model needs (Mistral-Large-4's
sets NoPE layers, KV sharing between layers of the same positional encoding, shared
indexers and an `index_topk` below the test sequence length). A core maintainer's
caveat: when a model was added quickly the tester can be thin or unrepresentative, so
these findings are advisory, never a source of fixture values.

A static `get_small_config_defaults()` can expose a literal dictionary shared with
the tester constructor. The tester file is read and parsed, never imported or run.
"""

import ast
import functools
import re
import urllib.request

RAW = "https://raw.githubusercontent.com/huggingface/transformers/{ref}/tests/models/{folder}/test_modeling_{folder}.py"
BASE = "https://raw.githubusercontent.com/huggingface/transformers/{ref}/{path}"
# Shared testers a model's tester may only subclass (SmolLM3 sets nothing itself):
# their defaults are generic, so they rank below the model's own values.
BASES = {"CausalLMModelTester": "tests/causal_lm_tester.py", "VLMModelTester": "tests/vlm_tester.py"}

# Fields the recipe keeps on purpose: the fixture must load the real tokenizer and
# processor files, so vocabulary, special tokens and input geometry stay real.
PRESERVED = re.compile(
    r"vocab_size|_token_id|_token_index|max_position|image_size|patch_size|num_channels|"
    r"mel_bins|sampling_rate|merge_size|seq_length|batch_size|num_labels|num_choices|"
    r"window|sliding|kernel|stride|rope|theta|"
    # Behaviour, not width: BigBird's attention block_size, SeamlessM4T's chunking,
    # a convolution's groups (WavLM grew from 4.3M to 4.6M with the tester's 2).
    r"block_size|chunk_size|groups"
)
# Fields that size a model: what the tester shrinks and the fixture may have kept.
SIZES = re.compile(r"(size|dim|heads|experts|rank|topk|_per_tok)$")


def folder(model_type):
    """The tests/models folder of a model type, from its config class's module."""
    from transformers import CONFIG_MAPPING

    module = CONFIG_MAPPING[model_type].__module__  # transformers.models.<folder>.configuration_x
    return module.split(".")[2]


@functools.lru_cache(maxsize=64)
def fetch(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def source(model_type, ref="main"):
    url = RAW.format(ref=ref, folder=folder(model_type))
    return url, fetch(url)


def bases(code):
    """Names of the shared base testers the file's testers subclass."""
    names = set()
    for cls in ast.walk(ast.parse(code)):
        if isinstance(cls, ast.ClassDef) and "Tester" in cls.name:
            names |= {ast.unparse(b).split(".")[-1] for b in cls.bases}
    return sorted(names & set(BASES))


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return None


def values(code):
    """Read static small-config defaults and legacy constructor literals without executing code."""
    found = {}
    for cls in ast.walk(ast.parse(code)):
        if not (isinstance(cls, ast.ClassDef) and "Tester" in cls.name):
            continue
        defaults = {}
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "get_small_config_defaults":
                continue
            if not any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in method.decorator_list):
                continue
            body = method.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if len(body) == 1 and isinstance(body[0], ast.Return):
                data = literal(body[0].value)
                if isinstance(data, dict) and all(isinstance(key, str) for key in data):
                    defaults = data
        init = next((f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
        args = init.args.args[-len(init.args.defaults):] if init and init.args.defaults else []
        for arg, default in zip(args, init.args.defaults if init else []):
            value = literal(default)
            if value is not None:
                found.setdefault(arg.arg, value)
        for node in ast.walk(init) if init else []:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
            ):
                value = literal(node.args[1])
                if value is not None:
                    found.setdefault(node.args[0].value, value)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    value = literal(node.value)
                    if value is not None:
                        found.setdefault(target.attr, value)
        for key, value in defaults.items():
            found.setdefault(key, value)
    return found


SUBS = ("text_config", "language_config", "llm_config", "decoder", "vision_config", "audio_config")


def locate(config, key):
    """(path, value) of a field in the config or the first sub-config that has it."""
    if key in config:
        return (key,), config[key]
    for sub in SUBS:
        if isinstance(config.get(sub), dict) and key in config[sub]:
            return (sub, key), config[sub][key]
    return None, None


def lookup(config, key):
    """A field from the config or the first sub-config that has it."""
    if key in config:
        return config[key]
    for sub in ("text_config", "language_config", "llm_config", "decoder", "vision_config", "audio_config"):
        if isinstance(config.get(sub), dict) and key in config[sub]:
            return config[sub][key]
    return None


def compare(tester, original, shrunk, kept_whole=False):
    """Advisory findings: kinds the tester covers that the fixture lacks, sizes it kept.

    Only per-layer kind lists count (as long as the tester's depth, of strings, booleans
    or 0/1 flags, like the shrink rule's kinds): stage lists such as `depths` or
    `out_features` are not kinds. A fixture kept whole (a small source) has no sizes to shrink.
    """
    findings = []
    depth = tester.get("num_hidden_layers")
    for key, value in sorted(tester.items()):
        fixture = lookup(shrunk, key)
        if isinstance(value, list) and isinstance(fixture, list) and value and len(value) == depth and all(
            isinstance(v, (str, bool)) or v in (0, 1) for v in value
        ):
            missing = sorted(set(map(str, value)) - set(map(str, fixture)))
            if missing:
                findings.append(
                    f"{key}: the model tester covers {', '.join(missing)}, the fixture has only "
                    f"{', '.join(sorted(set(map(str, fixture))))}"
                )
        elif (
            not kept_whole
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            and SIZES.search(key)
            and not PRESERVED.search(key)
            and isinstance(fixture, int)
            and fixture == lookup(original, key)
            and fixture >= 8 * value
        ):
            findings.append(f"{key} kept at the checkpoint's {fixture}; the model tester uses {value}")
    return findings


def unshrunk_sizes(tester, original, shrunk):
    """{path: tester value} for the size fields the generic rule left at the checkpoint's value."""
    adopted = {}
    for key, value in sorted(tester.items()):
        path, fixture = locate(shrunk, key)
        if (
            path
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            and SIZES.search(key)
            and not PRESERVED.search(key)
            and isinstance(fixture, int)
            and fixture == lookup(original, key)
            and fixture >= 8 * value
        ):
            adopted[path] = value
    return adopted


def tester_values(model_type, ref="main"):
    """(url, merged literal values) or (None, {}): the model's tester, then its base tester."""
    try:
        url, code = source(model_type, ref)
        found = values(code)
        for base in bases(code):
            for key, value in values(fetch(BASE.format(ref=ref, path=BASES[base]))).items():
                found.setdefault(key, value)
        return url, found
    except Exception:  # no tester, renamed folder, offline
        return None, {}


def check(original, shrunk, ref="main", kept_whole=False):
    """{'url', 'values', 'findings'} or {'error'}: never raises."""
    try:
        url, code = source(shrunk.get("model_type") or original["model_type"], ref)
        found = values(code)
        own = len(found)
        for base in bases(code):
            for key, value in values(fetch(BASE.format(ref=ref, path=BASES[base]))).items():
                found.setdefault(key, value)
    except Exception as error:  # no tester, renamed folder, offline
        return {"error": f"{type(error).__name__}: {error}"[:200]}
    return {"url": url, "values": len(found), "own_values": own, "findings": compare(found, original, shrunk, kept_whole)}
