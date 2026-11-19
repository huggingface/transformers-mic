"""Compare a fixture's config with the model's own Transformers test config.

Each model's `ModelTester.__init__` in `tests/models/<model>/test_modeling_<model>.py`
usually sets every field a small but representative model needs (Mistral-Large-4's
sets NoPE layers, KV sharing between layers of the same positional encoding, shared
indexers and an `index_topk` below the test sequence length). A core maintainer's
caveat: when a model was added quickly the tester can be thin or unrepresentative, so
these findings are advisory, never a source of fixture values.

The tester file is read and parsed, never imported or run. Since transformers#50462 a
tester declares its small values in a literal `small_config_overrides` dict and the shared
bases in `defaults` dicts (`{**Parent.defaults, ...}`): those are read statically, merged
as `Tester.get_small_config_defaults()` merges them, derived fields included. Testers that
still set values in `__init__` (`kwargs.setdefault`, `self.x = ...`) are parsed as before,
and those values win, as they do at runtime.
"""

import ast
import functools
import math
import re
import urllib.request

RAW = "https://raw.githubusercontent.com/huggingface/transformers/{ref}/tests/models/{folder}/test_modeling_{folder}.py"
BASE = "https://raw.githubusercontent.com/huggingface/transformers/{ref}/{path}"
# Shared testers a model's tester may only subclass (SmolLM3 sets nothing itself):
# their defaults are generic, so they rank below the model's own values.
BASES = {
    "ModelTesterConfigMixin": "tests/model_tester_config.py",
    "CausalLMModelTester": "tests/causal_lm_tester.py",
    "MultiModalModelTester": "tests/multimodal_tester.py",
    "VLMModelTester": "tests/vlm_tester.py",
    "ALMModelTester": "tests/alm_tester.py",
}

# Fields the recipe keeps on purpose: the fixture must load the real tokenizer and
# processor files, so vocabulary, special tokens and input geometry stay real.
PRESERVED = re.compile(
    r"vocab_size|_token_id|_token_index|max_position|image_size|patch_size|num_channels|"
    r"mel_bins|sampling_rate|merge_size|seq_length|batch_size|num_labels|num_choices|"
    r"window|sliding|kernel|stride|rope|theta|"
    # Behaviour, not width: BigBird's attention block_size, SeamlessM4T's chunking,
    # a convolution's groups (WavLM grew from 4.3M to 4.6M with the tester's 2).
    r"block_size|chunk_size|groups|"
    # A table of positions, not a width: Gemma 4's vision position_embedding_size must cover the
    # patch grid of a real image, and a build check runs no forward pass to catch it.
    r"position_embedding"
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


def init_values(cls):
    """Literal values one tester class's __init__ sets: arg defaults, kwargs.setdefault, self.x = ..."""
    found = {}
    init = next((f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    if init is None:
        return found
    args = init.args.args[-len(init.args.defaults):] if init.args.defaults else []
    for arg, default in zip(args, init.args.defaults):
        value = literal(default)
        if value is not None:
            found.setdefault(arg.arg, value)
    for node in ast.walk(init):
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
    return found


def testers(tree):
    return [c for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and "Tester" in c.name]


def values(code):
    """Literal config values the testers' __init__ set: defaults, kwargs.setdefault, self.x = ..."""
    found = {}
    for cls in testers(ast.parse(code)):
        for key, value in init_values(cls).items():
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


@functools.lru_cache(maxsize=128)
def parse(code):
    return ast.parse(code)


def class_node(code, name):
    tree = parse(code)
    return tree, next((c for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == name), None)


def imported_path(module, level):
    """The repo path of a test module a tester file imports from (`...models.bamba.test_modeling_bamba`,
    `..bert.test_modeling_bert`, `tests.models.x.test_modeling_x`), or None."""
    parts = (module or "").split(".")
    if level == 0 and parts[0] == "tests":
        parts = parts[1:]
    elif level == 2:  # tests/models/<folder>/ -> tests/models
        parts = ["models", *parts]
    elif level != 3:  # tests/models/<folder>/ -> tests
        return None
    return "/".join(["tests", *parts]) + ".py" if parts and parts[-1] else None


def find_class(name, tree, ref, seen=()):
    """(tree, class) of a tester a file can name: its own, one it imports from another test
    module (cohere2 subclasses cohere's tester), or a shared base. (None, None) otherwise."""
    if name in seen:
        return None, None
    module_alias, _, name = name.rpartition(".")  # `pocket_tts.PocketTTSModelTester`
    own = next((c for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == name), None)
    if own is not None and not module_alias:
        return tree, own
    if name in BASES:
        return class_node(fetch(BASE.format(ref=ref, path=BASES[name])), name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            bound = alias.asname or alias.name
            if module_alias and bound == module_alias:  # from ..pocket_tts import test_modeling_pocket_tts as pocket_tts
                path = imported_path(".".join(filter(None, (node.module, alias.name))), node.level)
            elif not module_alias and bound == name:  # from ...models.bamba.test_modeling_bamba import BambaModelTester
                path, name = imported_path(node.module, node.level), alias.name
            else:
                continue
            if path is None:
                return None, None
            try:
                return class_node(fetch(BASE.format(ref=ref, path=path)), name)
            except (OSError, KeyError, SyntaxError, UnicodeDecodeError):  # not a test module we can read
                return None, None
    return None, None


def ancestors(cls, tree, ref, seen=()):
    """(tree, class) for the class and every tester class above it, nearest first."""
    yield tree, cls
    seen = (*seen, cls.name)
    for base in cls.bases:
        base_tree, base_cls = find_class(ast.unparse(base), tree, ref, seen)
        if base_cls is not None and base_cls.name not in seen:
            yield from ancestors(base_cls, base_tree, ref, seen)


def resolve(node, tree, ref, seen=()):
    """A dict literal's values, following `**X.defaults` / `**X.small_config_overrides` into the
    shared bases and the module's (or an imported module's) testers, and `**NAME` into the
    module's own assignments; values that are not literals are left out."""
    if isinstance(node, ast.Dict):
        found = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                found.update(resolve(value, tree, ref, seen))
            elif isinstance(key, ast.Constant):
                try:  # a literal, or constant arithmetic (`32 * 4`)
                    found[key.value] = evaluate(value, {})
                except (Unknown, KeyError, IndexError, ValueError, TypeError, AttributeError, ArithmeticError):
                    pass
        return found
    if isinstance(node, ast.Name):
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == node.id for t in stmt.targets):
                return resolve(stmt.value, tree, ref, seen)
    if isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
        owner_name = ast.unparse(node.value)
        if (owner_name, node.attr) in seen:
            return {}
        owner_tree, owner = find_class(owner_name, tree, ref)
        if owner is not None:
            return class_attr(owner, node.attr, owner_tree, ref, (*seen, (owner_name, node.attr))) or {}
    return {}


def class_attr(cls, attr, tree, ref, seen=()):
    """A class-level dict (`defaults`, `small_config_overrides`), declared by the class or
    inherited from the nearest tester above it that declares it; None if none does."""
    for owner_tree, owner in ancestors(cls, tree, ref):
        for stmt in owner.body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in stmt.targets):
                return resolve(stmt.value, owner_tree, ref, seen)
    return None


def base_attr(name, attr, ref, seen=()):
    """A shared base's class-level dict, or None where no class declares it (before #50462)."""
    if (name, attr) in seen:
        return None
    tree, cls = class_node(fetch(BASE.format(ref=ref, path=BASES[name])), name)
    return class_attr(cls, attr, tree, ref, (*seen, (name, attr))) if cls else None


def lineage(name, ref, seen=()):
    """The shared base and every shared base above it."""
    if name in seen or name not in BASES:
        return set()
    _, cls = class_node(fetch(BASE.format(ref=ref, path=BASES[name])), name)
    names = {name}
    for base in cls.bases if cls else ():
        names |= lineage(ast.unparse(base).split(".")[-1], ref, (*seen, name))
    return names


def derive(values, lineage, explicit):
    """The fields `get_small_config_defaults()` derives (CausalLMModelTester, VLMModelTester)."""
    if "VLMModelTester" in lineage and {"image_size", "patch_size"} <= values.keys():
        values.setdefault("num_image_tokens", (values["image_size"] // values["patch_size"]) ** 2)
        if "seq_length" not in explicit:
            values["seq_length"] = 7 + values["num_image_tokens"]
    if lineage & {"CausalLMModelTester", "VLMModelTester"} and values.get("num_attention_heads"):
        values.setdefault("head_dim", values["hidden_size"] // values["num_attention_heads"])
    return values


OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
}
# The only functions an override may call: pure builtins over plain values.
FUNCTIONS = {
    "len": len, "list": list, "tuple": tuple, "dict": dict, "set": set, "sorted": sorted, "min": min,
    "max": max, "sum": sum, "abs": abs, "int": int, "float": float, "str": str, "bool": bool,
    "range": range, "enumerate": enumerate, "zip": zip,
}
# Pure functions of the math module (`math.ceil(min_size / 32)`).
MATH = {"ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt, "prod": math.prod, "log2": math.log2, "gcd": math.gcd}
# Methods on the plain dicts and lists the override works with.
METHODS = {"get", "setdefault", "pop", "copy", "keys", "values", "items", "index", "count"}


class Unknown(Exception):
    """The override does something a static reader does not follow; skip the statement."""


def evaluate(node, scope):
    """An expression over the override's own names (`values`, `kwargs`, `cls`, its locals),
    plain literals and displays, arithmetic, comparisons, comprehensions and whitelisted
    builtins. Anything else (an import, an attribute of a module, torch_device) raises Unknown."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in scope:
            return scope[node.id]
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        raise Unknown(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "cls":
        if node.attr in scope["cls"]:
            return scope["cls"][node.attr]
        raise Unknown(f"cls.{node.attr}")
    if isinstance(node, ast.Subscript):
        container = evaluate(node.value, scope)
        if isinstance(node.slice, ast.Slice):
            parts = [evaluate(x, scope) if x is not None else None for x in (node.slice.lower, node.slice.upper, node.slice.step)]
            return container[slice(*parts)]
        return container[evaluate(node.slice, scope)]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left, scope), evaluate(node.right, scope))
    if isinstance(node, ast.UnaryOp):
        operand = evaluate(node.operand, scope)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return not operand
        raise Unknown(ast.unparse(node))
    if isinstance(node, ast.BoolOp):
        result = None
        for value in node.values:
            result = evaluate(value, scope)
            if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = evaluate(node.left, scope)
        for op, right_node in zip(node.ops, node.comparators):
            right = evaluate(right_node, scope)
            if type(op) not in COMPARE:
                raise Unknown(ast.unparse(node))
            if not COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return evaluate(node.body if evaluate(node.test, scope) else node.orelse, scope)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                items.extend(evaluate(elt.value, scope))
            else:
                items.append(evaluate(elt, scope))
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(evaluate(value, scope))
            else:
                result[evaluate(key, scope)] = evaluate(value, scope)
        return result
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return comprehension(node, scope)
    if isinstance(node, ast.Call) and not any(isinstance(a, ast.Starred) for a in node.args):
        args = [evaluate(a, scope) for a in node.args]
        kwargs = {}
        for keyword in node.keywords:  # f(a=1, **text_kwargs)
            if keyword.arg:
                kwargs[keyword.arg] = evaluate(keyword.value, scope)
            else:
                kwargs.update(evaluate(keyword.value, scope) or {})
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_small_config_defaults"
            and isinstance(node.func.value, ast.Name)
            and "testers" in scope
            and not args
        ):  # a sibling tester's values: BlipTextModelTester.get_small_config_defaults(**text_kwargs)
            return scope["testers"](node.func.value.id, kwargs)
        if isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and node.func.id not in scope:
            return FUNCTIONS[node.func.id](*args, **kwargs)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
            and node.func.attr in MATH
        ):
            return MATH[node.func.attr](*args, **kwargs)
        if isinstance(node.func, ast.Attribute) and node.func.attr in METHODS:
            target = evaluate(node.func.value, scope)
            if isinstance(target, (dict, list, tuple)):
                return getattr(target, node.func.attr)(*args, **kwargs)
    raise Unknown(ast.unparse(node))


def comprehension(node, scope):
    def walk(generators, local):
        if not generators:
            if isinstance(node, ast.DictComp):
                yield evaluate(node.key, local), evaluate(node.value, local)
            else:
                yield evaluate(node.elt, local)
            return
        first, rest = generators[0], generators[1:]
        if first.is_async:
            raise Unknown("async comprehension")
        for item in evaluate(first.iter, local):
            inner = dict(local)
            bind(first.target, item, inner)
            if all(evaluate(cond, inner) for cond in first.ifs):
                yield from walk(rest, inner)

    items = list(walk(node.generators, dict(scope)))
    if isinstance(node, ast.DictComp):
        return dict(items)
    return set(items) if isinstance(node, ast.SetComp) else list(items)


def bind(target, value, scope):
    if isinstance(target, ast.Name):
        scope[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        value = list(value)
        if len(value) != len(target.elts):
            raise Unknown("unpack")
        for elt, item in zip(target.elts, value):
            bind(elt, item, scope)
    else:
        raise Unknown(ast.unparse(target))


def is_super_call(node):
    """`super().get_small_config_defaults(**kwargs)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_small_config_defaults"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    )


def split_override(method):
    """(statements before the super() call, statements after it) of an override."""
    for index, stmt in enumerate(method.body):
        value = getattr(stmt, "value", None)
        if is_super_call(value):
            return method.body[:index], method.body[index + 1:]
    return [], method.body  # no super() call: treat the whole body as derivations


def run_derivations(body, scope):
    """A model's own `get_small_config_defaults()` override, statement by statement: name and
    item assignments, `values`/`kwargs` mutations, `if`/`for`; a statement it cannot follow
    is skipped, and a `return` ends the walk."""
    for stmt in body:
        try:
            if isinstance(stmt, ast.Return):
                return scope
            if isinstance(stmt, ast.Assign):
                value = evaluate(stmt.value, scope)
                for target in stmt.targets:  # a = b = value
                    if isinstance(target, ast.Subscript):
                        container = evaluate(target.value, scope)
                        if not isinstance(container, (dict, list)):
                            raise Unknown(ast.unparse(target))
                        container[evaluate(target.slice, scope)] = value
                    else:
                        if isinstance(target, ast.Name) and target.id in ("values", "kwargs", "cls"):
                            raise Unknown(f"rebinding {target.id}")
                        bind(target, value, scope)
            elif isinstance(stmt, ast.AugAssign) and type(stmt.op) in OPS:  # values["seq_length"] += n
                current = evaluate(stmt.target, scope)
                value = OPS[type(stmt.op)](current, evaluate(stmt.value, scope))
                if isinstance(stmt.target, ast.Subscript):
                    container = evaluate(stmt.target.value, scope)
                    if not isinstance(container, (dict, list)):
                        raise Unknown(ast.unparse(stmt.target))
                    container[evaluate(stmt.target.slice, scope)] = value
                else:
                    bind(stmt.target, value, scope)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                evaluate(stmt.value, scope)  # values.setdefault / kwargs.setdefault
            elif isinstance(stmt, ast.If):
                run_derivations(stmt.body if evaluate(stmt.test, scope) else stmt.orelse, scope)
            elif isinstance(stmt, ast.For) and not stmt.orelse:
                for item in list(evaluate(stmt.iter, scope)):
                    bind(stmt.target, item, scope)
                    run_derivations(stmt.body, scope)
        except (Unknown, KeyError, IndexError, ValueError, TypeError, AttributeError, ArithmeticError):
            # The statement would have set these: a value left from before it is stale, not known.
            for key in assigned_keys(stmt, scope):
                scope["values"].pop(key, None)
    return scope


def assigned_keys(stmt, scope):
    """Constant keys a statement assigns in `values` (`values["k"] = ...`, `values.setdefault("k", ...)`)."""
    if not isinstance(scope.get("values"), dict):
        return set()
    keys = set()
    for node in ast.walk(stmt):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
        for target in targets:
            for sub in ast.walk(target):
                if (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "values"
                    and isinstance(sub.slice, ast.Constant)
                ):
                    keys.add(sub.slice.value)
    return keys


def method_scope(method, scope):
    """The override's own scope: its parameters' defaults (`text_kwargs=None`) over the shared names."""
    local = dict(scope)
    args = method.args
    for arg, default in zip(args.args[len(args.args) - len(args.defaults):], args.defaults):
        local[arg.arg] = literal(default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        local[arg.arg] = literal(default) if default is not None else None
    for name in list(local["kwargs"]):  # a named parameter takes its value out of **kwargs
        if any(a.arg == name for a in (*args.args, *args.kwonlyargs)):
            local[name] = local["kwargs"].pop(name)
    return local


def class_values(cls, tree, ref, kwargs=None, depth=0):
    """(merged values, own values, static) for one tester class, as `cls.get_small_config_defaults(**kwargs)`.

    Precedence is the runtime's: the caller's kwargs > what `__init__` still sets
    (kwargs.setdefault, self.x) > `small_config_overrides` > `defaults` (each the class's own,
    else the nearest tester's above it); then the fields the bases' and the testers' own
    `get_small_config_defaults()` derive, nearest override first, as super() chains them.
    An override that fills `kwargs` before calling `super()` (llava_next, hunyuan_vl) is
    followed as a caller would be: those values win and feed the bases' derivations.
    A sibling tester's `get_small_config_defaults(...)` is evaluated the same way.
    A base from before transformers#50462 has no `defaults`; its `__init__` is parsed instead.
    """
    legacy = init_values(cls)
    chain = list(ancestors(cls, tree, ref))
    overrides = class_attr(cls, "small_config_overrides", tree, ref) or {}
    defaults = class_attr(cls, "defaults", tree, ref)
    names = {c.name for _, c in chain} & set(BASES)
    lines = set().union(*(lineage(name, ref) for name in names)) if names else set()
    static = defaults is not None or bool(overrides)
    if defaults is None:
        defaults = {}
        for name in sorted(names):
            for key, value in values(fetch(BASE.format(ref=ref, path=BASES[name]))).items():
                defaults.setdefault(key, value)
    own = {**overrides, **legacy}
    if not static:
        return {**defaults, **own}, own, static

    def sibling(name, sibling_kwargs):
        sibling_tree, sibling_cls = find_class(name, tree, ref)
        if sibling_cls is None or depth > 4:
            raise Unknown(name)
        return class_values(sibling_cls, sibling_tree, ref, sibling_kwargs, depth + 1)[0]

    passed = dict(kwargs or {})  # what the overrides hand super() in kwargs, starting with the caller's
    attributes = {}  # plain class attributes the overrides read (`cls.base` on funnel's testers)
    for _, owner in chain:
        for stmt in owner.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                value = literal(stmt.value)
                if value is not None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                    attributes.setdefault(stmt.targets[0].id, value)
    attributes.update(small_config_overrides=dict(overrides), defaults=dict(defaults))
    shared = {"kwargs": passed, "testers": sibling, "cls": attributes}
    methods = []  # the testers' own overrides, nearest first; the shared bases' are `derive`
    for _, owner in chain:
        if owner.name in BASES:
            break
        method = next((f for f in owner.body if isinstance(f, ast.FunctionDef) and f.name == "get_small_config_defaults"), None)
        if method is not None:
            methods.append((method, *split_override(method)))
    scopes = []
    for method, before, _ in methods:
        scope = method_scope(method, shared)
        run_derivations(before, scope)
        scopes.append(scope)
        if not any(is_super_call(getattr(stmt, "value", None)) for stmt in method.body):
            break  # no super() call: nothing further up runs
    found = {**defaults, **own, **passed}
    derive(found, lines, set(own) | set(passed))
    for (method, _, after), scope in reversed(list(zip(methods, scopes))):
        scope["values"] = found
        run_derivations(after, scope)
    return found, own, static


def gather(code, ref="main", only=None):
    """(merged values, the model's own values, static) over the file's tester classes, the
    first class's value winning; `only` names one class."""
    tree = ast.parse(code)
    found, own, static = {}, {}, False
    for cls in testers(tree):
        if only and cls.name != only:
            continue
        cls_found, cls_own, cls_static = class_values(cls, tree, ref)
        for key, value in cls_found.items():
            found.setdefault(key, value)
        for key, value in cls_own.items():
            own.setdefault(key, value)
        static = static or cls_static
    return found, own, static


def tester_values(model_type, ref="main"):
    """(url, merged literal values) or (None, {}): the model's tester, then its base tester."""
    try:
        url, code = source(model_type, ref)
        return url, gather(code, ref)[0]
    except Exception:  # no tester, renamed folder, offline
        return None, {}


def check(original, shrunk, ref="main", kept_whole=False):
    """{'url', 'values', 'findings'} or {'error'}: never raises."""
    try:
        url, code = source(shrunk.get("model_type") or original["model_type"], ref)
        found, own, static = gather(code, ref)
    except Exception as error:  # no tester, renamed folder, offline
        return {"error": f"{type(error).__name__}: {error}"[:200]}
    return {
        "url": url,
        "values": len(found),
        "own_values": len(own),
        "static": static,
        "findings": compare(found, original, shrunk, kept_whole),
    }
