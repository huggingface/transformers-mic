"""Call the public entry point affected by each replayed regression.

    PYTHONPATH=<build>/src python usage_repros.py <case> <real|slim>

Prints one JSON observation. Compare the output of the bad and fixed builds.
"slim" uses the seed-17 overfit fixture frozen by replay.py under
$REPLAY_WORKDIR/frozen (processor files re-saved); "copied" uses fixtures built
by `usage_repros.py prepare` on a known-good build, with processor files copied
byte for byte from the real model; "real" uses the pinned real checkpoint.
"""
import json, os, shutil, sys, tempfile, warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import torch
import transformers
from PIL import Image

transformers.logging.set_verbosity_error()
import experiment as exp

WORKDIR = Path(os.environ.get("REPLAY_WORKDIR", Path(tempfile.gettempdir()) / "model-integration-replay"))


EXTRA_SOURCES = {
    "bigbird": ("google/bigbird-roberta-base", "5a145f7852cba9bd431386a58137bf8a29903b90", ""),
    "mms": ("facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd", ""),
}
WEIGHTS = (".safetensors", ".bin")


def real_path(family):
    if family in EXTRA_SOURCES:
        return exp.snapshot(EXTRA_SOURCES[family])
    return exp.snapshot(exp.CLIP_REFERENCE if family == "clip" else exp.SOURCES[family])


def path(family, target):
    if target == "real":
        return real_path(family)
    if target == "copied":
        return WORKDIR / "copied" / family
    return WORKDIR / "frozen" / family / "17" / "overfit"


def copy_processor_files(src, dst):
    """Copy every non-weight, non-config file unchanged."""
    for f in Path(src).iterdir():
        if f.is_file() and not f.name.endswith(WEIGHTS) and f.name not in ("config.json", "generation_config.json"):
            shutil.copy(f, dst / f.name)


def prepare():
    """Build "copied" fixtures on the current (known-good) build."""
    from transformers import AutoConfig, BigBirdForCausalLM
    root = WORKDIR / "copied"
    # Fixtures from replay.py with their re-saved processor files replaced by the originals.
    for family in ("clip", "whisper", "wav2vec2"):
        dst = root / family
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for f in (WORKDIR / "frozen" / family / "17" / "overfit").iterdir():
            if f.name.endswith(WEIGHTS) or f.name in ("config.json", "generation_config.json"):
                shutil.copy(f, dst / f.name)
        copy_processor_files(exp.snapshot(exp.SOURCES[family]), dst)
    # MMS: only the tokenizer is involved, copied unchanged.
    dst = root / "mms"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    copy_processor_files(real_path("mms"), dst)
    # BigBird: real config shrunk, fixed seed, original tokenizer files.
    dst = root / "bigbird"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    config = AutoConfig.from_pretrained(real_path("bigbird"))
    config.hidden_size, config.intermediate_size = 64, 128
    config.num_hidden_layers, config.num_attention_heads = 2, 4
    config.hidden_dropout_prob = config.attention_probs_dropout_prob = 0.0
    torch.manual_seed(17)
    BigBirdForCausalLM(config).save_pretrained(dst)
    copy_processor_files(real_path("bigbird"), dst)
    print(json.dumps({"prepared": sorted(p.name for p in root.iterdir())}))


def r(x):
    return [round(v, 4) for v in torch.as_tensor(x).flatten().tolist()]


def clip_text_kwargs(p):
    """#46195: flat return_tensors dropped when text_kwargs is passed."""
    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained(p)
    images = [Image.new("RGB", (256, 256), c) for c in ((255, 0, 0), (0, 0, 255))]
    out = proc(text=["a red square", "a blue square"], images=images,
               return_tensors="pt", text_kwargs={"padding": True})
    model = CLIPModel.from_pretrained(p).eval()
    with torch.no_grad():
        logits = model(**out).logits_per_image
    return {"input_ids_type": type(out["input_ids"]).__name__, "logits": r(logits)}


def t5_batch_decode(p):
    """#47793: decode() on a batch ignores clean_up_tokenization_spaces."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(p)
    model = T5ForConditionalGeneration.from_pretrained(p).eval()
    prompts = (["translate English to German: Hello, world. How are you?",
                "translate English to French: Yes , it is ."] if "snapshots" in str(p)
               else ["capital of France:", "capital of Germany:"])
    x = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        ids = model.generate(**x, max_new_tokens=16, do_sample=False).tolist()
    return {
        "generate_then_decode": tok.decode(ids, skip_special_tokens=True),
        "roundtrip": tok.decode(tok(["a . b", "c , d"]).input_ids, skip_special_tokens=True),
    }


def legacy_special_tokens(p):
    """#47848: additional_special_tokens lost when extra_special_tokens is empty."""
    from transformers import AutoTokenizer
    tmp = Path(tempfile.mkdtemp())
    for f in p.iterdir():
        if f.name.startswith(("tokenizer", "special_tokens", "vocab", "merges")):
            shutil.copy(f, tmp / f.name)
    cfg = json.loads((tmp / "tokenizer_config.json").read_text())
    cfg["extra_special_tokens"] = {}
    (tmp / "tokenizer_config.json").write_text(json.dumps(cfg))
    tok = AutoTokenizer.from_pretrained(tmp)
    ids = tok("<|im_start|>user\nhi<|im_end|>").input_ids
    return {"has_additional": "additional_special_tokens" in json.loads((p / "tokenizer_config.json").read_text()),
            "decoded": tok.decode(ids, skip_special_tokens=True),
            "n_special": len(tok.all_special_tokens)}


def whisper_timestamps(p):
    """#47997: batch decode_with_timestamps crashes."""
    from transformers import WhisperProcessor
    proc = WhisperProcessor.from_pretrained(p)
    begin = proc.tokenizer.convert_tokens_to_ids("<|0.00|>")
    eot = proc.tokenizer.eos_token_id
    ids = [[begin, 2182, begin + 50, eot], [begin, 3092, begin + 50, eot]]
    return {"batch_decode": proc.batch_decode(ids, decode_with_timestamps=True),
            "decode_2d": proc.tokenizer.decode(ids, decode_with_timestamps=True)}


def config_helpers(p, family):
    """#47628: get_*_config() returns {} for repos with only preprocessor_config.json."""
    if family == "clip":
        from transformers.models.auto.image_processing_auto import get_image_processor_config as get
    else:
        from transformers.models.auto.feature_extraction_auto import get_feature_extractor_config as get
    cfg = get(str(p))
    return {"keys": sorted(cfg)[:8], "n_keys": len(cfg),
            "files": sorted(f.name for f in Path(p).iterdir() if f.name.endswith("config.json"))}


def mms_target_lang(p):
    """#47440: set_target_lang leaves stale added-token encoder entries."""
    from transformers import Wav2Vec2CTCTokenizer
    tok = Wav2Vec2CTCTokenizer.from_pretrained(p, target_lang="eng")
    out = {}
    for lang, text in (("fra", "bonjour à tous"), ("deu", "guten morgen"), ("eng", "hello there")):
        tok.set_target_lang(lang)
        ids = tok(text).input_ids
        out[lang] = {"ids": ids, "decoded": tok.decode(ids),
                     "stale_added": sorted(set(tok._added_tokens_encoder) - {str(t) for t in tok._added_tokens_decoder.values()})}
    return out


def clip_text_projection(p):
    """#46406: CLIP conversion mapping applied to the wrong nesting level."""
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    tok = AutoTokenizer.from_pretrained(p)
    model, info = CLIPTextModelWithProjection.from_pretrained(p, output_loading_info=True)
    with torch.no_grad():
        emb = model(**tok(["a photo of a cat"], return_tensors="pt")).text_embeds
    return {"missing": len(info["missing_keys"]), "unexpected": len(info["unexpected_keys"]),
            "text_embeds": r(emb[0, :8])}


def bigbird_clm(p):
    """#48785: BigBird/RemBERT/RoFormer CLM attends to future tokens."""
    from transformers import AutoTokenizer, BigBirdForCausalLM
    tok = AutoTokenizer.from_pretrained(p)
    model = BigBirdForCausalLM.from_pretrained(p, is_decoder=True, attention_type="original_full").eval()
    ids = tok(["The capital of France is Paris, and it is lovely."], return_tensors="pt").input_ids
    with torch.no_grad():
        full = model(input_ids=ids).logits
        prefix = model(input_ids=ids[:, :4]).logits
    return {"future_leak": float((full[:, :4] - prefix).abs().max()), "logits": r(full[0, 3, :6])}


CASES = {
    "clip_text_kwargs": ("clip", clip_text_kwargs),
    "t5_batch_decode": ("t5", t5_batch_decode),
    "legacy_special_tokens": ("text", legacy_special_tokens),
    "whisper_timestamps": ("whisper", whisper_timestamps),
    "config_helpers_clip": ("clip", lambda p: config_helpers(p, "clip")),
    "config_helpers_whisper": ("whisper", lambda p: config_helpers(p, "whisper")),
    "config_helpers_wav2vec2": ("wav2vec2", lambda p: config_helpers(p, "wav2vec2")),
    "mms_target_lang": ("mms", mms_target_lang),
    "clip_text_projection": ("clip", clip_text_projection),
    "bigbird_clm": ("bigbird", bigbird_clm),
}

if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
        sys.exit()
    family, fn = CASES[sys.argv[1]]
    try:
        obs = fn(path(family, sys.argv[2]))
    except Exception as exc:
        obs = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    print(json.dumps(obs))
