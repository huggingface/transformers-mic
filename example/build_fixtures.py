"""Build the slim fixtures listed in the SmolLM contract's manifest.

    python example/build_fixtures.py

One fixture serves every checkpoint whose fixture key matches: the processor
files byte for byte, plus the config fields that survive shrinking. The build
fails if a checkpoint assigned to a fixture has a different key, instead of
silently testing it through the wrong fixture. Each fixture shrinks only the
model config and weights (2 layers, hidden size 64), is trained until it
answers the contract's prompts, and gets the processor files copied unchanged.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import experiment as exp  # noqa: E402

MANIFEST = HERE / "hub/SmolLM-integration/integration.json"
LOCK = HERE / "transformers/tests/contracts/fleet.lock"
PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
# Config fields that shrinking keeps. Sizes (width, depth, head counts) are
# replaced, so checkpoints that differ only in size share a fixture.
KEY_FIELDS = (
    "model_type",
    "architectures",
    "vocab_size",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "max_position_embeddings",
    "bos_token_id",
    "eos_token_id",
    "rms_norm_eps",
    "hidden_act",
    "attention_bias",
    "mlp_bias",
)
SEED = 17


def fixture_key(path):
    config = json.loads((path / "config.json").read_text())
    fields = {name: config.get(name) for name in KEY_FIELDS}
    fields["grouped_kv_heads"] = config["num_key_value_heads"] < config["num_attention_heads"]
    processor = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in PROCESSOR_FILES}
    blob = json.dumps({"config": fields, "processor": processor}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest(), fields


def shrink(source, grouped):
    config = AutoConfig.from_pretrained(source)
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    # Keep the attention layout: grouped key/value heads stay grouped.
    config.num_key_value_heads = 2 if grouped else 4
    config.head_dim = 16
    config.pad_token_id = config.eos_token_id
    return AutoModelForCausalLM.from_config(config, attn_implementation="eager", dtype=torch.float32)


def main():
    manifest = json.loads(MANIFEST.read_text())
    revisions = json.loads(LOCK.read_text())["contracts"]["SmolLM"]["checkpoints"]
    for name, fixture in manifest["fixtures"].items():
        keys = {}
        for repo in fixture["covers"]:
            path = Path(snapshot_download(repo, revision=revisions[repo], allow_patterns=["*.json", "*.txt"]))
            keys[repo] = fixture_key(path)
        if len({key for key, _ in keys.values()}) != 1:
            raise SystemExit(f"{name}: checkpoints have different fixture keys: {sorted(keys)}")
        key, fields = next(iter(keys.values()))
        source_repo = fixture["built_from"]
        source = Path(snapshot_download(source_repo, revision=revisions[source_repo], allow_patterns=["*.json", "*.txt"]))
        task = exp.TextTask(source)
        torch.manual_seed(SEED)
        model = shrink(source, fields["grouped_kv_heads"])
        training = exp.train(task, model, max_steps=500)
        if not training["converged"]:
            raise SystemExit(f"{name}: fixture did not converge: {training}")
        out = HERE / "hub" / name.split("/")[-1]
        out.mkdir(parents=True, exist_ok=True)
        for file in out.glob("*.json"):
            file.unlink()
        model.save_pretrained(out)
        for file in PROCESSOR_FILES:
            shutil.copyfile(source / file, out / file)
        build = {
            "fixture": name,
            "key": key,
            "key_fields": fields,
            "covers": {repo: revisions[repo] for repo in fixture["covers"]},
            "built_from": {"repo": source_repo, "revision": revisions[source_repo]},
            "seed": SEED,
            "training": training,
            "copied_unchanged": list(PROCESSOR_FILES),
            "sha256": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted(out.iterdir())
                if file.is_file() and file.name not in ("README.md", "build.json")
            },
        }
        (out / "build.json").write_text(json.dumps(build, indent=1) + "\n")
        exp.log("built", fixture=name, key=key[:12], covers=fixture["covers"], steps=training["steps"])


if __name__ == "__main__":
    main()
