"""Compare frozen-random and overfit fixtures on identical synthetic faults.

Uses pinned, already-cached Hub snapshots; never modifies or publishes them.
Run with a Python environment containing torch, transformers, Pillow, and their
processor dependencies. See --help. Results are observations, not a benchmark of
historical bugs or long-term fixture maintenance.
"""

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import IncompleteSnapshotError
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertForMaskedLM,
    CLIPModel,
    CLIPProcessor,
    T5ForConditionalGeneration,
    UMT5Config,
    UMT5ForConditionalGeneration,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.llama import modeling_llama
from transformers.models.t5 import modeling_t5
from transformers.models.whisper import modeling_whisper


SOURCES = {
    "text": (
        "HuggingFaceTB/SmolLM-135M",
        "1d461723eec654e65efdc40cf49301c89c0c92f4",
        "",
    ),
    "clip": (
        "openai/clip-vit-base-patch32",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "",
    ),
}
# This cached mirror supplies the full CLIP checkpoint; provenance is separate.
CLIP_REFERENCE = (
    "sentence-transformers/clip-ViT-B-32",
    "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f",
    "0_CLIPModel",
)
ATOL = RTOL = 1e-4


def snapshot(source):
    repo, revision, subfolder = source
    try:
        path = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except IncompleteSnapshotError:
        # Only the files these tasks need are cached; use them as they are.
        path = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    return path / subfolder


def log(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def zero_attention(module, args, output):
    return torch.zeros_like(output[0]), output[1]


class TextTask:
    name = "text"
    faults = (
        "wrong_country_token",
        "ignore_padding",
        "remove_causal_mask",
        "disable_rope",
        "zero_attention",
        "zero_mlp",
        "detach_mlp",
        "wrong_training_labels",
    )
    required_gradients = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    )

    def __init__(self, path):
        self.path = path
        self.processor = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.processor.pad_token = self.processor.eos_token
        self.processor.padding_side = "left"
        self.prompts = [
            f"The capital of {country} is"
            for country in ("France", "Germany", "Italy", "the United Kingdom")
        ]
        self.answers = [" Paris", " Berlin", " Rome", " London"]
        self.inference_batch = dict(
            self.processor(self.prompts, padding=True, return_tensors="pt")
        )
        prefixes = [
            self.processor.encode(p, add_special_tokens=False) for p in self.prompts
        ]
        self.gold = [
            self.processor.encode(a, add_special_tokens=False)
            + [self.processor.eos_token_id]
            for a in self.answers
        ]
        rows = [p + a for p, a in zip(prefixes, self.gold)]
        self.training_batch = dict(
            self.processor.pad({"input_ids": rows}, return_tensors="pt")
        )
        labels = torch.full_like(self.training_batch["input_ids"], -100)
        for index, answer in enumerate(self.gold):
            labels[index, -len(answer) :] = torch.tensor(answer)
        self.training_batch["labels"] = labels
        self.sample_ids = sorted(
            set(
                torch.linspace(0, len(self.processor) - 1, 32).long().tolist()
                + [token for answer in self.gold for token in answer]
            )
        )
        self.france = self.processor.encode(" France", add_special_tokens=False)[-1]
        self.germany = self.processor.encode(" Germany", add_special_tokens=False)[-1]

    def make_model(self):
        config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.pad_token_id = self.processor.pad_token_id
        return AutoModelForCausalLM.from_config(
            config, attn_implementation="eager", dtype=torch.float32
        )

    def load_model(self, path, attn="eager"):
        return AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            attn_implementation=attn,
            dtype=torch.float32,
        )

    def loss(self, model, batch):
        return model(**batch, use_cache=False).loss

    def observe(self, model, train_batch, infer_batch):
        output = model(
            **{k: v for k, v in train_batch.items() if k != "labels"}, use_cache=False
        )
        logits = output.logits[:, :, self.sample_ids][
            self.training_batch["attention_mask"].bool()
        ]
        generated = model.generate(
            **infer_batch,
            max_new_tokens=4,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
        )[:, infer_batch["input_ids"].shape[1] :].tolist()
        tokens = []
        for row in generated:
            if self.processor.eos_token_id in row:
                row = row[: row.index(self.processor.eos_token_id) + 1]
            tokens.append(row)
        return {
            "numerical": logits.flatten().tolist(),
            "outputs": tokens,
            "semantic_pass": tokens == self.gold,
        }

    def mutate(self, name, model, train, infer, stack):
        if name == "wrong_training_labels":
            train["labels"] = train["labels"].roll(1, dims=0)
        elif name == "wrong_country_token":
            for batch in (train, infer):
                batch["input_ids"][batch["input_ids"] == self.france] = self.germany
        elif name == "ignore_padding":
            for batch in (train, infer):
                batch["attention_mask"] = torch.ones_like(batch["attention_mask"])
        elif name == "remove_causal_mask":
            stack.enter_context(
                patch.object(modeling_llama, "create_causal_mask", return_value=None)
            )
        elif name == "disable_rope":
            stack.enter_context(
                patch.object(
                    modeling_llama,
                    "apply_rotary_pos_emb",
                    side_effect=lambda q, k, *a, **kw: (q, k),
                )
            )
        elif name == "zero_attention":
            stack.callback(
                model.model.layers[0]
                .self_attn.register_forward_hook(zero_attention)
                .remove
            )
        elif name in ("zero_mlp", "detach_mlp"):
            hook = (
                (lambda m, a, out: torch.zeros_like(out))
                if name == "zero_mlp"
                else (lambda m, a, out: out.detach())
            )
            stack.callback(model.model.layers[0].mlp.register_forward_hook(hook).remove)


class ClipTask:
    name = "clip"
    faults = (
        "swap_red_blue",
        "repeat_first_image",
        "repeat_first_text",
        "scale_pixels",
        "zero_attention",
        "detach_visual_projection",
    )
    required_gradients = ("visual_projection.weight", "text_projection.weight")

    def __init__(self, path):
        self.path = path
        self.processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
        self.texts = [
            f"a {color} square" for color in ("red", "green", "blue", "yellow")
        ]
        self.images = [
            Image.new("RGB", (256, 256), color)
            for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
        ]
        self.inference_batch = dict(
            self.processor(
                text=self.texts, images=self.images, padding=True, return_tensors="pt"
            )
        )
        self.training_batch = self.inference_batch

    def make_model(self):
        config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        config.projection_dim = 32
        for sub in (config.text_config, config.vision_config):
            sub.hidden_size = 64
            sub.intermediate_size = 128
            sub.num_hidden_layers = 2
            sub.num_attention_heads = 4
            sub.projection_dim = 32
        config._attn_implementation = "eager"
        return CLIPModel(config).float()

    def load_model(self, path, attn="eager"):
        return CLIPModel.from_pretrained(
            path,
            local_files_only=True,
            attn_implementation=attn,
            dtype=torch.float32,
        )

    def loss(self, model, batch):
        return model(**batch, return_loss=True).loss

    def observe(self, model, train_batch, infer_batch):
        scores = model(**infer_batch).logits_per_image
        outputs = {
            "image_to_text": scores.argmax(1).tolist(),
            "text_to_image": scores.argmax(0).tolist(),
        }
        return {
            "numerical": scores.flatten().tolist(),
            "outputs": outputs,
            "semantic_pass": all(value == list(range(4)) for value in outputs.values()),
        }

    def mutate(self, name, model, train, infer, stack):
        for batch in (train, infer):
            if name == "swap_red_blue":
                batch["pixel_values"] = batch["pixel_values"][:, [2, 1, 0]]
            elif name == "repeat_first_image":
                batch["pixel_values"] = batch["pixel_values"][:1].expand_as(
                    batch["pixel_values"]
                )
            elif name == "repeat_first_text":
                for key in ("input_ids", "attention_mask"):
                    batch[key] = batch[key][:1].expand_as(batch[key])
            elif name == "scale_pixels":
                batch["pixel_values"] = batch["pixel_values"] / 255
        if name == "zero_attention":
            stack.callback(
                model.vision_model.encoder.layers[0]
                .self_attn.register_forward_hook(zero_attention)
                .remove
            )
        elif name == "detach_visual_projection":
            stack.callback(
                model.visual_projection.register_forward_hook(
                    lambda m, a, out: out.detach()
                ).remove
            )


def no_dropout(config):
    """Training probes compare losses, so fixtures must train deterministically."""
    for key, value in config.to_dict().items():
        if isinstance(value, float) and (
            "dropout" in key or "layerdrop" in key or key.startswith("mask_")
        ):
            setattr(config, key, 0.0)
    return config


def zeros(module, args, output):
    return torch.zeros_like(output)


def detach_last_hidden_state(module, args, output):
    output.last_hidden_state = output.last_hidden_state.detach()
    return output


def sample_ids(vocab_size, gold):
    return sorted(
        set(
            torch.linspace(0, vocab_size - 1, 32).long().tolist()
            + [token for answer in gold for token in answer]
        )
    )


def two_tone_clips(sampling_rate=16000):
    """Four one-second clips, each a distinct pair of tones, so order matters."""
    pairs = ((220, 880), (440, 1320), (660, 330), (990, 550))
    half = torch.arange(sampling_rate // 2) / sampling_rate
    return [
        torch.cat([torch.sin(2 * math.pi * a * half), torch.sin(2 * math.pi * b * half)])
        .mul(0.5)
        .numpy()
        for a, b in pairs
    ]


class BertTask:
    name = "bert"
    faults = (
        "wrong_country_token",
        "ignore_padding",
        "wrong_token_type",
        "disable_position_embeddings",
        "zero_attention",
        "detach_mlp",
        "wrong_training_labels",
    )
    required_gradients = (
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.intermediate.dense.weight",
    )

    def __init__(self, path):
        self.path = path
        self.processor = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mask = self.processor.mask_token
        self.prompts = [
            f"the capital of {country} is {mask}."
            for country in ("france", "germany", "italy", "the united kingdom")
        ]
        self.gold = [
            self.processor.encode(a, add_special_tokens=False)
            for a in ("paris", "berlin", "rome", "london")
        ]
        self.inference_batch = dict(
            self.processor(self.prompts, padding=True, return_tensors="pt")
        )
        self.masked = self.inference_batch["input_ids"] == self.processor.mask_token_id
        labels = torch.full_like(self.inference_batch["input_ids"], -100)
        labels[self.masked] = torch.tensor([a[0] for a in self.gold])
        self.training_batch = {**self.inference_batch, "labels": labels}
        self.sample_ids = sample_ids(len(self.processor), self.gold)
        self.france = self.processor.encode("france", add_special_tokens=False)[0]
        self.germany = self.processor.encode("germany", add_special_tokens=False)[0]

    def make_model(self):
        config = no_dropout(AutoConfig.from_pretrained(self.path, local_files_only=True))
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        return BertForMaskedLM._from_config(
            config, attn_implementation="eager", dtype=torch.float32
        )

    def load_model(self, path, attn="eager"):
        return BertForMaskedLM.from_pretrained(
            path, local_files_only=True, attn_implementation=attn, dtype=torch.float32
        )

    def loss(self, model, batch):
        return model(**batch).loss

    def observe(self, model, train_batch, infer_batch):
        logits = model(**infer_batch).logits
        tokens = [[t] for t in logits[self.masked].argmax(-1).tolist()]
        return {
            "numerical": logits[:, :, self.sample_ids][
                self.inference_batch["attention_mask"].bool()
            ]
            .flatten()
            .tolist(),
            "outputs": tokens,
            "semantic_pass": tokens == self.gold,
        }

    def mutate(self, name, model, train, infer, stack):
        layer = model.bert.encoder.layer[0]
        for batch in (train, infer):
            if name == "wrong_country_token":
                batch["input_ids"][batch["input_ids"] == self.france] = self.germany
            elif name == "ignore_padding":
                batch["attention_mask"] = torch.ones_like(batch["attention_mask"])
            elif name == "wrong_token_type":
                batch["token_type_ids"] = torch.ones_like(batch["token_type_ids"])
        if name == "wrong_training_labels":
            train["labels"][self.masked] = train["labels"][self.masked].roll(1)
        elif name == "disable_position_embeddings":
            hook = model.bert.embeddings.position_embeddings.register_forward_hook(zeros)
            stack.callback(hook.remove)
        elif name == "zero_attention":
            stack.callback(
                layer.attention.self.register_forward_hook(zero_attention).remove
            )
        elif name == "detach_mlp":
            stack.callback(
                layer.intermediate.register_forward_hook(
                    lambda m, a, out: out.detach()
                ).remove
            )


class T5Task:
    name = "t5"
    config_cls = AutoConfig
    model_cls = T5ForConditionalGeneration
    faults = (
        "wrong_country_token",
        "ignore_padding",
        "disable_position_bias",
        "remove_causal_mask",
        "zero_cross_attention",
        "detach_encoder",
        "wrong_training_labels",
    )
    required_gradients = (
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "decoder.block.0.layer.1.EncDecAttention.q.weight",
    )

    def __init__(self, path):
        self.path = path
        self.processor = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.prompts = [
            f"capital of {country}:"
            for country in ("France", "Germany", "Italy", "the United Kingdom")
        ]
        self.gold = [
            self.processor.encode(a) for a in ("Paris", "Berlin", "Rome", "London")
        ]
        self.inference_batch = dict(
            self.processor(self.prompts, padding=True, return_tensors="pt")
        )
        labels = self.processor.pad({"input_ids": self.gold}, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
        self.training_batch = {**self.inference_batch, "labels": labels}
        self.sample_ids = sample_ids(len(self.processor), self.gold)
        self.france = self.processor.encode("France", add_special_tokens=False)[-1]
        self.germany = self.processor.encode("Germany", add_special_tokens=False)[-1]

    def make_model(self):
        config = no_dropout(self.config_cls.from_pretrained(self.path, local_files_only=True))
        config.d_model = 64
        config.d_ff = 128
        config.d_kv = 16
        config.num_layers = config.num_decoder_layers = 2
        config.num_heads = 4
        return self.model_cls._from_config(
            config, attn_implementation="eager", dtype=torch.float32
        )

    def load_model(self, path, attn="eager"):
        return self.model_cls.from_pretrained(
            path, local_files_only=True, attn_implementation=attn, dtype=torch.float32
        )

    def loss(self, model, batch):
        return model(**batch).loss

    def observe(self, model, train_batch, infer_batch):
        # Teacher-force the recorded answers so label faults only affect training.
        labels = self.training_batch["labels"]
        logits = model(
            input_ids=train_batch["input_ids"],
            attention_mask=train_batch["attention_mask"],
            decoder_input_ids=model._shift_right(labels),
        ).logits
        generated = model.generate(
            **infer_batch, max_new_tokens=4, do_sample=False, num_beams=1
        )[:, 1:].tolist()
        eos = self.processor.eos_token_id
        tokens = [row[: row.index(eos) + 1] if eos in row else row for row in generated]
        return {
            "numerical": logits[:, :, self.sample_ids][labels != -100].flatten().tolist(),
            "outputs": tokens,
            "semantic_pass": tokens == self.gold,
        }

    def mutate(self, name, model, train, infer, stack):
        for batch in (train, infer):
            if name == "wrong_country_token":
                batch["input_ids"][batch["input_ids"] == self.france] = self.germany
            elif name == "ignore_padding":
                batch["attention_mask"] = torch.ones_like(batch["attention_mask"])
        if name == "wrong_training_labels":
            train["labels"] = train["labels"].roll(1, dims=0)
        elif name == "disable_position_bias":
            bias = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
            stack.callback(bias.register_forward_hook(zeros).remove)
        elif name == "remove_causal_mask":
            stack.enter_context(
                patch.object(modeling_t5, "create_causal_mask", return_value=None)
            )
        elif name == "zero_cross_attention":
            stack.callback(
                model.decoder.block[0]
                .layer[1]
                .EncDecAttention.register_forward_hook(
                    lambda m, a, out: (torch.zeros_like(out[0]), *out[1:])
                )
                .remove
            )
        elif name == "detach_encoder":
            stack.callback(
                model.encoder.register_forward_hook(detach_last_hidden_state).remove
            )


class WhisperTask:
    name = "whisper"
    faults = (
        "repeat_first_audio",
        "reverse_audio",
        "scale_features",
        "zero_encoder_positions",
        "remove_causal_mask",
        "zero_cross_attention",
        "detach_encoder",
        "wrong_training_labels",
    )
    required_gradients = (
        "model.encoder.conv1.weight",
        "model.decoder.layers.0.encoder_attn.q_proj.weight",
    )

    def __init__(self, path):
        self.path = path
        self.processor = WhisperProcessor.from_pretrained(path, local_files_only=True)
        tokenizer = self.processor.tokenizer
        self.prefix = tokenizer.convert_tokens_to_ids(
            ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        )
        eot = tokenizer.eos_token_id
        self.gold = [
            tokenizer.encode(w, add_special_tokens=False) + [eot]
            for w in (" red", " green", " blue", " yellow")
        ]
        features = self.processor.feature_extractor(
            two_tone_clips(), sampling_rate=16000, return_tensors="pt"
        )["input_features"]
        self.inference_batch = {"input_features": features}
        full = torch.tensor([self.prefix + answer for answer in self.gold])
        labels = full[:, 1:].clone()
        labels[:, : len(self.prefix) - 1] = -100
        self.training_batch = {
            "input_features": features,
            "decoder_input_ids": full[:, :-1],
            "labels": labels,
        }
        self.sample_ids = sample_ids(len(tokenizer), self.gold)

    def make_model(self):
        config = no_dropout(AutoConfig.from_pretrained(self.path, local_files_only=True))
        config.d_model = 64
        config.encoder_ffn_dim = config.decoder_ffn_dim = 128
        config.encoder_layers = config.decoder_layers = 2
        config.num_hidden_layers = 2
        config.encoder_attention_heads = config.decoder_attention_heads = 4
        return WhisperForConditionalGeneration._from_config(
            config, attn_implementation="eager", dtype=torch.float32
        )

    def load_model(self, path, attn="eager"):
        return WhisperForConditionalGeneration.from_pretrained(
            path, local_files_only=True, attn_implementation=attn, dtype=torch.float32
        )

    def loss(self, model, batch):
        return model(**batch).loss

    def observe(self, model, train_batch, infer_batch):
        labels = self.training_batch["labels"]
        logits = model(
            input_features=train_batch["input_features"],
            decoder_input_ids=self.training_batch["decoder_input_ids"],
        ).logits
        # Greedy loop with a fixed task prefix; generate() adds Whisper-specific
        # language and timestamp logic that is outside this comparison.
        encoder = model.model.encoder(infer_batch["input_features"])
        ids = torch.tensor([self.prefix] * len(self.gold))
        for _ in range(4):
            step = model(encoder_outputs=encoder, decoder_input_ids=ids).logits
            ids = torch.cat([ids, step[:, -1:].argmax(-1)], dim=1)
        eot = self.processor.tokenizer.eos_token_id
        generated = ids[:, len(self.prefix) :].tolist()
        tokens = [row[: row.index(eot) + 1] if eot in row else row for row in generated]
        return {
            "numerical": logits[:, :, self.sample_ids][labels != -100].flatten().tolist(),
            "outputs": tokens,
            "semantic_pass": tokens == self.gold,
        }

    def mutate(self, name, model, train, infer, stack):
        for batch in (train, infer):
            features = batch["input_features"]
            if name == "repeat_first_audio":
                batch["input_features"] = features[:1].expand_as(features)
            elif name == "reverse_audio":
                batch["input_features"] = features.flip(-1)
            elif name == "scale_features":
                batch["input_features"] = features * 2
        if name == "wrong_training_labels":
            train["labels"] = train["labels"].roll(1, dims=0)
        elif name == "zero_encoder_positions":
            positions = model.model.encoder.embed_positions
            stack.callback(positions.register_forward_hook(zeros).remove)
        elif name == "remove_causal_mask":
            stack.enter_context(
                patch.object(modeling_whisper, "create_causal_mask", return_value=None)
            )
        elif name == "zero_cross_attention":
            stack.callback(
                model.model.decoder.layers[0]
                .encoder_attn.register_forward_hook(zero_attention)
                .remove
            )
        elif name == "detach_encoder":
            stack.callback(
                model.model.encoder.register_forward_hook(
                    detach_last_hidden_state
                ).remove
            )


class Wav2Vec2Task:
    name = "wav2vec2"
    faults = (
        "repeat_first_audio",
        "reverse_audio",
        "zero_positional_conv",
        "zero_attention",
        "detach_feature_projection",
        "wrong_training_labels",
    )
    required_gradients = (
        "wav2vec2.feature_extractor.conv_layers.0.conv.weight",
        "wav2vec2.encoder.layers.0.attention.q_proj.weight",
    )

    def __init__(self, path):
        self.path = path
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        self.words = ["RED", "GREEN", "BLUE", "YELLOW"]
        self.inference_batch = dict(
            self.processor(two_tone_clips(), sampling_rate=16000, return_tensors="pt")
        )
        labels = self.processor(text=self.words, padding=True, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(
            labels["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )
        self.training_batch = {**self.inference_batch, "labels": labels}

    def make_model(self):
        config = no_dropout(AutoConfig.from_pretrained(self.path, local_files_only=True))
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.conv_dim = (32,) * len(config.conv_dim)
        config.num_conv_pos_embeddings = 16
        config.num_conv_pos_embedding_groups = 4
        return Wav2Vec2ForCTC._from_config(
            config, attn_implementation="eager", dtype=torch.float32
        )

    def load_model(self, path, attn="eager"):
        return Wav2Vec2ForCTC.from_pretrained(
            path, local_files_only=True, attn_implementation=attn, dtype=torch.float32
        )

    def loss(self, model, batch):
        return model(**batch).loss

    def observe(self, model, train_batch, infer_batch):
        logits = model(**infer_batch).logits
        text = self.processor.batch_decode(logits.argmax(-1))
        return {
            "numerical": logits.flatten().tolist(),
            "outputs": text,
            "semantic_pass": text == self.words,
        }

    def mutate(self, name, model, train, infer, stack):
        for batch in (train, infer):
            values = batch["input_values"]
            if name == "repeat_first_audio":
                batch["input_values"] = values[:1].expand_as(values)
            elif name == "reverse_audio":
                batch["input_values"] = values.flip(-1)
        if name == "wrong_training_labels":
            train["labels"] = train["labels"].roll(1, dims=0)
        elif name == "zero_positional_conv":
            positions = model.wav2vec2.encoder.pos_conv_embed
            stack.callback(positions.register_forward_hook(zeros).remove)
        elif name == "zero_attention":
            stack.callback(
                model.wav2vec2.encoder.layers[0]
                .attention.register_forward_hook(zero_attention)
                .remove
            )
        elif name == "detach_feature_projection":
            stack.callback(
                model.wav2vec2.feature_projection.register_forward_hook(
                    lambda m, a, out: tuple(o.detach() for o in out)
                ).remove
            )


class UMT5Task(T5Task):
    name = "umt5"
    config_cls = UMT5Config  # the Hub config predates model_type
    model_cls = UMT5ForConditionalGeneration


class OlmoTask(TextTask):
    name = "olmo"


TASKS = {
    task.name: task
    for task in (
        TextTask, ClipTask, BertTask, T5Task, WhisperTask, Wav2Vec2Task, UMT5Task, OlmoTask
    )
}
# Hub sources for the added families; only configs and processors are used.
SOURCES.update(
    bert=("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594", ""),
    t5=("google-t5/t5-small", "df1b051c49625cf57a3d0d8d3863ed4d13564fe4", ""),
    whisper=("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", ""),
    wav2vec2=(
        "facebook/wav2vec2-base-960h",
        "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        "",
    ),
    # Sibling families used to replay historical regressions.
    umt5=("google/umt5-small", "8c63c2b77efbf8e41206a2c8d994846cc9392360", ""),
    olmo=("allenai/OLMo-1B-hf", "aee7752d9c08ee4775e9b0091426d8410e8f6a89", ""),
)


@contextlib.contextmanager
def mutation(task, model, fault):
    train = {key: value.clone() for key, value in task.training_batch.items()}
    infer = {key: value.clone() for key, value in task.inference_batch.items()}
    with contextlib.ExitStack() as stack:
        task.mutate(fault, model, train, infer, stack)
        yield train, infer


def observe(task, model, fault="none"):
    model.eval()
    with mutation(task, model, fault) as (train, infer), torch.inference_mode():
        result = task.observe(model, train, infer)
    if not torch.isfinite(torch.tensor(result["numerical"])).all():
        raise ValueError("non-finite outputs")
    return result


def compare(actual, baseline):
    a, b = torch.tensor(actual["numerical"]), torch.tensor(baseline["numerical"])
    return {
        "numerical_detected": a.shape != b.shape
        or not torch.allclose(a, b, atol=ATOL, rtol=RTOL),
        "recorded_output_detected": actual["outputs"] != baseline["outputs"],
        "semantic_detected": (
            not actual["semantic_pass"] if baseline["semantic_pass"] else None
        ),
        "max_absolute_delta": float((a - b).abs().max())
        if a.shape == b.shape
        else None,
    }


def training_probe(task, model, fault="none"):
    """Check gradient coverage and an SGD update, then restore all parameters."""
    model.train()
    saved = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    try:
        optimizer.zero_grad(set_to_none=True)
        with mutation(task, model, fault) as (batch, _):
            loss = task.loss(model, batch)
            loss.backward()
        params = dict(model.named_parameters())
        active = [p for p in params.values() if p.grad is not None]
        finite = (
            bool(torch.isfinite(loss))
            and bool(active)
            and all(bool(torch.isfinite(p.grad).all()) for p in active)
        )
        required = all(
            params[name].grad is not None
            and bool(torch.count_nonzero(params[name].grad))
            for name in task.required_gradients
        )
        optimizer.step()
        updated = any(not torch.equal(p, saved[name]) for name, p in params.items())
        finite = finite and all(bool(torch.isfinite(p).all()) for p in params.values())
        return {
            "pass": finite and required and updated,
            "loss": loss.item() if torch.isfinite(loss) else None,
            "finite": finite,
            "required_gradients": required,
            "updated": updated,
        }
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(saved[name])
        model.zero_grad(set_to_none=True)
        model.eval()


def train(task, model, max_steps):
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    stable = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = task.loss(model, task.training_batch)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            semantic = observe(task, model)["semantic_pass"]
            stable = stable + 1 if semantic and loss.item() < 0.05 else 0
            log(
                "training",
                family=task.name,
                step=step,
                loss=loss.item(),
                semantic_pass=semantic,
            )
            if stable == 2:
                break
    model.zero_grad(set_to_none=True)
    return {
        "seconds": time.perf_counter() - start,
        "steps": step,
        "loss": loss.item(),
        "converged": stable == 2,
    }


def save_fixture(task, model, path):
    start = time.perf_counter()
    model.save_pretrained(path)
    task.processor.save_pretrained(path)
    files = {}
    for file in sorted(path.iterdir()):
        if file.is_file():
            files[file.name] = {
                "bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
    return {
        "seconds": time.perf_counter() - start,
        "files": files,
        "bytes": sum(f["bytes"] for f in files.values()),
    }


def evaluate(task, model, fixture_path=None, gradients=True, source_path=None):
    baseline = observe(task, model)
    result = {"baseline": baseline, "controls": [], "faults": []}
    if gradients:
        start = time.perf_counter()
        result["baseline_training"] = training_probe(task, model)
        result["training_probe_seconds"] = time.perf_counter() - start
        if not result["baseline_training"]["pass"]:
            raise ValueError("unmodified fixture fails training probe")

    def loss_changed(probe):
        return probe["loss"] is None or not math.isclose(
            probe["loss"],
            result["baseline_training"]["loss"],
            rel_tol=RTOL,
            abs_tol=ATOL,
        )

    def control(actual, control_task=task, control_model=model):
        row = compare(actual, baseline)
        if gradients:
            probe = training_probe(control_task, control_model)
            row.update(
                training_detected=not probe["pass"], loss_detected=loss_changed(probe)
            )
        return row

    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        actual = observe(task, model)
        latencies.append(time.perf_counter() - start)
        result["controls"].append({"name": "repeat", **control(actual)})
    result["inference_median_seconds"] = statistics.median(latencies)
    # Reload with SDPA rather than calling set_attn_implementation(), which does
    # not reach T5-family encoder/decoder stacks (transformers#49372).
    sdpa_path = fixture_path or source_path
    if sdpa_path:
        sdpa_model = task.load_model(sdpa_path, attn="sdpa")
        result["controls"].append(
            {"name": "sdpa", **control(observe(task, sdpa_model), task, sdpa_model)}
        )
        del sdpa_model
    if fixture_path:
        reloaded = task.load_model(fixture_path)
        reloaded_task = type(task)(fixture_path)
        result["controls"].append(
            {
                "name": "save_reload",
                **control(observe(reloaded_task, reloaded), reloaded_task, reloaded),
            }
        )
        del reloaded, reloaded_task
    for fault in task.faults:
        start = time.perf_counter()
        row = {"name": fault}
        try:
            actual = observe(task, model, fault)
            row.update(smoke_detected=False, **compare(actual, baseline))
        except Exception as exc:
            row.update(
                smoke_detected=True,
                numerical_detected=True,
                recorded_output_detected=True,
                semantic_detected=True if baseline["semantic_pass"] else None,
                error=f"{type(exc).__name__}: {exc}",
            )
        if gradients:
            try:
                row["training"] = training_probe(task, model, fault)
                row["training_detected"] = not row["training"]["pass"]
                row["loss_detected"] = loss_changed(row["training"])
            except Exception as exc:
                row["training_detected"] = True
                row["loss_detected"] = True
                row["training_error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = time.perf_counter() - start
        result["faults"].append(row)
    # Synthetic faults must leave the shared model unchanged after cleanup.
    if compare(observe(task, model), baseline)["numerical_detected"]:
        raise ValueError("mutation cleanup changed the unmodified baseline")
    return result


def summarize(fixtures):
    result = {}
    for weights in ("random", "overfit"):
        selected = [f for f in fixtures if f["weights"] == weights]
        faults = [fault for f in selected for fault in f["evaluation"]["faults"]]
        controls = [
            control for f in selected for control in f["evaluation"]["controls"]
        ]
        result[weights] = {
            "fixture_count": len(selected),
            "fault_seed_cases": len(faults),
            "smoke_detected": sum(f["smoke_detected"] for f in faults),
            "numerical_detected": sum(f["numerical_detected"] for f in faults),
            "recorded_output_detected": sum(
                f["recorded_output_detected"] for f in faults
            ),
            "numerical_or_training_detected": sum(
                f["numerical_detected"] or f["training_detected"] for f in faults
            ),
            "numerical_or_training_or_loss_detected": sum(
                f["numerical_detected"] or f["training_detected"] or f["loss_detected"]
                for f in faults
            ),
            "control_cases": len(controls),
            "control_false_alarms": sum(
                c["numerical_detected"]
                or c["recorded_output_detected"]
                or c.get("training_detected", False)
                or c.get("loss_detected", False)
                for c in controls
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument(
        "--families", nargs="+", choices=list(TASKS), default=["text", "clip"]
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--real-checkpoints", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "smollm_clip.json"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(tempfile.gettempdir()) / "model-integration-contracts-fixtures",
        help="Directory for generated weights and processors (outside the checkout by default).",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    transformers.logging.set_verbosity_error()
    source_dir = Path(transformers.__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_changes = subprocess.check_output(
        ["git", "-C", str(source_dir), "diff", "HEAD", "--", "src/transformers"],
        text=True,
    )
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformers_commit": commit,
            "transformers_source_diff_sha256": hashlib.sha256(
                tracked_changes.encode()
            ).hexdigest(),
            "transformers_source_dirty": bool(tracked_changes),
            "threads": args.threads,
            "device": "cpu",
            "dtype": "float32",
            "atol": ATOL,
            "rtol": RTOL,
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "fixtures": [],
        "real_checkpoints": [],
    }
    # Editable-install metadata can lag the imported source version above.
    report["environment"]["distribution_versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "safetensors",
            "numpy",
            "Pillow",
            "torchvision",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_report():
        report["summary"] = summarize(report["fixtures"])
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for family in args.families:
        task_cls = TASKS[family]
        start = time.perf_counter()
        task = task_cls(snapshot(SOURCES[family]))
        processor_seconds = time.perf_counter() - start
        for seed in args.seeds:
            torch.manual_seed(seed)
            start = time.perf_counter()
            model = task.make_model()
            build_seconds = time.perf_counter() - start
            for weights in ("random", "overfit"):
                log("fixture_start", family=family, seed=seed, weights=weights)
                row = {
                    "family": family,
                    "seed": seed,
                    "weights": weights,
                    "source": SOURCES[family],
                    "processor_seconds": processor_seconds,
                    "model_build_seconds": build_seconds,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "config": model.config.to_dict(),
                }
                if weights == "overfit":
                    row["training"] = train(task, model, args.max_steps)
                folder = args.artifacts / family / str(seed) / weights
                row["artifact"] = save_fixture(task, model, folder)
                row["evaluation"] = evaluate(task, model, folder)
                report["fixtures"].append(row)
                write_report()
                log(
                    "fixture_done",
                    family=family,
                    seed=seed,
                    weights=weights,
                    semantic_pass=row["evaluation"]["baseline"]["semantic_pass"],
                    inference_seconds=row["evaluation"]["inference_median_seconds"],
                )
            del model
            gc.collect()
        if args.real_checkpoints and family in ("text", "clip"):
            source = SOURCES[family] if family == "text" else CLIP_REFERENCE
            log("reference_start", family=family, source=source)
            model = task.load_model(snapshot(source))
            reference_task = task_cls(snapshot(source))
            report["real_checkpoints"].append(
                {
                    "family": family,
                    "source": source,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "evaluation": evaluate(
                        reference_task, model, gradients=False, source_path=snapshot(source)
                    ),
                }
            )
            write_report()
            del model, reference_task
            gc.collect()
    log("complete", output=str(args.output))


if __name__ == "__main__":
    main()
