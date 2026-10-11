"""Replay #48028 with frozen real, overfit, and random weights.

Use an archived Transformers src on PYTHONPATH. The controlled long-form
case constrains only the first segment's transcript to make previous-token
lengths unequal; subsequent segments are unconstrained. Observers record
actual decoder masks/positions and prefill logits without altering them.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import (
    LogitsProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]


class FirstSegment(LogitsProcessor):
    def __init__(self, schedules, initial_length, pad, previous):
        self.schedules = schedules
        self.initial_length = initial_length
        self.pad = pad
        self.previous = previous
        self.calls = 0

    def __call__(self, input_ids, scores):
        # Later segments start with <|startofprev|> (possibly left-padded).
        if any(self.previous in row.tolist() for row in input_ids):
            return scores
        step = input_ids.shape[-1] - self.initial_length
        if step < 0:
            raise ValueError("Unexpected first-segment prefix length")
        self.calls += 1
        forced = torch.full_like(scores, -float("inf"))
        for row, schedule in enumerate(self.schedules):
            token = schedule[step] if step < len(schedule) else self.pad
            forced[row, token] = 0
        return forced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["checkpoint", "overfit", "random"], required=True
    )
    parser.add_argument("--random-fixture", type=Path)
    parser.add_argument("--build", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model == "random" and args.random_fixture is None:
        parser.error("--random-fixture is required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    set_seed(0)
    lock = json.loads(
        (ROOT / "example/transformers/tests/contracts/fleet.lock").read_text()
    )["contracts"]["Whisper"]
    repo = (
        "openai/whisper-tiny"
        if args.model == "checkpoint"
        else "hf-internal-testing/proto-whisper-tiny-slim"
    )
    revision = (
        lock["checkpoints"][repo]
        if args.model == "checkpoint"
        else lock["fixtures"][repo]["revision"]
    )
    folder = (
        args.random_fixture
        if args.model == "random"
        else Path(
            snapshot_download(
                repo,
                revision=revision,
                local_files_only=True,
                allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
            )
        )
    )
    weight_hash = hashlib.sha256(
        (folder / "model.safetensors").read_bytes()
    ).hexdigest()
    if args.model != "checkpoint":
        build = json.loads((folder / "build.json").read_text())
        assert build["weights"] == args.model
        assert weight_hash == build["sha256"]["model.safetensors"]
    processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            folder,
            local_files_only=True,
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )
    tokenizer = processor.tokenizer
    tid = tokenizer.convert_tokens_to_ids
    sot, previous, pad = (
        tid("<|startoftranscript|>"),
        tid("<|startofprev|>"),
        model.config.pad_token_id,
    )
    initial = [sot, tid("<|en|>"), tid("<|transcribe|>")]
    timestamp = tid("<|0.00|>")
    words = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in [" red", " green blue yellow"]
    ]
    probe_ids = sorted(
        set(
            list(range(256))
            + sum(words, [])
            + initial
            + [previous, pad]
            + list(range(timestamp, timestamp + 1501, 50))
        )
    )
    traces = []

    def observe(module, inputs, kwargs, output):
        ids = kwargs.get("decoder_input_ids")
        if ids is None or ids.shape[-1] <= 1:
            return
        mask = kwargs.get("decoder_attention_mask")
        positions = kwargs.get("decoder_position_ids")
        traces.append(
            {
                "decoder_input_ids": ids.detach().cpu().tolist(),
                "decoder_attention_mask": None
                if mask is None
                else mask.detach().cpu().tolist(),
                "decoder_position_ids": None
                if positions is None
                else positions.detach().cpu().tolist(),
                "last_logits_probe": output.logits[:, -1, probe_ids]
                .float()
                .detach()
                .cpu()
                .tolist(),
                "last_logits_argmax": output.logits[:, -1]
                .argmax(-1)
                .detach()
                .cpu()
                .tolist(),
            }
        )

    handle = model.register_forward_hook(observe, with_kwargs=True)
    sr = 16000
    half = torch.arange(sr // 2) / sr
    audio = [
        torch.cat([torch.sin(2 * math.pi * f * half) for f in pair]).mul(0.5).numpy()
        for pair in [(220, 880), (440, 1320)]
    ]
    short = processor(audio, sampling_rate=sr, return_tensors="pt").to(args.device)
    long = processor(
        [a.repeat(36) for a in audio],
        sampling_rate=sr,
        padding="longest",
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(args.device)
    # Caller-supplied decoder_input_ids is supported (deprecated) by Whisper's
    # public generate(), and reaches the same preparation path as long-form.
    prompts = [[previous] + w + initial + [tid("<|notimestamps|>")] for w in words]
    width = max(map(len, prompts))
    padded = torch.tensor(
        [[pad] * (width - len(p)) + p for p in prompts], device=args.device
    )
    mask = padded.ne(pad)
    schedules = [[timestamp] + w + [timestamp + 1500, pad] for w in words]
    cases = ["short_unpadded", "short_left_padded", "long_unforced", "long_controlled"]
    rows = []
    for case in args.cases or cases:
        if case not in cases:
            raise ValueError(case)
        traces.clear()
        set_seed(0)
        kwargs = dict(
            language="en",
            task="transcribe",
            max_new_tokens=12,
            do_sample=False,
            temperature=0.0,
        )
        control = None
        if case == "short_left_padded":
            kwargs.update(decoder_input_ids=padded, decoder_attention_mask=mask)
        if case.startswith("long"):
            kwargs.update(
                return_timestamps=True,
                condition_on_prev_tokens=True,
                return_segments=True,
            )
        if case == "long_controlled":
            control = FirstSegment(schedules, len(initial), pad, previous)
            kwargs["logits_processor"] = [control]
        start = time.perf_counter()
        with torch.no_grad():
            result = model.generate(
                **(long if case.startswith("long") else short), **kwargs
            )
        sequences = result["sequences"] if isinstance(result, dict) else result
        segments = None
        if isinstance(result, dict):
            segments = [
                [
                    {
                        k: (v.detach().cpu().tolist() if torch.is_tensor(v) else v)
                        for k, v in segment.items()
                        if k in ["start", "end", "tokens"]
                    }
                    for segment in row
                ]
                for row in result["segments"]
            ]
        left_padded = [
            t
            for t in traces
            if t["decoder_attention_mask"] is not None
            and any(not row[0] for row in t["decoder_attention_mask"])
        ]
        rows.append(
            dict(
                case=case,
                sequences=sequences.detach().cpu().tolist(),
                segments=segments,
                traces=list(traces),
                left_padded_prefills=len(left_padded),
                forced_first_segment_calls=None if control is None else control.calls,
                seconds=time.perf_counter() - start,
            )
        )
        print(case, "left-padded prefills:", len(left_padded), flush=True)
    handle.remove()
    record = dict(
        model=args.model,
        repo=repo,
        revision=revision,
        weight_sha256=weight_hash,
        build=args.build,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            transformers_file=transformers.__file__,
            device=args.device,
            gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            attention="eager",
            dtype=str(model.dtype),
        ),
        probe_token_ids=probe_ids,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
