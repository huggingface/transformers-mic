"""Replay #48108 using encoder-decoder and decoder-only Whisper assistants.

Prompt controls use Whisper's public prompt_ids argument. Logits-processor observers return the original processors.
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
    WhisperForConditionalGeneration,
    WhisperForCausalLM,
    WhisperProcessor,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["checkpoint", "overfit", "random"], required=True
    )
    parser.add_argument("--random-fixture", type=Path)
    parser.add_argument("--speech", type=Path)
    parser.add_argument("--build", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
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
    tokenizer.set_prefix_tokens(
        language="english", task="transcribe", predict_timestamps=False
    )
    sr = 16000
    half = torch.arange(sr // 2) / sr
    tone = (
        torch.cat([torch.sin(2 * math.pi * f * half) for f in (220, 880)])
        .mul(0.5)
        .numpy()
    )
    features = processor(tone, sampling_rate=sr, return_tensors="pt").input_features.to(
        args.device
    )
    silence = processor(
        tone * 0, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(args.device)
    causal = (
        WhisperForCausalLM.from_pretrained(
            folder, local_files_only=True, attn_implementation="eager"
        )
        .eval()
        .to(args.device)
    )
    encoder_decoder = (
        WhisperForConditionalGeneration.from_pretrained(
            folder, local_files_only=True, attn_implementation="eager"
        )
        .eval()
        .to(args.device)
    )
    observed = []
    for role, current in [
        ("main", model),
        ("encoder_decoder", encoder_decoder),
        ("causal", causal),
    ]:
        original = current._get_logits_processor

        def wrap(original, role):
            def inspect(*args, **kwargs):
                result = original(*args, **kwargs)
                observed.append(
                    dict(
                        role=role,
                        processors=[
                            dict(
                                name=type(p).__name__,
                                begin_index=getattr(p, "begin_index", None),
                                suppress_tokens=(
                                    p.suppress_tokens.cpu().tolist()
                                    if torch.is_tensor(
                                        getattr(p, "suppress_tokens", None)
                                    )
                                    else getattr(p, "suppress_tokens", None)
                                ),
                            )
                            for p in result
                        ],
                    )
                )
                return result

            return inspect

        current._get_logits_processor = wrap(original, role)
    rows = []
    cases = [
        ("tone", features, None),
        ("silence", silence, None),
        ("red_prompt", features, " red"),
        ("three_word_prompt", features, " red green blue"),
    ]
    if args.speech:
        import soundfile as sf

        speech, rate = sf.read(args.speech, dtype="float32")
        speech_features = processor(
            speech, sampling_rate=rate, return_tensors="pt"
        ).input_features.to(args.device)
        cases.append(("speech", speech_features, None))
    for name, audio, text in cases:
        for assistant_name, assistant in [
            ("greedy", None),
            ("causal", causal),
            ("encoder_decoder", encoder_decoder),
        ]:
            observed.clear()
            set_seed(0)
            kwargs = dict(
                language="en", task="transcribe", max_new_tokens=64, do_sample=False
            )
            if assistant is not None:
                kwargs["assistant_model"] = assistant
            if text is not None:
                kwargs["prompt_ids"] = processor.get_prompt_ids(
                    text, return_tensors="pt"
                ).to(args.device)
            start = time.perf_counter()
            with torch.no_grad():
                result = model.generate(audio, **kwargs)
            rows.append(
                dict(
                    case=name,
                    assistant=assistant_name,
                    tokens=result.cpu().tolist(),
                    text=processor.batch_decode(result, skip_special_tokens=True),
                    processors=list(observed),
                    seconds=time.perf_counter() - start,
                )
            )
            print(name, assistant_name, rows[-1]["text"], flush=True)
    output = dict(
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
        ),
        speech_sha256=None
        if args.speech is None
        else hashlib.sha256(args.speech.read_bytes()).hexdigest(),
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1) + "\n")


if __name__ == "__main__":
    main()
