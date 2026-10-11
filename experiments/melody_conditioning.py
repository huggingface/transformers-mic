"""Compare MusicGen Melody conditioning with real, overfit, and random weights.

Run in separate processes with each archived Transformers build first on
PYTHONPATH. Build the random twin with example/build_fixtures.py --keep-random
--out first. Results retain generated codes and teacher-forced loss; input
perturbations are sensitivity controls, not historical regressions.

    python experiments/melody_conditioning.py --model overfit --build <sha> --output <file>
    python experiments/melody_conditioning.py --model random --random-fixture <folder> \
        --build <sha> --output <file>
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
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, set_seed

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "80s pop track with bassy drums and synth",
    "90s rock song with loud guitars and heavy drums",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("checkpoint", "overfit", "random"), required=True
    )
    parser.add_argument("--random-fixture", type=Path)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance", type=float, nargs="+", default=[1, 3])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--cases", nargs="+", help="run only these named input cases")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--skip-loss", action="store_true", help="generation-only checkpoint probes"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.model == "random" and args.random_fixture is None:
        parser.error("--random-fixture is required for random weights")

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    set_seed(0)
    lock = json.loads(
        (ROOT / "example/transformers/tests/contracts/fleet.lock").read_text()
    )
    contract = lock["contracts"]["MusicgenMelody"]
    repo = (
        "facebook/musicgen-melody"
        if args.model == "checkpoint"
        else "hf-internal-testing/proto-musicgen-melody-slim"
    )
    revision = (
        contract["checkpoints"][repo]
        if args.model == "checkpoint"
        else contract["fixtures"][repo]["revision"]
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
    if args.model != "checkpoint":
        build = json.loads((folder / "build.json").read_text())
        if build["weights"] != ("random" if args.model == "random" else "overfit"):
            raise ValueError("Fixture weight kind does not match --model")
        expected = build["sha256"]["model.safetensors"]
        if (
            hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("Fixture weights do not match build.json")
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    model = (
        MusicgenMelodyForConditionalGeneration.from_pretrained(
            folder, local_files_only=True
        )
        .eval()
        .to(args.device)
    )
    # Observe the codes actually sent to EnCodec. Score argmaxes also contain
    # positions later removed by the delay mask; do not mistake those for audio.
    decoded_codes = []
    decode = model.audio_encoder.decode

    def observe_decode(audio_codes, *decode_args, **decode_kwargs):
        decoded_codes.append(audio_codes.detach().cpu().tolist())
        return decode(audio_codes, *decode_args, **decode_kwargs)

    model.audio_encoder.decode = observe_decode
    rate = processor.feature_extractor.sampling_rate
    half = torch.arange(rate // 2) / rate
    melodies = [
        torch.cat([torch.sin(2 * math.pi * f * half) for f in pair]).mul(0.5).numpy()
        for pair in [(262, 330), (392, 494)]
    ]
    inputs = processor(
        audio=melodies,
        sampling_rate=rate,
        text=PROMPTS,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(args.device)
    inputs["input_features"] = inputs["input_features"].to(model.dtype)
    shared = processor(
        audio=melodies,
        sampling_rate=rate,
        text=[PROMPTS[0]] * 2,
        padding=True,
        return_tensors="pt",
    )
    shared = shared.to(args.device)
    shared["input_features"] = shared["input_features"].to(model.dtype)
    cases = {
        "original": dict(inputs),
        "swapped_melodies": {
            **inputs,
            "input_features": inputs["input_features"].flip(0),
        },
        "dropped_melody": {k: v for k, v in inputs.items() if k != "input_features"},
        "swapped_text": {
            **inputs,
            "input_ids": inputs["input_ids"].flip(0),
            "attention_mask": inputs["attention_mask"].flip(0),
        },
        "shared_text": dict(shared),
        "shared_text_swapped_melodies": {
            **shared,
            "input_features": shared["input_features"].flip(0),
        },
    }
    if args.cases:
        unknown = set(args.cases) - cases.keys()
        if unknown:
            parser.error(f"Unknown input cases: {sorted(unknown)}")
        cases = {name: cases[name] for name in args.cases}
    labels = torch.randint(
        0,
        model.decoder.config.vocab_size,
        (2, 8, model.decoder.num_codebooks),
        generator=torch.Generator().manual_seed(0),
    ).to(args.device)
    result = {
        "model": args.model,
        "build": args.build,
        "source": {
            "repo": repo,
            "revision": revision if args.model != "random" else None,
            "fixture_key": build["key"] if args.model != "checkpoint" else None,
        },
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "threads": args.threads,
        },
        "inputs": {
            "prompts": PROMPTS,
            "note_hz": [[262, 330], [392, 494]],
            "seconds": 1,
            "label_seed": 0,
            "max_new_tokens": args.max_new_tokens,
        },
        "cases": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for guidance in args.guidance:
        for name, batch in cases.items():
            start = time.perf_counter()
            decoded_codes.clear()
            with torch.no_grad():
                set_seed(0)
                generated = model.generate(
                    **batch,
                    do_sample=False,
                    guidance_scale=guidance,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                codes = (
                    torch.stack(generated.scores)
                    .argmax(-1)
                    .reshape(len(generated.scores), 2, -1)
                    .permute(1, 2, 0)
                )
                loss = (
                    None
                    if args.skip_loss
                    else model(**batch, labels=labels).loss.item()
                )
            key = f"guidance={guidance:g}/{name}"
            if args.device == "cuda":
                torch.cuda.synchronize()
            result["cases"][key] = {
                "codes": codes.tolist(),
                "decoded_codes": list(decoded_codes),
                "audio_shape": list(generated.sequences.shape),
                "audio_head": generated.sequences[:, 0, :32].float().tolist(),
                "teacher_forced_loss": loss,
                "seconds": time.perf_counter() - start,
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "build": args.build,
                        "case": key,
                        "loss": loss,
                        "seconds": result["cases"][key]["seconds"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
