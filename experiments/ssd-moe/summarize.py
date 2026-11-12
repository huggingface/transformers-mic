"""Merge per-engine results into one Markdown table (stdout) and summary.json."""

import json
import sys
from pathlib import Path

ORDER = [
    "transformers-streaming-q8_0",
    "transformers-streaming",
    "transformers-offload",
    "vllm",
    "llamacpp-bf16",
    "llamacpp-q8_0",
    "strata",
]
WEIGHTS = {
    "transformers-streaming-q8_0": "Q8_0 GGUF experts (Unsloth) + BF16 dense",
    "llamacpp-q8_0": "Q8_0 GGUF (Unsloth)",
    "strata": "Q8_0 GGUF (Unsloth)",
    "llamacpp-bf16": "BF16 GGUF (converted)",
}


def gib(value):
    return "-" if value is None else f"{value / 1024**3:.1f}"


def num(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def load(root, engine):
    path = root / engine / "result.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for run in report.get("runs", []):
        rows.append(
            {
                "ttft": run.get("time_to_first_token_s"),
                "decode": run.get("decode_tokens_per_s"),
                "tokens": run.get("completion_tokens")
                or len(run.get("generated_token_ids", [])),
                "read": run.get("pgpgin_bytes", run.get("kernel_storage_read_bytes")),
                "gpu": run.get(
                    "gpu_used_peak_bytes", report.get("device_peak_allocated_bytes")
                ),
                "text": run.get("generated_text", ""),
            }
        )
    return {"report": report, "runs": rows}


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    root = Path(sys.argv[1])
    results = {engine: load(root, engine) for engine in ORDER}
    present = [engine for engine in ORDER if results[engine] and results[engine]["runs"]]
    reference = results[present[0]]["runs"][0]["text"] if present else ""
    lines = [
        "| Engine | Weights | Run | TTFT s | Decode tok/s | Tokens | Storage read GiB | GPU peak GiB | Same text as the first row |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for engine in ORDER:
        result = results[engine]
        if result is None and not (root / engine).exists():
            continue  # not requested in this run
        if result is None:
            lines.append(
                f"| {engine} | {WEIGHTS.get(engine, 'BF16 safetensors')} | - | failed or missing | | | | | |"
            )
            continue
        for index, run in enumerate(result["runs"], 1):
            prefix = common_prefix(reference, run["text"]) if reference else 0
            same = (
                "yes"
                if run["text"] == reference and reference
                else f"first {prefix} chars"
            )
            lines.append(
                f"| {engine} | {WEIGHTS.get(engine, 'BF16 safetensors')} | {index} | {num(run['ttft'])} | "
                f"{num(run['decode'], 3)} | {run['tokens']} | {gib(run['read'])} | {gib(run['gpu'])} | {same} |"
            )
    print("\n".join(lines))
    print(
        "\nStorage read: Transformers = this process's /proc/self/io read_bytes; servers = system-wide pgpgin."
    )
    print(
        "GPU peak: Transformers = torch max_memory_allocated; servers = nvidia-smi memory.used (whole GPU)."
    )
    for engine in ORDER:
        if results[engine] and results[engine]["runs"]:
            print(f"\n**{engine}**: {results[engine]['runs'][0]['text'][:300]!r}")
    (root / "summary.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
