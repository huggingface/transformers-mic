"""Measure one OpenAI-compatible chat server the same way for every engine (stdlib only).

Streams a greedy chat completion and records client-side TTFT, decode tokens/s, generated text,
the kernel's system-wide page-in counter (/proc/vmstat pgpgin, includes every engine's storage reads)
and GPU memory sampled by nvidia-smi during the request.
"""

import argparse
import json
import os
import threading
import subprocess
import time
import urllib.request
from pathlib import Path


def pgpgin_bytes():
    if not Path("/proc/vmstat").exists():
        return None
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        if key == "pgpgin":
            return int(value) * 1024
    return None


def gpu_used_bytes():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        # Leave out the GPU balloon (run_engine.sh's simulated smaller GPU): only what the engine itself uses.
        return int(out.strip().splitlines()[0]) * 1024**2 - int(os.environ.get("GPU_BALLOON_BYTES", "0"))
    except Exception:
        return None


def meminfo():
    if not Path("/proc/meminfo").exists():
        return {}
    info = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":")
        info[key] = int(value.split()[0]) * 1024
    return {
        k: info.get(k)
        for k in ("MemTotal", "MemAvailable", "Cached", "AnonPages", "Mlocked")
    }


class GpuPeak:
    """Sample nvidia-smi while a request runs; the peak includes every process on the GPU."""

    def __init__(self):
        self.peak, self._stop = None, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used = gpu_used_bytes()
            if used is not None:
                self.peak = max(self.peak or 0, used)
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def evict(paths):
    if not hasattr(os, "posix_fadvise"):
        return None
    """Drop clean page-cache pages of model files (unprivileged posix_fadvise DONTNEED); mapped pages stay."""
    total = 0
    for root in paths:
        files = (
            [root]
            if os.path.isfile(root)
            else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        )
        for path in files:
            real = os.path.realpath(path)
            if os.path.isfile(real):
                fd = os.open(real, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    total += os.fstat(fd).st_size
                finally:
                    os.close(fd)
    return total


def wait_ready(base, timeout, api_key=None, server_pid=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_pid is not None and not Path(f"/proc/{server_pid}").exists():
            raise RuntimeError(f"server process {server_pid} exited before it was ready; see engine.log")
        for path in ("/health", "/v1/models"):
            try:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                with urllib.request.urlopen(
                    urllib.request.Request(base + path, headers=headers), timeout=5
                ) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
        time.sleep(5)
    raise TimeoutError(f"server at {base} not ready after {timeout}s")


def request(base, model, prompt, max_tokens, extra, api_key=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    before_io, before_mem = pgpgin_bytes(), meminfo()
    chunks, text, usage = [], [], None
    start = time.perf_counter()
    with GpuPeak() as gpu, urllib.request.urlopen(req, timeout=7200) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content") or ""
                if delta:
                    chunks.append(time.perf_counter() - start)
                    text.append(delta)
    total = time.perf_counter() - start
    completion = (usage or {}).get("completion_tokens") or len(chunks)
    decode_s = (chunks[-1] - chunks[0]) if len(chunks) > 1 else None
    return {
        "time_to_first_token_s": chunks[0] if chunks else None,
        "total_s": total,
        "completion_tokens": completion,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "stream_chunks": len(chunks),
        "decode_tokens_per_s": (completion - 1) / decode_s if decode_s else None,
        "pgpgin_bytes": None if before_io is None else pgpgin_bytes() - before_io,
        "meminfo_before": before_mem,
        "meminfo_after": meminfo(),
        "gpu_used_peak_bytes": gpu.peak,
        "generated_text": "".join(text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="model")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument(
        "--extra",
        default="{}",
        help="engine-specific JSON body fields, e.g. ignore_eos",
    )
    parser.add_argument(
        "--started-at",
        type=float,
        help="epoch seconds when the server process was launched",
    )
    parser.add_argument(
        "--meta", default="{}", help="JSON provenance merged into the report"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--server-pid", type=int, help="fail as soon as this process exits")
    parser.add_argument(
        "--evict",
        nargs="*",
        default=[],
        help="model files to drop from page cache once ready",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wait_ready(args.base, args.ready_timeout, args.api_key, args.server_pid)
    report = {
        "engine": args.engine,
        "ready_after_launch_s": None
        if args.started_at is None
        else time.time() - args.started_at,
        "gpu_used_bytes_idle": gpu_used_bytes(),
        "evicted_bytes_after_ready": evict(args.evict),
        "meminfo_idle": meminfo(),
        "prompt": args.prompt_file.read_text(encoding="utf-8"),
        "max_tokens": args.max_tokens,
        **json.loads(args.meta),
        "runs": [],
    }
    for _ in range(args.runs):
        report["runs"].append(
            request(
                args.base,
                args.model,
                report["prompt"],
                args.max_tokens,
                json.loads(args.extra),
                args.api_key,
            )
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
