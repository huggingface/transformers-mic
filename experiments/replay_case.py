"""Check frozen fixtures on the bad and fixed builds of one historical regression.

    python replay_case.py <fix-sha> [bad-sha]

Defaults the bad build to <fix-sha>^. Builds are extracted with git archive from
$TRANSFORMERS_REPO (never checked out) into $REPLAY_WORKDIR/builds and put first
on PYTHONPATH, followed by $OVERLAY (older dependency pins). $FROZEN selects the
fixtures frozen by replay.py, $ATTN the attention
implementation to check, $CHECKS the cache directory for check results.
"""
import json, subprocess, sys, os, tempfile
from pathlib import Path
S = Path(os.environ.get("REPLAY_WORKDIR", Path(tempfile.gettempdir()) / "model-integration-replay"))
TR = Path(os.environ.get("TRANSFORMERS_REPO", Path.home() / "Dev/transformers"))
PY = Path(os.environ.get("PYTHON", sys.executable))
HUB = Path(__file__).parent

def sha(ref):
    return subprocess.check_output(["git", "-C", TR, "rev-parse", "--short=10", ref], text=True).strip()

def build(ref):
    d = S / "builds" / ref
    if not d.exists():
        d.mkdir(parents=True)
        subprocess.run(f"git -C {TR} archive {ref} src | tar -x -C {d}", shell=True, check=True)
    return d

def check(ref):
    out = S / os.environ.get("CHECKS", "checks") / f"{ref}.json"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        env = {**os.environ, "PYTHONPATH": ":".join(filter(None, [str(build(ref) / "src"), os.environ.get("OVERLAY")]))}
        subprocess.run([PY, HUB / "replay.py", "check", "--fixtures", Path(os.environ.get("FROZEN", S / "frozen")),
                        "--build", ref, "--attn", os.environ.get("ATTN", "eager"), "--output", out], env=env, cwd=HUB, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text())["rows"]

def flags(row):
    return [k[:-9] for k in ("smoke_detected", "numerical_detected", "recorded_output_detected",
                             "training_detected", "loss_detected") if row.get(k)]

fix = sha(sys.argv[1]); bad = sha(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1] + "^")
rows = {"bad": check(bad), "fix": check(fix)}
print(f"fix={fix} bad={bad}")
for b, f in zip(rows["bad"], rows["fix"]):
    fb, ff = flags(b), flags(f)
    if fb or ff:
        print(f"  {b['family']:8} {b['weights']:7} {str(b['seed']):4} bad={fb} fix={ff} "
              f"delta={b.get('max_absolute_delta')} err={b.get('error','')[:120]}")
print("  clean on both" if not any(flags(r) for v in rows.values() for r in v) else "")
