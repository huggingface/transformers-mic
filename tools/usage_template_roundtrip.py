"""Round-trip the catalog contracts through usage_template: extract, render, check.

For each catalog contract, runs the approved usage.md and then the page rendered
from its extracted slots against the reviewed baselines on this machine, in every
variant this platform has baselines for, and reports whether the two agree.
The approved pages are restored afterwards, whatever happens.

    python tools/usage_template_roundtrip.py [--contract T5 ...]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "example"))

from transformers_mic import usage_template  # noqa: E402
import run  # noqa: E402

LOCK = ROOT / "example/transformers/tests/contracts/fleet.lock"
EXPECTATIONS = ROOT / "example/transformers/tests/contracts/expectations"
# Checkpoint-only contracts have no fixture to run.
CHECKPOINTS = {"WhisperSpeech"}


def variants(name):
    platform = EXPECTATIONS / name / run.platform_class("cpu")
    found = {"default"}
    for path in platform.glob("*.json"):
        parts = path.name.split(".")
        if len(parts) == 3:
            found.add(parts[1])
    return sorted(found)


def check(name, variant):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.jsonl"
        options = ["--framework", "transformers", "--local-contracts", "--contract", name,
                   "--variant", variant, "--output", str(out)]
        if name not in CHECKPOINTS:
            options.append("--fixtures-only")
        subprocess.run([sys.executable, str(ROOT / "example/run.py"), *options],
                       cwd=ROOT, capture_output=True, text=True)
        rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    labels = sorted({row.get("contract_revision") for row in rows if "target" in row})
    return {row["target"]: (row["status"], row.get("failures", [])) for row in rows if "target" in row}, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="append")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    report = {}
    for name, contract in lock["contracts"].items():
        if args.contract and name not in args.contract:
            continue
        folder = run.source_folder(LOCK.parent, contract["integration"]["repo"], contract["integration"])
        path = folder / "integrations/transformers/usage.md"
        approved = path.read_text()
        checkpoint = next(iter(json.loads((folder / "integration.json").read_text())["checkpoints"]))
        try:
            rendered = usage_template.render(usage_template.extract(approved), checkpoint)
        except Exception as error:  # a page the template cannot express
            report[name] = {"template": f"cannot express: {error!r}"}
            print(name, report[name], flush=True)
            continue
        report[name] = {}
        for variant in variants(name):
            before, approved_labels = check(name, variant)
            try:
                path.write_text(rendered)
                after, rendered_labels = check(name, variant)
            finally:
                path.write_text(approved)
            # A changed page runs as "local"; one that keeps the pinned label rendered byte for byte.
            identical = rendered == approved or rendered_labels == approved_labels
            same = before == after and (identical or rendered_labels == ["local"])
            report[name][variant] = {"same": same, "identical": identical, "approved": before, "rendered": after}
            print(name, variant, "SAME" if same else "DIFFERENT", "(byte-identical)" if identical else "",
                  {t: s for t, (s, _) in before.items()}, "->", {t: s for t, (s, _) in after.items()}, flush=True)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
