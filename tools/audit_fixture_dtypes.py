"""List published fixtures whose per-tensor dtypes differ from what the builder now saves.

    python tools/audit_fixture_dtypes.py                       # control + prototype fleets at main
    python tools/audit_fixture_dtypes.py --fleet prototypes --revision <dataset sha>

For each fixture pinned by the fleet lock, reads only headers: the fixture's
model.safetensors header, its build.json, and its source checkpoint's config,
index and safetensors headers (`built_from`). Each floating fixture tensor is
expected in the float dtype the source stores its pattern in when that is not the
source's main float dtype (numeric path components wildcarded, unanimous only),
else in the source's default dtype
(build_fixtures.source_dtype, which now reads sub-config dtypes). A fixture is
listed when any tensor differs; rebuilding it would change its bytes and key.
Nothing is downloaded in full, built or published.
"""

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from transformers_mic import fleet
from transformers_mic.tensor_dtypes import (
    FLOAT,
    checkpoint_dtypes,
    config_dtype,
    kept_dtypes,
    main_dtype,
    mixed,
    pattern,
)

API = HfApi()


def fixtures(names, revision):
    """{(fixture repo, revision): [contract, ...]} across the fleets."""
    found = collections.defaultdict(list)
    for name in names:
        path, _ = fleet.download(name, revision)
        for contract, entry in json.loads(Path(path).read_text())["contracts"].items():
            for repo, pin in entry["fixtures"].items():
                found[(repo, pin["revision"])].append(f"{name}:{contract}")
    return found


def audit(repo, revision):
    files = API.list_repo_files(repo, revision=revision)
    if "model.safetensors" not in files:
        weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
        return {"fixture": repo, "skipped": f"no root model.safetensors ({weights[:3]})"}
    build = json.loads(Path(hf_hub_download(repo, "build.json", revision=revision)).read_text())
    if build.get("builder") != "example/build_fixtures.py":
        return {"fixture": repo, "skipped": f"built by {build.get('builder')}, not example/build_fixtures.py"}
    source, source_revision = build["built_from"]["repo"], build["built_from"]["revision"]
    source_config = json.loads(Path(hf_hub_download(source, "config.json", revision=source_revision)).read_text())
    default = config_dtype(source_config) or "float32"
    stored = checkpoint_dtypes(source, source_revision)
    kept = kept_dtypes(stored)
    seen = collections.defaultdict(set)
    for name, value in stored.items():
        seen[pattern(name)].add(value["dtype"])
    header = API.parse_safetensors_file_metadata(repo, "model.safetensors", revision=revision, timeout=60)
    fixture = {name: {"dtype": info.dtype, "numel": info.parameter_count} for name, info in header.tensors.items()}
    changes = collections.defaultdict(lambda: {"fixture": set(), "tensors": 0})
    defaulted = collections.defaultdict(lambda: {"tensors": 0, "patterns": set()})
    for name, info in fixture.items():
        if info["dtype"] not in FLOAT:
            continue  # integer buffers are never cast
        key = pattern(name)
        expected = kept.get(key, default)
        actual = FLOAT[info["dtype"]]
        if actual == expected:
            continue
        if key in kept:
            row = changes[key]
            row["fixture"].add(actual)
            row["tensors"] += 1
            row["source"] = sorted(seen[key])
            row["expected"] = expected
        else:
            group = defaulted[(actual, expected)]
            group["tensors"] += 1
            group["patterns"].add(key)
    covers = {}
    for covered, covered_revision in (build.get("covers") or {}).items():
        if covered != source:
            other = checkpoint_dtypes(covered, covered_revision)
            if other and kept_dtypes(other) != kept:
                covers[covered] = "keeps other per-tensor dtypes than built_from: the covered keys would differ"
    return {
        "fixture": repo,
        "revision": revision,
        "source": f"{source}@{source_revision[:12]}",
        "source_has_headers": bool(stored),
        "source_main": main_dtype(stored),
        "source_mixed": mixed(stored),
        "fixture_main": main_dtype(fixture),
        "default": default,
        "kept": {
            key: {
                "source": row["source"],
                "fixture": sorted(row["fixture"]),
                "expected": row["expected"],
                "tensors": row["tensors"],
            }
            for key, row in sorted(changes.items())
        },
        "defaulted": [
            {
                "fixture": actual,
                "expected": expected,
                "tensors": group["tensors"],
                "patterns": sorted(group["patterns"]),
            }
            for (actual, expected), group in sorted(defaulted.items())
        ],
        **({"covers": covers} if covers else {}),
    }


def describe(row, verbose=False):
    lines = [
        f"{row['fixture']}@{row['revision'][:12]}  <-  {row['source']} "
        f"(source main {row['source_main']}, fixture main {row['fixture_main']})"
    ]
    for group in row["defaulted"]:
        lines.append(
            f"  default dtype: {group['tensors']} tensors in {len(group['patterns'])} patterns "
            f"{group['fixture']} -> {group['expected']}"
            + (" (config dtype read from a sub-config)" if row["fixture_main"] != row["default"] else "")
        )
        if verbose:
            lines.extend(f"    {key}" for key in group["patterns"])
    for key, change in row["kept"].items():
        lines.append(
            f"  {key}: source {'/'.join(change['source'])}, fixture {'/'.join(change['fixture'])} "
            f"-> {change['expected']} ({change['tensors']} tensors)"
        )
    for covered, note in (row.get("covers") or {}).items():
        lines.append(f"  covers {covered}: {note}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fleet", action="append", choices=sorted(fleet.FILES), help="default: every fleet")
    parser.add_argument("--revision", default="main", help="fleet dataset revision")
    parser.add_argument("--json", action="store_true", help="one JSON line per fixture instead of text")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="also list the patterns that move with the default")
    args = parser.parse_args()
    revision = fleet.resolve(args.revision)
    found = fixtures(args.fleet or sorted(fleet.FILES), revision)

    def safe(item):
        try:
            return audit(*item)
        except Exception as error:  # report and go on: one unreadable source must not hide the rest
            return {"fixture": item[0], "error": f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(safe, sorted(found)))
    changed = [r for r in rows if "error" not in r and "skipped" not in r and (r["kept"] or r["defaulted"])]
    if args.json:
        for row in rows:
            print(json.dumps(row, default=sorted))
        return 0
    print(f"# fleet dataset {fleet.REPOSITORY}@{revision}, fleets {', '.join(args.fleet or sorted(fleet.FILES))}")
    print(
        f"# {len(rows)} fixtures; {len(changed)} would change; "
        f"{sum('skipped' in r for r in rows)} skipped; {sum('error' in r for r in rows)} unreadable\n"
    )
    for row in changed:
        print(describe(row, args.verbose) + "\n")
    unchanged = [
        r
        for r in rows
        if "error" not in r
        and "skipped" not in r
        and r not in changed
        and r["source_main"] not in (None, r["fixture_main"])
    ]
    if unchanged:
        print(
            "# unchanged although the fixture's dtype is not the source's stored one: the fixture is in the dtype "
            "the config loads, and only dtypes relative to the main one are mirrored"
        )
        for row in unchanged:
            print(
                f"  {row['fixture']}  <-  {row['source']}: stored {row['source_main']}"
                f"{' (mixed)' if row['source_mixed'] else ''}, fixture {row['fixture_main']}"
            )
        print()
    no_headers = [r for r in rows if "error" not in r and "skipped" not in r and not r["source_has_headers"]]
    if no_headers:
        print("# sources with no safetensors headers (pytorch_model.bin): fixture default only")
        for row in no_headers:
            print(f"  {row['fixture']}  <-  {row['source']}")
        print()
    for row in rows:
        if "skipped" in row:
            print(f"# skipped {row['fixture']}: {row['skipped']}")
        if "error" in row:
            print(f"# unreadable {row['fixture']}: {row['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
