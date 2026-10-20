"""Mutable workspace inputs, immutable build artifacts and project provenance."""

import hashlib
import json
from pathlib import Path
import shutil

from . import resources


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def catalog_patches(catalog, name):
    needed = {
        "Whisper": {"config-field-generation-params", "whisper-decode-tensor-batch"},
        "WhisperSpeech": {
            "config-field-generation-params",
            "whisper-decode-tensor-batch",
        },
        "Wav2Vec2": {"masked-spec-embed-init"},
    }.get(name, set())
    return {
        key: value
        for key, value in catalog.get("local_patches", {}).items()
        if key in needed
    }


class Workspace:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.state_path = self.root / ".mic/project.json"
        self.lock_path = self.root / "example/transformers/tests/contracts/fleet.lock"

    @property
    def state(self):
        if not self.state_path.exists():
            raise ValueError(
                f"No MIC workspace at {self.root}; run tmic create MODEL --out {self.root}"
            )
        return read(self.state_path)

    @property
    def lock(self):
        return read(self.lock_path)

    @property
    def contract_root(self):
        entry = self.lock["contracts"][self.state["contract"]]["integration"]
        return (self.lock_path.parent / entry["path"]).resolve()

    @property
    def manifest(self):
        return read(self.contract_root / "integration.json")

    @property
    def fixture(self):
        return self.manifest["checkpoints"][self.state["model"]]["fixture"]

    @property
    def expectations(self):
        if (self.contract_root / ".mic-repo.json").exists():
            return self.contract_root / "integrations/transformers/expectations"
        return self.lock_path.parent / "expectations" / self.state["contract"]

    def update_layout(self):
        if not (self.contract_root / ".mic-repo.json").exists():
            return
        files = [
            p.relative_to(self.contract_root).as_posix()
            for p in self.contract_root.rglob("*")
            if p.is_file()
            and p.name not in ("README.md", ".gitattributes", ".mic-repo.json")
            and "expectations" not in p.relative_to(self.contract_root).parts
        ]
        write(
            self.contract_root / ".mic-repo.json",
            {"format": "mic-umbrella-v1", "contract_files": sorted(files)},
        )

    def save_state(self, **updates):
        state = self.state
        state.update(updates)
        write(self.state_path, state)

    def ensure_new(self):
        if self.state_path.exists():
            raise ValueError(f"Workspace already exists: {self.root}; use tmic update")

    def from_catalog(self, model, name, weights, readable):
        self.ensure_new()
        catalog = read(resources.catalog_lock())
        from .hub import upgrade_contract, umbrella_references

        original = catalog["contracts"][name]
        entry = upgrade_contract(original)
        source = resources.catalog() / "hub" / f"{name}-integration"
        upgraded = entry["integration"]["repo"] != original["integration"]["repo"]
        root = (
            self.root
            / "example/hub"
            / (entry["integration"]["repo"].split("/")[-1] if upgraded else source.name)
        )
        shutil.copytree(source, root, dirs_exist_ok=True)
        entry["integration"]["path"] = str(Path("../../../hub") / root.name)
        if upgraded:
            write(
                root / "integration.json",
                umbrella_references(read(root / "integration.json")),
            )
            files = [
                p.relative_to(root).as_posix()
                for p in root.rglob("*")
                if p.is_file()
                and p.name not in ("README.md", ".gitattributes")
                and "expectations" not in p.relative_to(root).parts
            ]
            write(
                root / ".mic-repo.json",
                {"format": "mic-umbrella-v1", "contract_files": sorted(files)},
            )
        for repo in original.get("fixtures", {}):
            fixture_source = resources.catalog() / "hub" / repo.split("/")[-1]
            shutil.copytree(
                fixture_source,
                self.root / "example/hub" / fixture_source.name,
                dirs_exist_ok=True,
            )
        shutil.copytree(
            resources.catalog_expectations() / name,
            root / "integrations/transformers/expectations"
            if upgraded
            else self.lock_path.parent / "expectations" / name,
            dirs_exist_ok=True,
        )
        write(
            self.lock_path,
            {
                "framework": "transformers",
                "contracts": {name: entry},
                "environment": catalog["environment"],
                "local_patches": catalog_patches(catalog, name),
            },
        )
        write(
            self.state_path,
            {
                "model": model,
                "contract": name,
                "weights": weights,
                "readable_output": readable,
                "origin": "catalog",
                "validated": False,
            },
        )

    def from_plan(
        self, model, revision, namespace, name, manifest, usage, weights, readable
    ):
        self.ensure_new()
        repo = next(iter(manifest["fixtures"]))
        root = self.root / "example/hub" / repo.split("/")[-1]
        entry = manifest["integrations"]["transformers"]["entry_point"]
        (root / entry).parent.mkdir(parents=True, exist_ok=True)
        (root / entry).write_text(usage)
        write(root / "integration.json", manifest)
        write(
            root / ".mic-repo.json",
            {
                "format": "mic-umbrella-v1",
                "contract_files": ["integration.json", entry],
            },
        )
        (root / "README.md").write_text(
            f"---\ntags: [mic, test-artifact]\nbase_model: {model}\n---\n\n# {manifest['name']}\n\nGenerated usage draft; expected outputs must be measured and reviewed.\n"
        )
        fixtures = {
            repo: {
                "revision": "unpublished",
                "path": f"../../../hub/{repo.split('/')[-1]}",
            }
            for repo in manifest["fixtures"]
        }
        catalog = read(resources.catalog_lock())
        write(
            self.lock_path,
            {
                "framework": "transformers",
                "environment": catalog["environment"],
                "contracts": {
                    name: {
                        "integration": {
                            "repo": repo,
                            "revision": "unpublished",
                            "path": f"../../../hub/{repo.split('/')[-1]}",
                        },
                        "fixtures": fixtures,
                        "checkpoints": {model: revision},
                    }
                },
            },
        )
        write(
            self.state_path,
            {
                "model": model,
                "contract": name,
                "weights": weights,
                "readable_output": readable,
                "origin": "llm",
                "validated": False,
            },
        )

    def snapshot(self):
        files = {
            str(p.relative_to(self.contract_root)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in self.contract_root.rglob("*")
            if p.is_file()
        }
        target = self.root / ".mic/history" / digest(files)
        if not target.exists():
            shutil.copytree(self.contract_root, target)
        return target

    def validation_fingerprint(self):
        paths = [self.lock_path]
        paths += [
            p
            for p in self.contract_root.rglob("*")
            if p.is_file() and p.name not in ("README.md", ".gitattributes")
        ]
        paths += list(self.expectations.rglob("*.json"))
        return digest(
            {
                str(p.relative_to(self.root)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in paths
            }
        )

    def contract_digest(self):
        files = {
            p.relative_to(self.contract_root).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in self.contract_root.rglob("*")
            if p.is_file()
            and p.name not in ("README.md", ".gitattributes", ".mic-repo.json")
            and "expectations" not in p.relative_to(self.contract_root).parts
            and not {"node_modules", ".cache", ".git"}.intersection(
                p.relative_to(self.contract_root).parts
            )
        }
        return digest(files)
