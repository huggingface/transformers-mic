"""Bundle the tested backend and catalog from their canonical repository sources."""

from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent


class BuildPy(build_py):
    def run(self):
        super().run()
        package = Path(self.build_lib) / "transformers_mic"
        backend = package / "_backend"
        if backend.exists():
            shutil.rmtree(backend)
        backend.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / "example").glob("*.py"):
            shutil.copy2(source, backend / source.name)
        for name in ("sweep.py", "layer_patterns.py"):
            shutil.copy2(ROOT / "experiments" / name, backend / name)
        shutil.copytree(
            ROOT / "example/schemas", backend / "schemas", dirs_exist_ok=True
        )
        catalog = package / "_catalog"
        if catalog.exists():
            shutil.rmtree(catalog)
        for source in (ROOT / "example/hub").glob("*-integration"):
            shutil.copytree(
                source,
                catalog / "hub" / source.name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("node_modules", "__pycache__", ".cache"),
            )
        for source in (ROOT / "example/hub").glob("*-slim"):
            target = catalog / "hub" / source.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ("README.md", "build.json"):
                if (source / name).exists():
                    shutil.copy2(source / name, target / name)
        shutil.copytree(
            ROOT / "example/transformers/tests/contracts",
            catalog / "contracts",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.candidate.json", "__pycache__"),
        )


setup(cmdclass={"build_py": BuildPy})
