import json
import os
import shutil
import subprocess
import sys
import unittest
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import build


ROOT = Path(__file__).resolve().parents[1]


class TorchBackendTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        work = Path(temp.name).resolve()
        self.install = work / "install"
        self.source = self.install / "comfyui"
        self.source.mkdir(parents=True)
        self.caller = work / "caller"
        self.caller.mkdir()
        for filename in ("build.py", "main.py", "run.py", "torch_backend.py"):
            shutil.copyfile(ROOT / filename, self.install / filename)
        (self.source / "main.py").write_text(
            "import json, os, sys, torch\n"
            "print(json.dumps(dict(backend=torch.backend, cwd=os.getcwd(), argv=sys.argv[1:])))\n",
            encoding="utf-8",
        )
        self.env = self.install / ".venv-candle"
        venv.create(self.env, with_pip=False)
        self.python = self.env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.site = Path(subprocess.check_output(
            [str(self.python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip())
        self.add_torch_package(self.site, "candle")
        bundled = self.site / "candle"
        bundled.mkdir()
        (bundled / "__init__.py").touch()
        candle = self.site / "candle_torch-0.1.0.dist-info"
        candle.mkdir()
        (candle / "METADATA").write_text("Name: candle-torch\nVersion: 0.1.0\n", encoding="utf-8")
        (candle / "RECORD").write_text("torch/__init__.py,,\ncandle/__init__.py,,\n", encoding="utf-8")

    def add_torch_package(self, parent, backend):
        package = parent / "torch"
        package.mkdir()
        (package / "__init__.py").write_text(f"backend = {backend!r}\n", encoding="utf-8")

    def launch(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(self.install / "main.py"), *args],
            cwd=self.caller, env={**os.environ, "PYTHONPATH": str(self.caller)},
            capture_output=True, text=True,
        )

    def test_default_launch_uses_candle_wheel_and_preserves_caller_arguments(self):
        self.add_torch_package(self.caller, "shadow")
        result = self.launch("--base-directory", "relative-data", "--cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertEqual(response["backend"], "candle")
        self.assertEqual(Path(response["cwd"]).resolve(), self.caller.resolve())
        self.assertEqual(response["argv"], ["--base-directory", "relative-data", "--cpu"])

    def test_explicit_candle_flag_is_accepted_but_pytorch_is_rejected(self):
        self.assertEqual(self.launch("--torch-backend", "candle").returncode, 0)
        result = self.launch("--torch-backend", "pytorch")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_launch_rejects_original_torch_distributions(self):
        for name in ("torch", "torchvision", "torchaudio"):
            with self.subTest(package=name):
                original = self.site / f"{name}-1.0.dist-info"
                original.mkdir()
                (original / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n", encoding="utf-8")
                result = self.launch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"原版 {name}", result.stderr)
                shutil.rmtree(original)

    def test_launch_rejects_package_shadowed_by_entry_directory(self):
        self.add_torch_package(self.source, "shadow")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("torch 导入被其他路径覆盖", result.stderr)

    def test_build_installs_compatibility_requirements_without_resolving_torch(self):
        requirements = self.source / "requirements.txt"
        compat = self.source / "requirements-candle-compat.txt"
        requirements.touch()
        compat.touch()
        with (
            patch.object(build, "ROOT", self.install),
            patch.object(build, "REQUIREMENTS", requirements),
            patch.object(build, "COMPAT_REQUIREMENTS", compat),
            patch.object(sys, "argv", ["build.py"]),
            patch.object(build.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
        ):
            build.main()
        calls = [call.args[0] for call in run.call_args_list]
        pip_install = [str(self.python), "-m", "pip", "install"]
        self.assertIn([*pip_install, "--only-binary", "candle-torch", "-r", str(requirements)], calls)
        self.assertIn([*pip_install, "--no-deps", "-r", str(compat)], calls)
        self.assertIn(
            [str(self.python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(self.install)],
            calls,
        )
