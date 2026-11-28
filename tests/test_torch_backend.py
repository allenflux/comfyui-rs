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
        candle = self.site / "candle_torch-0.1.0.dist-info"
        candle.mkdir()
        (candle / "METADATA").write_text("Name: candle-torch\nVersion: 0.1.0\n", encoding="utf-8")
        (candle / "RECORD").write_text("torch/__init__.py,,\n", encoding="utf-8")

    def add_torch_package(self, parent, backend):
        package = parent / "torch"
        package.mkdir()
        (package / "__init__.py").write_text(f"backend = {backend!r}\n", encoding="utf-8")

    def launch(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(self.install / "main.py"), "--torch-backend", "candle", *args],
            cwd=self.caller, env={**os.environ, "PYTHONPATH": str(self.caller)},
            capture_output=True, text=True,
        )

    def test_launch_uses_candle_wheel_and_preserves_caller_arguments(self):
        self.add_torch_package(self.caller, "shadow")
        result = self.launch("--base-directory", "relative-data", "--cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertEqual(response["backend"], "candle")
        self.assertEqual(Path(response["cwd"]).resolve(), self.caller.resolve())
        self.assertEqual(response["argv"], ["--base-directory", "relative-data", "--cpu"])

    def test_launch_rejects_original_torch_distribution(self):
        original = self.site / "torch-2.15.0.dist-info"
        original.mkdir()
        (original / "METADATA").write_text("Name: torch\nVersion: 2.15.0\n", encoding="utf-8")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("原版 torch", result.stderr)

    def test_launch_rejects_package_shadowed_by_entry_directory(self):
        self.add_torch_package(self.source, "shadow")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("torch 导入被其他路径覆盖", result.stderr)

    def test_build_candle_skips_stock_torch_dependencies(self):
        requirements = self.source / "requirements.txt"
        requirements.write_text("torch\n", encoding="utf-8")
        with (
            patch.object(build, "ROOT", self.install),
            patch.object(build, "REQUIREMENTS", requirements),
            patch.object(sys, "argv", ["build.py", "--torch-backend", "candle"]),
            patch.object(build.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run,
        ):
            build.main()
        calls = [call.args[0] for call in run.call_args_list]
        self.assertFalse(any("-r" in args for args in calls))
        self.assertIn(
            [str(self.python), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(self.install)],
            calls,
        )
