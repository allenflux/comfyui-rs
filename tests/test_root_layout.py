import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "comfyui"

# 使用真实参数解析和路径设置，只替换 YAML 读取，避免导入推理组件。
PROBE = """
import ast
import itertools
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from comfy import options
options.enable_args_parsing()
from comfy.cli_args import args
import folder_paths

source = Path(folder_paths.__file__).parent
entry = source / "main.py"
tree = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
apply_paths = next(node for node in tree.body if getattr(node, "name", None) == "apply_custom_paths")
configs = []
scope = dict(os=os, args=args, logging=logging, folder_paths=folder_paths, itertools=itertools,
             utils=SimpleNamespace(extra_config=SimpleNamespace(
                 load_extra_path_config=lambda path: configs.append(str(Path(path).resolve())))))
exec(compile(ast.Module(body=[apply_paths], type_ignores=[]), str(entry), "exec"), scope)
scope["apply_custom_paths"]()
nodes = folder_paths.get_folder_paths("custom_nodes")
print(json.dumps(dict(
    base=folder_paths.base_path, models=folder_paths.models_dir, nodes=nodes,
    model_configs=folder_paths.get_folder_paths("configs"), configs=configs,
    input=folder_paths.get_input_directory(), output=folder_paths.get_output_directory(),
    user=folder_paths.get_user_directory(), previews=folder_paths.get_previews_directory(),
    temp=folder_paths.get_temp_directory(),
    shadowed=folder_paths.bundled_custom_node_is_shadowed(
        str(source / "custom_nodes/websocket_image_save.py"), nodes),
)))
"""


class RootLayoutTests(unittest.TestCase):
    def setUp(self):
        """隔离安装目录和调用目录，不接触用户已有的模型或工作流。"""
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        work = Path(temp.name).resolve()
        self.install = work / "install"
        self.caller = work / "caller"
        self.install.mkdir()
        self.caller.mkdir()

    def probe(self, *flags):
        """在独立进程模拟根目录启动，保留调用者的当前目录。"""
        env = {**os.environ, "PYTHONPATH": str(SOURCE), "COMFYUI_RS_INSTALL_ROOT": str(self.install)}
        result = subprocess.run(
            [sys.executable, "-B", "-", *flags], cwd=self.caller, env=env,
            input=PROBE, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_data_directories_and_bundled_assets(self):
        """数据从安装根目录解析，内置模型配置和节点作为补充。"""
        (self.install / "extra_model_paths.yaml").touch()
        result = self.probe()
        self.assertEqual(result["base"], str(self.install))
        for directory in ("models", "input", "output", "temp", "previews", "user"):
            self.assertEqual(result[directory], str(self.install / directory))
        self.assertEqual(result["nodes"], [str(self.install / "custom_nodes"), str(SOURCE / "custom_nodes")])
        self.assertEqual(result["model_configs"], [str(self.install / "models/configs"), str(SOURCE / "models/configs")])
        self.assertEqual(result["configs"], [str(self.install / "extra_model_paths.yaml")])
        self.assertFalse(result["shadowed"])

    def test_explicit_paths_are_relative_to_caller(self):
        """显式路径优先，默认 YAML 仍放在安装根目录。"""
        (self.install / "extra_model_paths.yaml").touch()
        (self.caller / "selected-models").mkdir()
        result = self.probe(
            "--base-directory", "data", "--models-directory", "selected-models",
            "--output-directory", "selected-output", "--extra-model-paths-config", "mine.yaml",
        )
        self.assertEqual(result["base"], str(self.caller / "data"))
        self.assertEqual(result["models"], str(self.caller / "selected-models"))
        self.assertEqual(result["output"], str(self.caller / "selected-output"))
        self.assertEqual(result["input"], str(self.caller / "data/input"))
        self.assertEqual(result["configs"], [str(self.install / "extra_model_paths.yaml"), str(self.caller / "mine.yaml")])

    def test_root_node_or_disabled_copy_shadows_bundled_node(self):
        """用户迁入的同名节点及其禁用状态优先于内置副本。"""
        nodes = self.install / "custom_nodes"
        nodes.mkdir()
        custom = nodes / "websocket_image_save.py"
        custom.touch()
        self.assertTrue(self.probe()["shadowed"])
        custom.rename(nodes / "websocket_image_save.py.disabled")
        self.assertTrue(self.probe()["shadowed"])
