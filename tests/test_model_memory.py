"""验证 DynamicVRAM 加载排序与原 ComfyUI 顺序一致。"""

import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    from comfyui_rs._native import model_loading_order
except ImportError:
    model_loading_order = None


PATCHER = Path(__file__).resolve().parents[1] / "comfyui/comfy/model_patcher.py"


def load_sort(native_order):
    """读取 ComfyUI 实际使用的排序入口，隔离 Torch 的设备初始化。"""
    source = ast.parse(PATCHER.read_text(encoding="utf-8"))
    sort = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sort_dynamic_loading"
    )
    scope = {"_rust_model_loading_order": native_order}
    exec(compile(ast.Module(body=[sort], type_ignores=[]), str(PATCHER), "exec"), scope)
    return scope["_sort_dynamic_loading"]


class DynamicLoadingOrderTests(unittest.TestCase):
    @unittest.skipIf(model_loading_order is None, "需要构建指定版本的 comfyui-rs 原生扩展")
    def test_rust_sorts_all_keys_and_keeps_original_entries(self):
        """验证 Rust 排序覆盖四个字段并保留原有引用。"""
        candidates = [
            (True, -20, 4, "a", object(), {}),
            (False, -10, 6, "层", object(), {}),
            (False, -10, 6, "a", object(), {}),
            (False, -12, 8, "a", object(), {}),
            (False, -10, 4, "a", object(), {}),
        ]
        expected = sorted(candidates)
        native_order = Mock(wraps=model_loading_order)
        load_sort(native_order)(candidates)
        native_order.assert_called_once()
        self.assertEqual(len(candidates), len(expected))
        for actual, reference in zip(candidates, expected):
            self.assertIs(actual, reference)

    @unittest.skipIf(model_loading_order is None, "需要构建指定版本的 comfyui-rs 原生扩展")
    def test_duplicate_keys_compare_remaining_fields_in_python(self):
        """四项排序键相同时，剩余字段仍参与原有比较。"""
        candidates = [(False, -2, 1, "层", 2, {}), (False, -2, 1, "层", 1, {})]
        expected = sorted(candidates)
        self.assertIsNone(model_loading_order([candidate[:4] for candidate in candidates]))
        load_sort(model_loading_order)(candidates)
        self.assertEqual(candidates, expected)
