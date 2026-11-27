"""验证模型加载规划与原 ComfyUI 行为一致。"""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    from comfyui_rs._native import model_loading_order
except ImportError:
    model_loading_order = None


PATCHER = Path(__file__).resolve().parents[1] / "comfyui/comfy/model_patcher.py"
HAS_TORCH = importlib.util.find_spec("torch") is not None


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


def get_load_list(torch):
    """提取实际遍历方法，仅隔离完整的运行时和驱动初始化。"""
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelPatcher")
    load_list = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_list")

    def module_size(module):
        """使用实际张量大小实现模块计量。"""
        return sum(tensor.nbytes for tensor in module.state_dict().values())

    scope = {"torch": torch, "comfy": SimpleNamespace(model_management=SimpleNamespace(module_size=module_size))}
    exec(compile(ast.Module(body=[load_list], type_ignores=[]), str(PATCHER), "exec"), scope)
    return scope["_load_list"]


@unittest.skipUnless(HAS_TORCH, "请在项目构建环境中运行模型加载规划测试")
class LoadListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载 CPU 上的 PyTorch 和 ComfyUI 遍历方法。"""
        import torch

        cls.torch = torch
        cls.load_list = staticmethod(get_load_list(torch))

    def test_stock_leaf_keeps_weight_order_size_and_references(self):
        """普通叶模块保留经典和 DynamicVRAM 元组的内容及参数引用。"""
        torch = self.torch
        leaf = torch.nn.Linear(8, 4)
        model = torch.nn.Sequential(torch.nn.Identity(), leaf)
        patcher = SimpleNamespace(model=model, patches={})
        size = leaf.weight.nbytes + leaf.bias.nbytes
        for dynamic, keys in ((False, (size,)), (True, (False, -size))):
            with self.subTest(dynamic=dynamic):
                plan = self.load_list(patcher, for_dynamic=dynamic)
                self.assertEqual(len(plan), 1)
                self.assertEqual(plan[0][:-1], (*keys, size, "1", leaf))
                params = plan[0][-1]
                self.assertEqual(list(params), ["weight", "bias"])
                self.assertIs(params["weight"], leaf.weight)
                self.assertIs(params["bias"], leaf.bias)

    def test_custom_recursive_parameters_can_exclude_a_leaf(self):
        """没有注册子层的自定义模块仍可报告仅递归枚举可见的参数。"""
        torch = self.torch

        class CustomLinear(torch.nn.Linear):
            def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
                """为递归扫描添加自定义参数名称。"""
                yield from super().named_parameters(prefix, recurse, remove_duplicate)
                if recurse:
                    yield "virtual.weight", self.weight

        model = torch.nn.Sequential(CustomLinear(8, 4))
        self.assertEqual(self.load_list(SimpleNamespace(model=model, patches={})), [])

    def test_custom_named_modules_can_supply_a_virtual_child(self):
        """参数枚举沿用自定义模块遍历所提供的虚拟子层。"""
        torch = self.torch

        class VirtualParent(torch.nn.Linear):
            def named_modules(self, memo=None, prefix="", remove_duplicate=True):
                """报告一个不在注册子模块字典中的模块。"""
                yield from super().named_modules(memo, prefix, remove_duplicate)
                virtual_prefix = f"{prefix}.virtual" if prefix else "virtual"
                yield virtual_prefix, virtual

        virtual = torch.nn.Linear(4, 4)
        model = torch.nn.Sequential(VirtualParent(8, 4))
        plan = self.load_list(SimpleNamespace(model=model, patches={}))
        self.assertEqual([entry[2] for entry in plan], ["0.virtual"])
        self.assertIs(plan[0][3], virtual)

    def test_parent_parameters_still_move_to_default_dtype(self):
        """包含自身参数和子层的父模块仍按指定设备与类型处理自身参数。"""
        torch = self.torch
        parent = torch.nn.Module()
        parent.direct = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        parent.direct_comfy_model_dtype = torch.float64
        parent.child = torch.nn.Linear(4, 4)
        patcher = SimpleNamespace(model=parent, patches={})

        plan = self.load_list(patcher, default_device=torch.device("cpu"))

        self.assertEqual([entry[2] for entry in plan], ["child"])
        self.assertEqual(parent.direct.dtype, torch.float64)
        self.assertEqual(parent.child.weight.dtype, torch.float32)


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
