import ast
import importlib
import importlib.util
import os
import random
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/comfyui_rs/__init__.py"
MODEL_MANAGEMENT = Path("comfy/model_management.py")
BUNDLED_MODEL_MANAGEMENT = ROOT / "comfyui" / MODEL_MANAGEMENT
MODE_VAR = "COMFYUI_RS_EVICTION_MODE"


def python_order(candidates):
    """用 Python 生成稳定排序索引，作为原生结果的对照。"""
    return sorted(range(len(candidates)), key=candidates.__getitem__)


def installed_native():
    """未构建扩展时，让其他测试仍可运行。"""
    try:
        return importlib.import_module("comfyui_rs._native")
    except ImportError:
        return None


@contextmanager
def adapter(mode, native_order):
    """重新导入适配层，使每轮测试独立读取启动配置。"""
    spec = importlib.util.spec_from_file_location("comfyui_rs", PACKAGE)
    module = importlib.util.module_from_spec(spec)
    native = ModuleType("comfyui_rs._native")
    native.assess_budget = Mock()
    native.eviction_order = native_order
    with (
        patch.dict(sys.modules, {spec.name: module, native.__name__: native}),
        patch.dict(os.environ),
    ):
        if mode is None:
            os.environ.pop(MODE_VAR, None)
        else:
            os.environ[MODE_VAR] = mode
        spec.loader.exec_module(module)
        yield module.sort_eviction_candidates


class EvictionModeTests(unittest.TestCase):
    def setUp(self):
        """准备部分排序字段相同的候选项，覆盖后续字段的比较。"""
        self.candidates = [(-4, 2, 10, 8), (-7, 3, 10, 6), (-4, 1, 10, 7)]

    def test_python_mode_skips_native(self):
        """确认显式切回 Python 时不会调用原生排序。"""
        native_order = Mock()
        with adapter("python", native_order) as sort:
            self.assertEqual(sort(self.candidates), sorted(self.candidates))
        native_order.assert_not_called()

    def test_shadow_returns_python_order_when_native_disagrees(self):
        """让两种排序产生分歧，验证对照模式告警并保留 Python 顺序。"""
        native_order = Mock(return_value=[0, 1, 2])
        with adapter("shadow", native_order) as sort, self.assertLogs(level="WARNING"):
            result = sort(self.candidates)
        native_order.assert_called_once_with(self.candidates)
        self.assertEqual(result, sorted(self.candidates))

    def test_rust_uses_native_order_and_keeps_candidate_objects(self):
        """用不同于 Python 的原生顺序验证切换，并检查对象引用。"""
        indices = [1, 0, 2]
        for mode in (None, "rust"):
            with self.subTest(mode=mode):
                native_order = Mock(return_value=indices)
                with adapter(mode, native_order) as sort:
                    result = sort(self.candidates)
                native_order.assert_called_once_with(self.candidates)
                for item, index in zip(result, indices, strict=True):
                    self.assertIs(item, self.candidates[index])

    def test_rust_falls_back_for_unsupported_keys(self):
        """用超出 Rust 整数范围的键检查 Python 回退。"""
        candidates = [(-1, 9, 0, 3), (-(2**128), 0, 0, 2), (-1, 8, 0, 1)]
        native_order = Mock(return_value=None)
        with adapter("rust", native_order) as sort:
            self.assertEqual(sort(candidates), sorted(candidates))
        native_order.assert_called_once_with(candidates)


class NativeEvictionTests(unittest.TestCase):
    def test_native_order_matches_python_on_seeded_candidates(self):
        """用不同数量的候选项和重复键核对原生接口的排序结果。"""
        native = installed_native()
        if native is None:
            self.skipTest("需要安装 comfyui-rs 原生扩展")

        rng = random.Random(2711)
        for size in range(65):
            candidates = [
                (-rng.randrange(8), rng.randrange(6), rng.randrange(16), index)
                for index in range(size)
            ]
            if candidates:
                # 完全相同的排序键也应保持输入顺序。
                candidates.extend((candidates[-1], candidates[-1]))
            rng.shuffle(candidates)
            self.assertEqual(native.eviction_order(candidates), python_order(candidates))
        self.assertIsNone(native.eviction_order([(-(2**128), 0, 0, 0)]))


class FakeLoadedModel:
    def __init__(
        self, name, key, unloaded, *, device="cuda:0", dead=False, can_unload=True
    ):
        """让测试控制模型所在设备、排序键和卸载结果。"""
        self.name = name
        self.offloaded, refs, self.memory = key
        self.model = SimpleNamespace(model=object(), refs=refs)
        self.unloaded = unloaded
        self.device = device
        self.dead = dead
        self.can_unload = can_unload
        self.currently_used = True

    def is_dead(self):
        """模拟已失效的模型，验证回收前的候选过滤。"""
        return self.dead

    def model_offloaded_memory(self):
        """返回排序首项取负之前的已卸载字节数。"""
        return self.offloaded

    def model_memory(self):
        """提供模型总大小，区分前两项相同的候选项。"""
        return self.memory

    def model_unload(self, memory_to_free):
        """记录回收请求，并模拟部分释放后仍需保留模型。"""
        self.unloaded.append((self.name, memory_to_free))
        return self.can_unload


def run_free_memory(path):
    """执行源文件中的回收函数和排序绑定，隔离 Torch 初始化和设备调用。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "free_memory")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_sort_eviction_candidates"
                for target in node.targets
            )
        )
        or any(
            isinstance(child, ast.ImportFrom) and child.module == "comfyui_rs"
            for child in ast.walk(node)
        )
    ]

    unload_calls = []
    models = [
        FakeLoadedModel("a", (4, 4, 10), unload_calls),
        FakeLoadedModel("b", (8, 3, 10), unload_calls),
        FakeLoadedModel("c", (8, 2, 80), unload_calls),
        FakeLoadedModel("d", (8, 2, 40), unload_calls),
        FakeLoadedModel("e", (8, 2, 40), unload_calls, can_unload=False),
        FakeLoadedModel("other-device", (99, 1, 1), unload_calls, device="cuda:1"),
        FakeLoadedModel("keep", (99, 1, 1), unload_calls),
        FakeLoadedModel("dead", (99, 1, 1), unload_calls, dead=True),
    ]
    kept = models[6]
    scope = {
        "current_loaded_models": models,
        "DISABLE_SMART_MEMORY": True,
        "cleanup_models_gc": Mock(),
        "detail": Mock(),
        "logging": Mock(),
        "os": os,
        "soft_empty_cache": Mock(),
        "sys": SimpleNamespace(getrefcount=lambda model: model.refs),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    returned = scope["free_memory"](64, "cuda:0", keep_loaded=[kept])
    return (
        unload_calls,
        [model.name for model in returned],
        [(model.name, model.currently_used) for model in models],
        scope["soft_empty_cache"].call_count,
    )


class BundledComfyUITests(unittest.TestCase):
    def test_bundled_free_memory_preserves_unload_behavior(self):
        """验证各模式在实际回收路径中的卸载顺序和回收后状态。"""
        baseline = (
            [(name, 1e32) for name in ("d", "e", "c", "b", "a")],
            ["d", "c", "b", "a"],
            [("e", False), ("other-device", True), ("keep", True), ("dead", True)],
            1,
        )
        checkout = os.getenv("COMFYUI_RS_COMFYUI_PATH")
        if checkout:
            original = Path(checkout) / MODEL_MANAGEMENT
            self.assertTrue(original.is_file(), f"没有找到 {original}")
            self.assertEqual(run_free_memory(original), baseline)

        native = installed_native()
        order = python_order if native is None else native.eviction_order

        for mode in (None, "python", "shadow", "rust"):
            with self.subTest(mode=mode):
                native_order = Mock(side_effect=order)
                with adapter(mode, native_order):
                    self.assertEqual(run_free_memory(BUNDLED_MODEL_MANAGEMENT), baseline)
                if mode == "python":
                    native_order.assert_not_called()
                else:
                    native_order.assert_called_once()

        with (
            patch.dict(sys.modules, {"comfyui_rs": None}),
            patch.dict(os.environ, {MODE_VAR: "rust"}),
        ):
            self.assertEqual(run_free_memory(BUNDLED_MODEL_MANAGEMENT), baseline)


if __name__ == "__main__":
    unittest.main()
