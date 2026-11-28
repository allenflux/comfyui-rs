"""验证分配尺寸和张量缓冲区布局在迁移后保持原有语义。"""

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


COMFYUI = Path(__file__).resolve().parents[1] / "comfyui"
HAS_TENSOR_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "comfy_aimdo", "comfy_kitchen")
)


@unittest.skipUnless(HAS_TENSOR_RUNTIME, "请在项目构建环境中运行张量布局测试")
class MemoryManagementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """从内置 ComfyUI 加载实际内存管理入口。"""
        cls.torch = importlib.import_module("torch")
        with patch.object(sys, "path", [str(COMFYUI), *sys.path]):
            cls.memory = importlib.import_module("comfy.memory_management")

    def test_geometry_numel_preserves_shape_edge_cases(self):
        """检查标量、零维度、超出原生整数范围和一次性形状迭代器。"""
        torch = self.torch
        cases = (
            (torch.Size([]), 1),
            (torch.Size([2, 3, 4]), 24),
            (torch.Size([2, 0, 4]), 0),
            ((2**63, 2**63), 2**126),
            ((2**63, 2**63, 8), 2**129),
            ((2**65, 8), 2**68),
            ((2**65, 8, 0), 0),
            ((2, -3), -6),
            (iter([2, 3]), 6),
        )
        for shape, expected in cases:
            with self.subTest(shape=shape):
                self.assertEqual(self.memory.TensorGeometry(shape, torch.float32).numel(), expected)

    def test_alignment_rounds_each_storage_field_and_skips_missing_tensors(self):
        """逐项补齐 1024 字节边界，空张量和缺失项不占空间。"""
        torch = self.torch
        tensors = [
            torch.empty((), dtype=torch.float16),
            [
                None,
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty(1024, dtype=torch.uint8),
            ],
            torch.empty(1025, dtype=torch.uint8),
        ]
        self.assertEqual(self.memory.vram_aligned_size(tensors), 4096)

    def test_small_layouts_do_not_call_the_native_planner(self):
        """常见小批量布局和带多个空槽位的单张量布局均留在 Python。"""
        torch = self.torch
        weight = torch.empty((2, 3), dtype=torch.int16)
        geometry = self.memory.TensorGeometry(torch.Size([]), torch.float32)
        gathered = torch.empty(1028, dtype=torch.uint8)
        with patch.object(self.memory, "_rust_gathered_byte_layout") as native:
            self.assertEqual(geometry.numel(), 1)
            self.assertEqual(self.memory.vram_aligned_size(weight), 1024)
            self.assertEqual(self.memory.vram_aligned_size([weight, geometry]), 2048)
            for templates in ([weight], [weight, None], [weight] + [None] * 15):
                with self.subTest(slots=len(templates)):
                    views = self.memory.interpret_gathered_like(templates, gathered)
                    self.assertEqual(len(views), len(templates))
                    self.assertEqual(views[0].data_ptr(), gathered.data_ptr())
                    self.assertEqual(views[1:], templates[1:])

            first, missing, scalar = self.memory.interpret_gathered_like([weight, None, geometry], gathered)
            self.assertEqual(tuple(first.shape), (2, 3))
            self.assertIsNone(missing)
            self.assertEqual(tuple(scalar.shape), ())
            self.assertEqual(scalar.data_ptr(), gathered.data_ptr() + 1024)
            native.assert_not_called()

    def test_mixed_bulk_layout_calls_the_native_planner_once(self):
        """一次提交普通张量和几何元数据，保留视图顺序、空项和末尾边界。"""
        binding = self.memory._rust_gathered_byte_layout
        if binding is None:
            self.skipTest("需要启用包含批量布局计算的 comfyui-rs 原生扩展")
        torch = self.torch
        weight = torch.nn.Parameter(torch.empty((2, 3), dtype=torch.int16), requires_grad=False)
        geometry = self.memory.TensorGeometry(torch.Size([]), torch.float32)
        templates = [weight, None, geometry] * 8
        gathered = torch.empty(15 * 1024 + 4, dtype=torch.uint8)

        with patch.object(self.memory, "_rust_gathered_byte_layout", wraps=binding) as native:
            self.assertEqual(self.memory.vram_aligned_size(templates), 16 * 1024)
            native.assert_not_called()
            views = self.memory.interpret_gathered_like(templates, gathered)
            native.assert_called_once()
            (descriptors,) = native.call_args.args
            self.assertIn(12, descriptors)
            self.assertIn(((), 4), descriptors)

        self.assertEqual(len(views), len(templates))
        for index in range(8):
            first, missing, scalar = views[index * 3:index * 3 + 3]
            self.assertIsNone(missing)
            self.assertEqual(first.data_ptr(), gathered.data_ptr() + index * 2048)
            self.assertEqual(scalar.data_ptr(), gathered.data_ptr() + index * 2048 + 1024)
            self.assertEqual(tuple(scalar.shape), ())
            self.assertEqual(scalar.dtype, torch.float32)

    def test_fullgraph_compile_preserves_layout_calculations(self):
        """完整追踪时沿用可追踪运算，即使模板数量足以启用原生布局。"""
        torch = self.torch
        memory = self.memory

        def add_layout_metadata(tensor, gathered):
            """将形状、对齐大小和最后一项视图都用于被追踪的张量计算。"""
            geometry = memory.TensorGeometry(tensor.shape, tensor.dtype)
            last = memory.interpret_gathered_like([tensor] * 16, gathered)[-1]
            return tensor + geometry.numel() + memory.vram_aligned_size([tensor, None]) + last

        tensor = torch.arange(6, dtype=torch.int16).reshape(2, 3)
        gathered = torch.zeros(15 * 1024 + 12, dtype=torch.uint8)
        gathered[15 * 1024:].view(torch.int16).fill_(5)
        try:
            compiled = torch.compile(add_layout_metadata, backend="eager", fullgraph=True)
        except NotImplementedError as error:
            self.skipTest(str(error))
        torch.testing.assert_close(compiled(tensor, gathered), tensor + 1035, rtol=0, atol=0)

    def test_gathered_views_accept_last_slice_without_trailing_padding(self):
        """验证逐项对齐、缺失槽位和最后一项未补齐时的原地视图。"""
        torch = self.torch
        templates = [
            torch.empty((2, 3), dtype=torch.int16),
            None,
            torch.empty((1,), dtype=torch.float32),
        ]
        gathered = torch.zeros(1028, dtype=torch.uint8)
        first, missing, last = self.memory.interpret_gathered_like(templates, gathered)
        self.assertIsNone(missing)
        self.assertEqual(first.data_ptr(), gathered.data_ptr())
        self.assertEqual(last.data_ptr(), gathered.data_ptr() + 1024)
        self.assertEqual(tuple(first.shape), (2, 3))
        self.assertEqual(first.dtype, torch.int16)
        self.assertEqual(last.dtype, torch.float32)
        first.fill_(12)
        last.fill_(3.5)
        self.assertEqual(gathered[:12].view(torch.int16).tolist(), [12] * 6)
        self.assertEqual(gathered[1024:].view(torch.float32).tolist(), [3.5])
        self.assertEqual(torch.count_nonzero(gathered[12:1024]).item(), 0)

        with self.assertRaisesRegex(ValueError, r"needs 1028 bytes, but only has 1027"):
            self.memory.interpret_gathered_like(templates, gathered[:-1])
        with self.assertRaisesRegex(ValueError, r"Buffer must be 1D and single-byte"):
            self.memory.interpret_gathered_like(templates, gathered.view(1, -1))

    def test_empty_field_after_nonempty_field_still_checks_aligned_start(self):
        """即使后一项是空张量，也须保留对实际偏移位置的边界校验。"""
        torch = self.torch
        templates = [torch.empty(1, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8)]
        with self.assertRaisesRegex(ValueError, r"needs 1024 bytes, but only has 1"):
            self.memory.interpret_gathered_like(templates, torch.empty(1, dtype=torch.uint8))
        _, empty = self.memory.interpret_gathered_like(templates, torch.empty(1024, dtype=torch.uint8))
        self.assertEqual(empty.numel(), 0)

    def test_gathered_preserves_first_pytorch_view_error(self):
        """第一项的视图错误优先于后一项的缓冲区越界。"""
        torch = self.torch
        templates = [torch.empty((), dtype=torch.float32), *([torch.empty(1, dtype=torch.uint8)] * 15)]
        gathered = torch.zeros(5, dtype=torch.uint8)[1:]
        with self.assertRaises(RuntimeError):
            self.memory.interpret_gathered_like(templates, gathered)

        # 后续几何元数据不受支持时，回退也应先报告第一项的视图错误。
        templates[-1] = self.memory.TensorGeometry((1,), torch.bool)
        with self.assertRaises(RuntimeError):
            self.memory.interpret_gathered_like(templates, gathered)

    def test_quantized_cat_keeps_layout_and_storage_aliases(self):
        """量化数据和元数据各自对齐，原地合并后的布局能再次解析。"""
        torch = self.torch
        if not hasattr(torch.Tensor, "_make_wrapper_subclass"):
            self.skipTest("comfy-kitchen 量化包装需要 Tensor._make_wrapper_subclass")
        weights = [
            self.memory.QuantizedTensor.from_float(
                torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8),
                "TensorWiseINT8Layout",
                per_channel=True,
            )
            for rows in (2, 3)
        ]
        gathered = torch.empty(self.memory.vram_aligned_size(weights), dtype=torch.uint8)
        self.assertEqual(gathered.numel(), 4096)
        views = self.memory.interpret_gathered_like(weights, gathered)
        for weight, view in zip(weights, views, strict=True):
            fields, _ = weight.__tensor_flatten__()
            view.__tensor_flatten__()
            for field in fields:
                getattr(view, field).copy_(getattr(weight, field))

        combined = self.memory.cat_gathered(views, gathered)
        self.assertEqual(tuple(combined.shape), (5, 8))
        self.assertEqual(combined._qdata.data_ptr(), gathered.data_ptr())
        torch.testing.assert_close(
            combined.dequantize(),
            torch.cat([weight.dequantize() for weight in weights]),
            rtol=0,
            atol=0,
        )
        readback = self.memory.interpret_gathered_like([combined], gathered)[0]
        fields, _ = combined.__tensor_flatten__()
        readback.__tensor_flatten__()
        for field in fields:
            self.assertEqual(getattr(combined, field).data_ptr(), getattr(readback, field).data_ptr())


if __name__ == "__main__":
    unittest.main()
