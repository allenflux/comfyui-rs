import asyncio
from contextlib import ExitStack, nullcontext
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, call, patch


COMFYUI = Path(__file__).resolve().parents[1] / "comfyui"
HAS_TENSOR_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "comfy_aimdo", "comfy_kitchen")
)


@unittest.skipUnless(HAS_TENSOR_RUNTIME, "请在安装了 ComfyUI 张量运行时的环境中运行")
class WorkflowModelOffloadTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        with patch.object(sys, "path", [str(COMFYUI), *sys.path]):
            from comfy.cli_args import args

            with patch.object(args, "cpu", True):
                from comfy import model_management
                from comfy.model_patcher import ModelPatcher

        cls.torch = torch
        cls.manager = model_management
        cls.patcher_class = ModelPatcher
        cls.cpu = torch.device("cpu")
        cls.gpu = torch.device("cuda:1")

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.multiple(
            self.manager, current_loaded_models=[], vram_state=self.manager.VRAMState.NORMAL_VRAM,
            DISABLE_SMART_MEMORY=False, in_training=False,
        ))
        self.device_context = stack.enter_context(patch.object(
            self.manager, "cuda_device_context", side_effect=lambda device: nullcontext(),
        ))
        self.clear_cache = stack.enter_context(patch.object(self.manager, "soft_empty_cache"))

    def make_patcher(self):
        return self.patcher_class(self.torch.nn.Linear(1, 1), self.cpu, self.cpu)

    def register_model(self, device):
        patcher = self.make_patcher()
        entry = self.manager.LoadedModel(patcher)
        entry.model_load()
        entry.device = device
        self.manager.current_loaded_models.append(entry)
        return patcher, entry

    def test_offload_detaches_only_targeted_resident_model(self):
        target, _ = self.register_model(self.gpu)
        kept, _ = self.register_model(self.gpu)
        empty, _ = self.register_model(self.gpu)
        pinned, _ = self.register_model(self.gpu)
        host, _ = self.register_model(self.cpu)
        empty.model.model_loaded_weight_memory = 0
        pinned.offload_device = self.gpu

        self.assertEqual(self.manager.offloadable_models(), [target, kept])
        self.assertEqual(self.manager.offload_models([target.clone()]), 0)
        self.assertEqual(self.manager.offload_models([target, empty, pinned, host]), 1)
        self.assertEqual(self.manager.loaded_models(), [kept, empty, pinned, host])
        self.assertEqual(target.loaded_size(), 0)
        self.assertGreater(kept.loaded_size(), 0)
        self.device_context.assert_called_once_with(self.gpu)
        self.clear_cache.assert_called_once()

    def test_offload_synchronizes_dynamic_models_once_per_device(self):
        first, first_entry = self.register_model(self.gpu)
        second, second_entry = self.register_model(self.gpu)
        first.model.model_loaded_weight_memory //= 2

        events = Mock()
        with (
            patch.object(first, "is_dynamic", return_value=True),
            patch.object(second, "is_dynamic", return_value=True),
            patch.object(self.manager, "synchronize") as synchronize,
            patch.object(first_entry, "model_unload", wraps=first_entry.model_unload) as unload_first,
            patch.object(second_entry, "model_unload", wraps=second_entry.model_unload) as unload_second,
        ):
            events.attach_mock(synchronize, "synchronize")
            events.attach_mock(unload_first, "unload_first")
            events.attach_mock(unload_second, "unload_second")
            self.assertEqual(self.manager.offload_models([first, second]), 2)

        self.assertEqual(events.mock_calls, [call.synchronize(), call.unload_first(), call.unload_second()])
        self.clear_cache.assert_called_once()

    def test_offload_removes_entry_by_identity_after_detach(self):
        target, entry = self.register_model(self.gpu)
        original_detach = target.detach
        replacement = self.manager.LoadedModel(target)

        def rebuild_registration(unpatch_all=True):
            original_detach(unpatch_all)
            replacement.model_load()
            replacement.device = self.gpu
            self.manager.current_loaded_models[:] = [replacement]

        with patch.object(target, "detach", side_effect=rebuild_registration):
            self.assertEqual(self.manager.offload_models([target]), 1)

        self.assertEqual(entry, replacement)
        self.assertIs(self.manager.current_loaded_models[0], replacement)

    def test_offload_obeys_explicit_memory_modes(self):
        state = self.manager.VRAMState
        target, _ = self.register_model(self.gpu)
        for overrides in (
            {"vram_state": state.DISABLED}, {"vram_state": state.SHARED}, {"vram_state": state.HIGH_VRAM},
            {"DISABLE_SMART_MEMORY": True}, {"in_training": True},
        ):
            with self.subTest(overrides=overrides), patch.multiple(self.manager, **overrides):
                self.assertFalse(self.manager.model_offload_enabled())
                self.assertEqual(self.manager.offloadable_models(), [])
                self.assertEqual(self.manager.offload_models([target]), 0)

        self.assertEqual(self.manager.loaded_models(), [target])
        self.device_context.assert_not_called()
        self.clear_cache.assert_not_called()

    async def test_capture_tracks_expanded_requests_in_pending_async_tasks(self):
        primary, auxiliary, outside = [self.make_patcher() for _ in range(3)]
        captured = set()
        ready = asyncio.Event()

        async def load_when_ready():
            await ready.wait()
            self.manager.load_models_gpu([primary])

        with patch.object(primary, "model_patches_models", return_value=[auxiliary]):
            with self.manager.capture_model_loads(captured):
                task = asyncio.create_task(load_when_ready())
            self.manager.load_models_gpu([outside])
            ready.set()
            await task

        self.assertEqual(captured, {primary, auxiliary})


if __name__ == "__main__":
    unittest.main()
