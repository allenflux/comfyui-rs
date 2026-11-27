"""覆盖模型计时探针的同任务归因和无同步 CUDA 事件处理。"""

import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import model_phase_probe


class ProbeFixture:
    """提供与 Comfy 入口同名的轻量替身，CUDA API 无需实体设备。"""

    def __init__(self, cuda):
        self.calls = []
        self.device = SimpleNamespace(type="cuda" if cuda else "cpu")

        class Executor:
            async def execute_async(self, prompt, prompt_id):
                return prompt()

        class Module:
            def to(self, device):
                return self

        class Patcher:
            pass

        self.executor = Executor()
        self.patcher = Patcher
        self.mm = SimpleNamespace(
            LoadedModel=type("LoadedModel", (), {}),
            get_torch_device=lambda: self.device,
            load_models_gpu=lambda models: len(models),
            cast_to_gathered=self.gather,
        )
        self.utils = SimpleNamespace(load_torch_file=lambda filename: filename)
        self.torch = SimpleNamespace(
            nn=SimpleNamespace(Module=Module),
            compiler=SimpleNamespace(is_compiling=lambda: False),
            version=SimpleNamespace(cuda="13.0" if cuda else None),
        )
        self.modules = {
            "torch": self.torch,
            "execution": SimpleNamespace(PromptExecutor=Executor),
            "comfy": SimpleNamespace(model_management=self.mm, model_patcher=SimpleNamespace(ModelPatcher=Patcher),
                                     sd=SimpleNamespace(), utils=self.utils),
        }

    def gather(self, tensors, r, non_blocking=False, stream=None, r2=None):
        """记录底层调用，确认探针既不改参数也不重复执行。"""
        self.calls.append((tensors, r, non_blocking, stream, r2))

    def run(self, report, jobs):
        """通过公共异步入口执行任务并读取报告。"""
        with patch.dict(sys.modules, self.modules):
            model_phase_probe._install(report)
            results = [asyncio.run(self.executor.execute_async(prompt=job, prompt_id=str(i)))
                       for i, job in enumerate(jobs)]
        rows = [json.loads(line) for line in report.read_text().splitlines()]
        return rows, results


class ModelPhaseProbeTests(unittest.TestCase):
    def test_cpu_correlates_checkpoint_and_switch_without_cuda(self):
        """CPU 不访问 CUDA，记录检查点与真正变化的模型请求集合。"""
        fixture = ProbeFixture(cuda=False)
        a, b = fixture.patcher(), fixture.patcher()

        def job(model):
            return lambda: (fixture.utils.load_torch_file("fixture"), fixture.mm.load_models_gpu([model]))

        with TemporaryDirectory() as directory:
            rows, results = fixture.run(Path(directory) / "probe.jsonl", [job(a), job(a), job(b)])
        marker, *points = rows
        self.assertTrue(marker["probe_ready"])
        self.assertEqual(results, [("fixture", 1)] * 3)
        self.assertEqual([point["prompt_id"] for point in points], ["0", "1", "2"])
        self.assertEqual([point["phases"]["checkpoint_read"]["calls"] for point in points], [1, 1, 1])
        self.assertEqual([point["phases"]["model_switch"]["calls"] for point in points], [0, 0, 1])
        self.assertTrue(all(point["weight_move_cuda"]["stream_ms"] is None for point in points))

    def test_cuda_only_reads_completed_pairs_on_the_copy_stream(self):
        """阶段报告保留未完成和超限计数，记录及读取都不需要同步。"""
        done = []
        fixture = ProbeFixture(cuda=True)

        class Stream:
            device = fixture.device

        class Event:
            def __init__(self, enable_timing):
                assert enable_timing

            def record(self, stream):
                self.index = len(done)
                done.append(stream)

            def query(self):
                return self.index != 3

            def elapsed_time(self, end):
                assert self.index + 1 == end.index
                return 2.0

        stream = Stream()
        destination = SimpleNamespace(device=stream.device)
        fixture.torch.cuda = SimpleNamespace(Stream=Stream, Event=Event,
                                              stream=lambda _: nullcontext(),
                                              is_current_stream_capturing=lambda: False)

        def copies():
            for _ in range(4):
                fixture.mm.cast_to_gathered([], destination, non_blocking=True, stream=stream)

        with TemporaryDirectory() as directory, patch.object(model_phase_probe, "_CUDA_EVENT_LIMIT", 3):
            (_, point), _ = fixture.run(Path(directory) / "probe.jsonl", [copies])
        self.assertEqual(done, [stream] * 6)
        self.assertEqual(fixture.calls, [([], destination, True, stream, None)] * 4)
        self.assertEqual(point["phases"]["weight_move_host"]["calls"], 4)
        self.assertEqual(point["weight_move_cuda"], {
            "events_complete": 2, "events_incomplete": 1, "events_dropped": 1,
            "events_failed": 0, "stream_ms": 4.0,
        })


if __name__ == "__main__":
    unittest.main()
