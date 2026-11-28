import asyncio
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch


COMFYUI = Path(__file__).resolve().parents[1] / "comfyui"
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "comfy_aimdo", "comfy_kitchen"))


async def _check_prompt_executor(base_directory):
    sys.path.insert(0, str(COMFYUI))
    sys.argv[:] = [str(COMFYUI / "main.py"), "--cpu", "--disable-assets", "--base-directory", base_directory,
                   "--memory-strategy", "aggressive"]
    from comfy import options

    options.enable_args_parsing()
    import execution
    import nodes
    import torch
    from comfy.cli_args import args
    from comfy.model_management import load_model_gpu
    from comfy.model_patcher import ModelPatcher

    cpu = torch.device("cpu")
    patcher = ModelPatcher(torch.nn.Linear(2, 2), cpu, cpu)
    ready = asyncio.Event()

    class TestOutput:
        OUTPUT_NODE = True
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    class AsyncOutput(TestOutput):
        async def run(self):
            await ready.wait()
            load_model_gpu(patcher)
            return (1,)

    class SyncOutput(TestOutput):
        def run(self):
            ready.set()
            return (2,)

    nodes.NODE_CLASS_MAPPINGS.update({"AsyncOutput": AsyncOutput, "SyncOutput": SyncOutput})
    prompt = {
        "async": {"class_type": "AsyncOutput", "inputs": {}},
        "sync": {"class_type": "SyncOutput", "inputs": {}},
    }
    planner = Mock()
    complete_calls = []

    def record_completion(node_id, pending, used_models, *, allow_offload, cached):
        complete_calls.append((node_id, tuple(pending), used_models, allow_offload, cached))

    planner.after_node.side_effect = record_completion

    def executor():
        server = Mock(client_id=None, last_node_id=None)
        return execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})

    async def run(worker, prompt_id):
        ready.clear()
        await worker.execute_async(prompt, prompt_id, execute_outputs=list(prompt))
        assert worker.success, worker.status_messages

    with patch.object(execution, "WorkflowMemoryPlanner", return_value=planner) as create_planner:
        await run(executor(), "cpu")
        create_planner.assert_not_called()

        with patch.object(execution.comfy.model_management, "model_offload_enabled", return_value=True):
            worker = executor()
            await run(worker, "async")
            assert complete_calls == [("sync", ("async",), set(), False, False), ("async", (), {patcher}, True, False)], complete_calls

            create_planner.reset_mock()
            planner.reset_mock()
            complete_calls.clear()
            await run(worker, "cache")
            create_planner.assert_called_once()
            assert create_planner.call_args.args[3] == {"async": [[1]], "sync": [[2]]}
            planner.observe_outputs.assert_has_calls([call("async", [[1]]), call("sync", [[2]])])
            assert complete_calls == [("async", ("sync",), set(), True, True), ("sync", (), set(), True, True)], complete_calls

            create_planner.reset_mock()
            args.memory_strategy = "default"
            await run(executor(), "default")
            create_planner.assert_not_called()


@unittest.skipUnless(HAS_RUNTIME, "请在项目构建环境中运行工作流调度测试")
class WorkflowMemoryExecutionTests(unittest.TestCase):
    def test_prompt_executor_preserves_async_loads_and_cached_outputs(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--check-executor", directory],
                capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--check-executor"]:
        asyncio.run(_check_prompt_executor(sys.argv[2]))
    else:
        unittest.main()
