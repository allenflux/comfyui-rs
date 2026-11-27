import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


OBSERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "comfyui/custom_nodes/comfyui_rs_observer/__init__.py"
)
LOG_PREFIX = "comfyui_rs_observation "


class ObserverTests(unittest.TestCase):
    def setUp(self):
        """用测试替身提供执行器和 CUDA 计数器，避免依赖 Torch 或 GPU。"""
        class PromptExecutor:
            def __init__(self):
                """为每项测试保留独立的状态消息和调用记录。"""
                self.success = True
                self.status_messages = []
                self.run = Mock()

            def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
                """保留上游调用签名，便于核对观察器是否原样传参。"""
                return self.run(prompt, prompt_id, extra_data, execute_outputs)

        class OutOfMemoryError(RuntimeError):
            pass

        execution = ModuleType("execution")
        execution.PromptExecutor = PromptExecutor
        self.executor_class = PromptExecutor
        self.executor = PromptExecutor()
        self.original_execute = PromptExecutor.execute
        self.device = SimpleNamespace(type="cuda", index=0)
        self.cuda = SimpleNamespace(
            OutOfMemoryError=OutOfMemoryError,
            mem_get_info=Mock(return_value=(800, 1600)),
            memory_reserved=Mock(return_value=300),
            reset_peak_memory_stats=Mock(),
            max_memory_reserved=Mock(return_value=550),
            max_memory_allocated=Mock(return_value=440),
        )
        torch = ModuleType("torch")
        torch.version = SimpleNamespace(cuda="12.8")
        torch.cuda = self.cuda
        torch.OutOfMemoryError = OutOfMemoryError
        comfy = ModuleType("comfy")
        comfy.model_management = ModuleType("comfy.model_management")
        comfy.model_management.get_torch_device = Mock(return_value=self.device)
        comfy.model_management.InterruptProcessingException = type(
            "InterruptProcessingException", (Exception,), {}
        )
        self.native = ModuleType("comfyui_rs")
        self.native.assess_budget = Mock(return_value=(False, 700, 50))
        module_patch = patch.dict(
            sys.modules,
            {
                "execution": execution,
                "comfy": comfy,
                "comfy.model_management": comfy.model_management,
                "torch": torch,
                "comfyui_rs": self.native,
            },
        )
        env_patch = patch.dict(os.environ)
        for patcher in (module_patch, env_patch):
            patcher.start()
            self.addCleanup(patcher.stop)
        for key in (
            "COMFYUI_RS_OBSERVE",
            "COMFYUI_RS_EXPECTED_EXTRA_BYTES",
            "COMFYUI_RS_HEADROOM_BYTES",
        ):
            os.environ.pop(key, None)

    def load_observer(self, *, enabled=False):
        """从插件文件导入观察器，触发实际的安装逻辑。"""
        if enabled:
            os.environ["COMFYUI_RS_OBSERVE"] = "1"
        spec = importlib.util.spec_from_file_location("comfyui_rs_observer", OBSERVER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def observation(self, logs):
        """只提取结构化观测记录，每次执行应恰好产生一条。"""
        messages = [
            record.getMessage().removeprefix(LOG_PREFIX)
            for record in logs.records
            if record.getMessage().startswith(LOG_PREFIX)
        ]
        self.assertEqual(len(messages), 1)
        return json.loads(messages[0])

    def test_disabled_leaves_executor_unchanged(self):
        """关闭观察器时，即使缺少运行依赖也能直接加载插件。"""
        with patch.dict(sys.modules, {"torch": None, "execution": None}):
            self.load_observer()
        self.assertIs(self.executor_class.execute, self.original_execute)

    def test_advisory_budget_does_not_block_execution(self):
        """显存预算不足时仍执行任务，重复加载也只记录一次。"""
        os.environ["COMFYUI_RS_EXPECTED_EXTRA_BYTES"] = "750"
        os.environ["COMFYUI_RS_HEADROOM_BYTES"] = "100"
        self.load_observer(enabled=True)
        wrapped = self.executor_class.execute
        self.load_observer(enabled=True)
        self.assertIs(self.executor_class.execute, wrapped)

        prompt = {"42": {"class_type": "Example", "inputs": {"secret": "do not log"}}}
        outputs = ["42"]
        expected_result = object()
        self.executor.run.return_value = expected_result
        with self.assertLogs(level="INFO") as logs:
            result = self.executor.execute(
                prompt=prompt, prompt_id="p1", execute_outputs=outputs
            )

        self.assertIs(result, expected_result)
        self.executor.run.assert_called_once_with(prompt, "p1", {}, outputs)
        self.native.assess_budget.assert_called_once_with(800, 750, 100)
        self.cuda.reset_peak_memory_stats.assert_called_once_with(self.device)
        record = self.observation(logs)
        self.assertEqual(record["prompt_id"], "p1")
        self.assertEqual(record["status"], "success")
        self.assertGreaterEqual(record["duration_ms"], 0)
        self.assertNotIn("do not log", "\n".join(logs.output))
        for field, value in {
            "device_index": 0,
            "driver_free_bytes": 800,
            "torch_reserved_start_bytes": 300,
            "torch_peak_reserved_bytes": 550,
            "torch_peak_allocated_bytes": 440,
            "torch_reserved_growth_bytes": 250,
        }.items():
            self.assertEqual(record["cuda"][field], value, field)
        self.assertEqual(
            record["cuda"]["budget"],
            {
                "fits": False,
                "available_bytes": 700,
                "shortfall_bytes": 50,
                "expected_extra_bytes": 750,
                "headroom_bytes": 100,
            },
        )

    def test_handled_errors_and_interruptions_use_current_prompt_status(self):
        """终态来自当前任务的状态消息，私人诊断信息不进入观测日志。"""
        self.load_observer(enabled=True)
        for event, exception_type, expected_status in (
            ("execution_error", "ValueError", "error"),
            ("execution_error", "torch.OutOfMemoryError", "oom"),
            ("execution_interrupted", None, "interrupted"),
        ):
            with self.subTest(status=expected_status):
                def run_prompt(*args):
                    """模拟执行期间更新状态，同时混入另一任务的终态。"""
                    self.executor.status_messages = [
                        (
                            event,
                            {
                                "prompt_id": "p1",
                                "exception_type": exception_type,
                                "exception_message": "private diagnostic",
                            },
                        ),
                        ("execution_success", {"prompt_id": "other"}),
                    ]

                self.executor.run.side_effect = run_prompt
                with self.assertLogs(level="INFO") as logs:
                    self.executor.execute({}, "p1")
                self.assertEqual(self.observation(logs)["status"], expected_status)
                self.assertNotIn("private diagnostic", "\n".join(logs.output))

    def test_telemetry_failure_preserves_executor_result_and_exception(self):
        """CUDA 采样失败时，返回值和原始 OOM 异常仍透传给调用方。"""
        self.cuda.mem_get_info.side_effect = RuntimeError("telemetry unavailable")
        self.load_observer(enabled=True)
        expected_result = object()
        self.executor.run.return_value = expected_result

        with self.assertLogs(level="INFO") as logs:
            self.assertIs(self.executor.execute({}, "p1"), expected_result)
        self.assertEqual(self.observation(logs)["status"], "success")

        error = self.cuda.OutOfMemoryError("workflow failed")
        self.executor.run.side_effect = error
        with (
            self.assertLogs(level="INFO") as logs,
            self.assertRaises(type(error)) as caught,
        ):
            self.executor.execute({}, "p2")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.observation(logs)["status"], "oom")

    def test_broken_log_handler_preserves_executor_exception(self):
        """日志写入也出错时，调用方仍收到原始的工作流异常。"""
        observer = self.load_observer(enabled=True)
        handler_error = RuntimeError("log handler failed")
        workflow_error = ValueError("workflow failed")
        self.executor.run.side_effect = workflow_error
        with (
            patch.object(observer._logger, "handle", side_effect=handler_error) as handle,
            patch.object(observer._logger, "isEnabledFor", return_value=True),
            self.assertRaises(ValueError) as caught,
        ):
            self.executor.execute({}, "p1")
        handle.assert_called()
        self.assertIs(caught.exception, workflow_error)


if __name__ == "__main__":
    unittest.main()
