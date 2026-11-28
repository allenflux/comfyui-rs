"""验证监控接在真实节点调用处，保留映射器的同步与异步行为。"""

import ast
import asyncio
from contextlib import contextmanager
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


COMFY = Path(__file__).resolve().parents[1] / "comfyui"


def load_module(name):
    """独立加载执行上下文和监控模块，不启动 ComfyUI 服务。"""
    path = COMFY / "comfy_execution" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"node_monitor_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_monitor():
    """加载真实监控逻辑，用轻量替身隔离设备与容器 API。"""
    comfy = ModuleType("comfy")
    comfy.model_management = SimpleNamespace()
    comfy.system_memory = SimpleNamespace(virtual_memory_available=lambda: 1000, cgroup_memory_limit=lambda: None)
    torch = ModuleType("torch")
    torch.version = SimpleNamespace(cuda=None)
    torch.cuda = SimpleNamespace()
    with patch.dict(sys.modules, {"comfy": comfy, "torch": torch}):
        return load_module("node_monitor")


class ExecutionBlocker:
    pass


class SchemaNode:
    pass


class NodeMonitorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """加载生产映射函数，只替换 CUDA 采样和第三方依赖。"""
        self.context = load_module("utils")
        monitor_module = load_monitor()
        self.spans = []
        self.finished = []
        self.interrupt = Mock()

        @contextmanager
        def record_call(source, prompt_id, node_id, list_index, function):
            """记录探针起止位置，验证两端都能读取当前节点信息。"""
            self.assertIsNotNone(source)
            context = (prompt_id, node_id, list_index)
            self.assertEqual(self.context.get_executing_context(), context)
            self.spans.append((context, function))
            try:
                yield
            except BaseException as error:
                self.finished.append((context, function, error))
                raise
            else:
                self.finished.append((context, function, None))
            finally:
                self.assertEqual(self.context.get_executing_context(), context)

        scope = {
            "asyncio": asyncio,
            "inspect": inspect,
            "CurrentNodeContext": self.context.CurrentNodeContext,
            "ExecutionBlocker": ExecutionBlocker,
            "nodes": SimpleNamespace(before_node_execution=self.interrupt),
            "_ComfyNodeInternal": SchemaNode,
            "is_class": inspect.isclass,
            "make_locked_method_func": lambda cls, name, clone: getattr(clone, name),
            "_io": SimpleNamespace(build_nested_inputs=lambda inputs, *_args, **_kwargs: inputs),
            "custom_node_source": monitor_module.custom_node_source,
            "monitor_node_call": record_call,
        }
        path = COMFY / "execution.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        selected = {"_async_map_node_over_list", "resolve_map_node_over_list_results"}
        functions = [node for node in parsed.body if isinstance(node, ast.AsyncFunctionDef) and node.name in selected]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
        self.map_node = scope["_async_map_node_over_list"]
        self.resolve = scope["resolve_map_node_over_list_results"]

    async def test_sync_custom_node_preserves_mapped_values_and_context(self):
        """按原列表规则重复短输入，同步调用在对应节点上下文内执行。"""
        test = self

        class CustomNode:
            RELATIVE_PYTHON_MODULE = "custom_nodes.example"

            def run(self, value, factor):
                index = value - 1
                test.assertEqual(test.context.get_executing_context(), ("prompt", "node", index))
                test.assertEqual(test.spans[-1], (("prompt", "node", index), "run"))
                return (value * factor,)

        callback = Mock()
        values = await self.map_node("prompt", "node", CustomNode(), {"value": [1, 2, 3], "factor": [6]},
                                     "run", allow_interrupt=True, pre_execute_cb=callback)
        self.assertEqual(values, [(6,), (12,), (18,)])
        self.assertEqual(self.interrupt.call_count, 3)
        self.assertEqual([call.args for call in callback.call_args_list], [(0,), (1,), (2,)])
        self.assertEqual(self.spans, [(("prompt", "node", index), "run") for index in range(3)])
        self.assertEqual(self.finished, [(context, name, None) for context, name in self.spans])
        self.assertIsNone(self.context.get_executing_context())

    async def test_async_custom_node_stays_monitored_until_coroutine_returns(self):
        """带挂起点的节点仍返回原有 Task，恢复执行后才结束计时。"""
        test = self
        resume = asyncio.Event()
        result = object()

        class AsyncNode:
            RELATIVE_PYTHON_MODULE = "custom_nodes.example"

            async def run(self):
                test.assertEqual(test.context.get_executing_context(), ("prompt", "async", None))
                await resume.wait()
                test.assertEqual(test.context.get_executing_context(), ("prompt", "async", None))
                return result

        values = await self.map_node("prompt", "async", AsyncNode(), {}, "run")
        self.assertEqual(len(values), 1)
        self.assertIsInstance(values[0], asyncio.Task)
        self.assertFalse(values[0].done())
        self.assertEqual(self.spans, [(("prompt", "async", None), "run")])
        self.assertEqual(self.finished, [])
        self.assertIsNone(self.context.get_executing_context())
        resume.set()
        self.assertEqual(await self.resolve(values), [result])
        self.assertEqual(self.finished, [(("prompt", "async", None), "run", None)])

    async def test_builtin_nodes_and_blocked_calls_have_no_measurements(self):
        """内置节点照常运行，阻塞结果不会执行第三方节点或生成监控记录。"""
        class BuiltinNode:
            RELATIVE_PYTHON_MODULE = "comfy_extras.example"
            run = Mock(return_value=(True,))

        class BlockedNode:
            RELATIVE_PYTHON_MODULE = "custom_nodes.example"
            run = Mock(side_effect=AssertionError("blocked node was called"))

        blocker, replacement = ExecutionBlocker(), object()
        blocked_callback = Mock(return_value=replacement)
        self.assertEqual(await self.map_node("prompt", "builtin", BuiltinNode(), {}, "run"), [(True,)])
        self.assertEqual(await self.map_node("prompt", "blocked", BlockedNode(), {"value": [blocker]}, "run",
                                             execution_block_cb=blocked_callback), [replacement])
        BuiltinNode.run.assert_called_once_with()
        BlockedNode.run.assert_not_called()
        blocked_callback.assert_called_once_with(blocker)
        self.assertEqual(self.spans, [])

    async def test_schema_preparation_is_outside_and_callable_name_is_preserved(self):
        """V3 准备阶段未开始计时，指纹与验证使用各自的函数名。"""
        test = self

        class SchemaExtension(SchemaNode):
            RELATIVE_PYTHON_MODULE = "custom_nodes.example"

            @classmethod
            def VALIDATE_CLASS(cls):
                test.assertIsNone(test.context.get_executing_context())
                test.assertEqual(len(test.spans), len(test.finished))

            @classmethod
            def PREPARE_CLASS_CLONE(cls, data):
                test.assertIsNone(test.context.get_executing_context())
                return cls()

            async def fingerprint_inputs(self):
                return "fingerprint"

            def validate_inputs(self):
                return True

        for function, value in (("fingerprint_inputs", "fingerprint"), ("validate_inputs", True)):
            self.assertEqual(await self.map_node("prompt", "schema", SchemaExtension, {}, function), [value])
        self.assertEqual([name for _, name in self.spans], ["fingerprint_inputs", "validate_inputs"])

    async def test_sync_error_and_async_cancel_keep_original_semantics(self):
        """同步异常原样冒泡，已挂起的节点取消时作用域正常退出。"""
        expected = RuntimeError("failed node")
        wait_forever = asyncio.Event()

        class FailingNode:
            RELATIVE_PYTHON_MODULE = "custom_nodes.example"

            def execute(self):
                raise expected

            async def wait(self):
                await wait_forever.wait()

        with self.assertRaises(RuntimeError) as raised:
            await self.map_node("prompt", "failing", FailingNode(), {}, "execute")
        self.assertIs(raised.exception, expected)
        self.assertIs(self.finished[0][2], expected)
        (task,) = await self.map_node("prompt", "cancelled", FailingNode(), {}, "wait")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsInstance(self.finished[1][2], asyncio.CancelledError)
        self.assertIsNone(self.context.get_executing_context())


class NodeMonitorMetricsTests(unittest.TestCase):
    def setUp(self):
        """所有写盘测试只使用独立的临时日志目录。"""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {"COMFYUI_RS_NODE_MONITOR_LOG_DIR": str(self.root / "override")})
        environment.start()
        self.addCleanup(environment.stop)
        self.monitor = load_monitor()
        self.addCleanup(self.close_log_file)

    def close_log_file(self):
        """清理文件描述符后再删除临时目录。"""
        handler = self.monitor._file_handler
        if handler is not None:
            handler.close()

    def test_cpu_snapshot_survives_unavailable_metrics(self):
        """CPU 快照允许单个指标缺失，且不会调用任何 CUDA 方法。"""
        monitor = self.monitor
        with patch.object(monitor, "_rss_bytes", side_effect=OSError("metric unavailable")):
            self.assertEqual(monitor._snapshot(), {
                "rss_bytes": None, "ram_available_bytes": 1000, "cgroup_limit_bytes": None, "cuda": None,
            })

    def test_oom_logs_only_metadata_and_keeps_sampled_peak(self):
        """开始、心跳和结束共用调用 ID，结束记录保留采样峰值。"""
        monitor = self.monitor
        private_input = "private tensor path"
        oom = RuntimeError(f"CUDA out of memory: {private_input}")

        def metrics(rss, available, allocated, reserved):
            return {"rss_bytes": rss, "ram_available_bytes": available, "cgroup_limit_bytes": 1024,
                    "cuda": {"backend": "cuda", "allocator_backend": "native", "devices": {
                        "0": {"allocated_bytes": allocated, "reserved_bytes": reserved},
                    }}}

        start, sample, end = metrics(100, 700, 10, 30), metrics(800, 40, 40, 70), metrics(500, 100, 25, 55)
        clock = SimpleNamespace(perf_counter=Mock(side_effect=(10.0, 11.0, 14.0)),
                                monotonic=time.monotonic, time=time.time, time_ns=time.time_ns)
        with (patch.object(monitor, "_snapshot", side_effect=(start, end)),
              patch.object(monitor, "_register") as register,
              patch.object(monitor, "time", clock),
              self.assertLogs("comfyui_rs.node_monitor", level="INFO") as logs):
            with self.assertRaises(RuntimeError) as raised:
                with monitor.monitor_node_call(("ThirdParty", "example"), "prompt", "node", 0, "run"):
                    call = register.call_args.args[0]
                    call.emit(call.record("sample", sample, 13.0))
                    raise oom

        self.assertIs(raised.exception, oom)
        self.assertNotIn(private_input, "\n".join(logs.output))
        rows = [json.loads(entry.getMessage().removeprefix("comfyui_rs_node ")) for entry in logs.records]
        self.assertEqual([entry["event"] for entry in rows], ["start", "sample", "finish"])
        self.assertEqual(len({entry["call_id"] for entry in rows}), 1)
        finish = rows[-1]
        self.assertEqual((finish["status"], finish["oom_kind"], finish["samples"]), ("oom", "cuda_runtime", 3))
        self.assertEqual(finish["memory"]["start"], start)
        self.assertEqual(finish["memory"]["current"], end)
        self.assertEqual(finish["memory"]["peak"], {
            "rss_bytes": 800, "min_ram_available_bytes": 40, "cuda": sample["cuda"],
        })

    def test_log_handler_failure_preserves_node_exception(self):
        """监控日志不可写时仍将节点的原始异常交给执行器。"""
        monitor = self.monitor
        expected = ValueError("original error")
        with (patch.object(monitor, "_register"),
              patch.object(monitor, "_LOGGER", SimpleNamespace(info=Mock(side_effect=OSError("log unavailable"))))):
            with self.assertRaises(ValueError) as raised:
                with monitor.monitor_node_call(("ThirdParty", "example"), "prompt", "node", None, "run"):
                    raise expected
        self.assertIs(raised.exception, expected)

    def test_jsonl_default_directory_and_rotation_keep_only_configured_backups(self):
        """文件内容是纯 JSONL，默认路径生效，并限制备份数量和单文件大小。"""
        monitor = self.monitor
        folder_paths = SimpleNamespace(get_user_directory=lambda: str(self.root))
        with (patch.dict(os.environ), patch.dict(sys.modules, {"folder_paths": folder_paths}),
              patch.object(monitor, "_FILE_MAX_BYTES", 2048), patch.object(monitor, "_FILE_BACKUPS", 2),
              patch.object(monitor, "_register")):
            os.environ.pop("COMFYUI_RS_NODE_MONITOR_LOG_DIR", None)
            for index in range(10):
                with monitor.monitor_node_call(("ThirdParty", "example"), "prompt", str(index), index, "run"):
                    pass

        (current,) = (self.root / "node-monitor").glob("*.jsonl")
        backups = sorted(current.parent.glob(f"{current.name}.*"))
        self.assertEqual([file.suffix for file in backups], [".1", ".2"])
        records = []
        for path in [*reversed(backups), current]:
            self.assertLessEqual(path.stat().st_size, 2048)
            records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        self.assertEqual({row["schema"] for row in records}, {"comfyui-rs.node-memory.v1"})
        self.assertEqual(len({row["worker_run_id"] for row in records}), 1)
        self.assertTrue(all(row["timestamp_ms"] > 0 for row in records))
        self.assertEqual((records[-1]["node_id"], records[-1]["event"]), ("9", "finish"))

    def test_invalid_log_directory_keeps_console_data_and_result(self):
        """覆盖目录不可创建时，节点仍返回原对象并有控制台记录。"""
        monitor = self.monitor
        invalid = self.root / "plain-file"
        invalid.touch()
        expected = object()

        def execute():
            with monitor.monitor_node_call(("ThirdParty", "example"), "prompt", "node", None, "run"):
                return expected

        with (patch.dict(os.environ, {"COMFYUI_RS_NODE_MONITOR_LOG_DIR": str(invalid)}),
              patch.object(monitor, "_register"), self.assertLogs("comfyui_rs.node_monitor", level="INFO") as logs):
            result = execute()
        self.assertIs(result, expected)
        self.assertIsNone(monitor._file_handler)
        rows = [record.getMessage() for record in logs.records if record.getMessage().startswith("comfyui_rs_node ")]
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
