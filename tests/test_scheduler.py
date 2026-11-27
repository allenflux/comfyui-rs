import ast
import asyncio
import copy
import heapq
import logging
import math
import os
import sys
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, NamedTuple, Optional
from unittest.mock import AsyncMock, Mock, patch


COMFYUI = Path(__file__).resolve().parents[1] / "comfyui"


def load_definitions(filename, names, scope):
    """运行源码中的目标定义，跳过模型和设备的初始化。"""
    path = COMFYUI / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    return scope


def queue_for_test(mode=None):
    """用当前队列实现和可用的原生扩展搭建一个进程内测试队列。"""
    class Scheduler:
        """在未构建 Rust 扩展的测试环境复现调度器的接口。"""

        def __init__(self):
            """把候选任务和优先级保存在独立的表中。"""
            self.keys = {}

        def push(self, ticket, priority, prompt_id):
            """复现原生堆对重复键及非有限优先级的回退信号。"""
            key = (priority, prompt_id)
            if not math.isfinite(priority) or key in self.keys.values():
                return False
            self.keys[ticket] = key
            return True

        def pop(self):
            """优先级和任务 ID 共同决定取出顺序。"""
            ticket = min(self.keys, key=self.keys.__getitem__)
            del self.keys[ticket]
            return ticket

        def remove(self, ticket):
            """按调度票据移除指定任务。"""
            return self.keys.pop(ticket, None) is not None

        def clear(self):
            """模拟丢弃全部待执行任务。"""
            self.keys.clear()

        def tickets(self):
            """提供快照和取消接口所需的待执行票据。"""
            return list(self.keys)

    try:
        from comfyui_rs._native import PromptScheduler
    except ImportError:
        PromptScheduler = Scheduler
    scope = load_definitions("execution.py", {"PromptQueue"}, {
        "copy": copy, "heapq": heapq, "os": os, "threading": threading,
        "List": List, "Literal": Literal, "NamedTuple": NamedTuple, "Optional": Optional,
        "MAXIMUM_HISTORY_SIZE": 10000,
    })
    native = SimpleNamespace(PromptScheduler=PromptScheduler)
    with patch.dict(os.environ), patch.dict(sys.modules, {"comfyui_rs._native": native}):
        if mode is None:
            os.environ.pop("COMFYUI_RS_NATIVE_MODE", None)
        else:
            os.environ["COMFYUI_RS_NATIVE_MODE"] = mode
        return scope["PromptQueue"](SimpleNamespace(queue_updated=Mock()))


class PromptSchedulerTests(unittest.TestCase):
    def test_priority_cancellation_and_history_in_both_modes(self):
        """显式优先级、ID 比较和旧任务元组经过两种调度方式保持一致。"""
        for mode in (None, "python"):
            with self.subTest(mode=mode):
                queue = queue_for_test(mode)
                self.assertEqual(queue._scheduler is None, mode == "python")
                tasks = [(2, "later", {}), (0, "b", {}), (-1, "front", {}), (0, "a", {})]
                for item in tasks:
                    self.assertIsNone(queue.put(item))
                running, pending = queue.get_current_queue_volatile()
                self.assertEqual(running, [])
                self.assertCountEqual(pending, tasks)
                self.assertTrue(queue.delete_queue_item(lambda item: item[1] == "later"))

                for prompt_id in ("front", "a", "b"):
                    item, task_id = queue.get(timeout=0)
                    self.assertEqual(item[1], prompt_id)
                    status = queue.ExecutionStatus("success", True, [])
                    queue.task_done(task_id, {"outputs": {"value": prompt_id}}, status)
                    history = queue.get_history(prompt_id)[prompt_id]
                    self.assertEqual(history["prompt"], item)
                    self.assertTrue(history["status"]["completed"])
                self.assertEqual(queue.get_tasks_remaining(), 0)
                self.assertIsNone(queue.get(timeout=0))

    def test_fallback_preserves_complete_tuple_order_and_legacy_queue_access(self):
        """重复键、超出浮点精度的编号和旧式列表访问均保留原行为。"""
        queue = queue_for_test()
        tasks = [(5, "id", "later"), (5, "id", "earlier"), (-2, "first", {})]
        for item in tasks:
            queue.put(item)
        self.assertIsNone(queue._scheduler)
        self.assertEqual([queue.get(timeout=0)[0] for _ in tasks], sorted(tasks))

        queue = queue_for_test()
        queue.put((2**53, "z", {}))
        queue.put((2**53 + 1, "a", {}))
        self.assertEqual(queue.get(timeout=0)[0][1], "z")

        queue = queue_for_test()
        queue.put((2, "existing", {}))
        heapq.heappush(queue.queue, (1, "legacy", {}))
        self.assertEqual(queue.get(timeout=0)[0][1], "legacy")
        queue.wipe_queue()
        self.assertEqual(queue.get_tasks_remaining(), 1)

    def test_pending_limit_is_atomic(self):
        """多个线程同时抢占容量时，只有上限允许的任务能入队。"""
        queue = queue_for_test()
        barrier = threading.Barrier(8)

        def submit(index):
            """等所有提交就绪后同时检查并使用名额。"""
            barrier.wait(timeout=5)
            return queue.try_put((index, str(index), {}), max_pending=2)

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(submit, range(8))), 2)
        self.assertEqual(queue.get_tasks_remaining(), 2)
        queue.get(timeout=0)
        self.assertTrue(queue.try_put((9, "retry", {}), max_pending=2))


class PromptAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def make_route(self, *, max_pending=None, max_concurrent=None, validate=None, mode=None):
        """装载真实请求处理函数，替代网络和节点校验依赖。"""
        server = SimpleNamespace(
            trigger_on_prompt=lambda data: data,
            number=0,
            node_replace_manager=SimpleNamespace(apply_replacements=Mock()),
            deploy_to_cloud_routes=None,
            max_pending_prompts=max_pending,
            max_concurrent_submissions=max_concurrent,
            prompt_queue=queue_for_test(),
        )
        scope = load_definitions(
            "server.py", {"_create_prompt_submission_slots", "_prompt_overload_response", "submit_prompt", "post_prompt"}, {
                "asyncio": asyncio,
                "execution": SimpleNamespace(
                    validate_prompt=validate or AsyncMock(return_value=(True, None, ["1"], {})),
                    SENSITIVE_EXTRA_DATA_KEYS=(),
                ),
                "logging": logging,
                "os": os,
                "routes": SimpleNamespace(post=lambda path: lambda method: method),
                "self": server,
                "time": time,
                "uuid": uuid,
                "valid_workflow_metadata": lambda _: None,
                "workflow_metadata_from_prompt": lambda _: None,
                "web": SimpleNamespace(json_response=lambda data, status=200: (data, status)),
            },
        )
        with patch.dict(os.environ):
            if mode is None:
                os.environ.pop("COMFYUI_RS_NATIVE_MODE", None)
            else:
                os.environ["COMFYUI_RS_NATIVE_MODE"] = mode
            server.prompt_submission_slots = scope["_create_prompt_submission_slots"](max_concurrent)
        return scope["post_prompt"], server.prompt_queue

    def request(self):
        """创建一次请求，并跟踪处理函数是否读取了请求体。"""
        return SimpleNamespace(json=AsyncMock(return_value={"prompt": {}}), headers={})

    async def test_pending_limit_rejects_and_admits_retry_after_pop(self):
        """待执行队列满时返回 429，任务转为运行中后可重新提交。"""
        route, queue = self.make_route(max_pending=1)
        self.assertEqual((await route(self.request()))[1], 200)
        response, status = await route(self.request())
        self.assertEqual(status, 429)
        self.assertEqual(response["error"]["type"], "prompt_queue_full")
        queue.get(timeout=0)
        self.assertEqual((await route(self.request()))[1], 200)

    async def test_concurrent_limit_rejects_before_reading_request_body(self):
        """验证占用全部槽位时，后续请求无需解析就收到 429。"""
        entered = asyncio.Event()
        proceed = asyncio.Event()

        async def validate(*args):
            """阻塞首个校验，确定第二个请求抵达时名额已占用。"""
            entered.set()
            await proceed.wait()
            return (True, None, ["1"], {})

        route, _ = self.make_route(max_concurrent=1, validate=validate)
        pending = asyncio.create_task(route(self.request()))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            request = self.request()
            response, status = await route(request)
            self.assertEqual(status, 429)
            self.assertEqual(response["error"]["type"], "prompt_submission_busy")
            request.json.assert_not_called()
        finally:
            proceed.set()
            self.assertEqual((await pending)[1], 200)
        self.assertEqual((await route(self.request()))[1], 200)

    async def test_cancelled_submission_returns_its_slot(self):
        """请求取消后立即释放并发名额，供下一次提交使用。"""
        for mode in (None, "python"):
            with self.subTest(mode=mode):
                entered = asyncio.Event()

                async def validate(*args):
                    """首个校验暂停等待取消，之后正常接受提交。"""
                    if not entered.is_set():
                        entered.set()
                        await asyncio.Future()
                    return (True, None, ["1"], {})

                route, _ = self.make_route(max_concurrent=1, validate=validate, mode=mode)
                pending = asyncio.create_task(route(self.request()))
                await asyncio.wait_for(entered.wait(), timeout=5)
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
                self.assertEqual((await route(self.request()))[1], 200)


if __name__ == "__main__":
    unittest.main()
