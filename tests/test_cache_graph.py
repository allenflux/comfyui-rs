import ast
import bisect
import logging
import math
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    from comfyui_rs import _native
except ImportError:
    _native = None


EXECUTION = Path(__file__).resolve().parents[1] / "comfyui" / "comfy_execution"


def load_methods(filename, class_name, names, **scope):
    """从实际模块加载相关方法，同时隔离 PyTorch 初始化。"""
    source = EXECUTION / filename
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    return type(class_name, (), {name: scope[name] for name in names})()


def load_ram_cache(keys, generations, timestamps, removed, ram_eviction_order):
    """构造可记录删除顺序的 RAM 压力缓存，并在两次淘汰后恢复内存。"""
    class Entries(dict):
        def __delitem__(self, key):
            """按实际删除顺序记录条目键。"""
            super().__delitem__(key)
            removed.append(key)

    cache = load_methods(
        "caching.py", "RAMPressureCache", ("ram_release",),
        _ram_eviction_order=ram_eviction_order,
        virtual_memory_available=lambda: 100 if len(removed) >= 2 else 0,
        all_outputs_dynamic=lambda outputs: False,
        is_model_patcher_output=lambda output: False,
        RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER=1.3,
        RAM_CACHE_DEFAULT_RAM_USAGE=0.05,
        torch=SimpleNamespace(Tensor=type("Tensor", (), {})),
        Mapping=Mapping, bisect=bisect, math=math, logging=logging,
    )
    cache.cache = Entries.fromkeys(keys, SimpleNamespace(outputs=[object()]))
    cache.used_generation = dict(zip(keys, generations, strict=True))
    cache.timestamps = dict(zip(keys, timestamps, strict=True))
    cache.children = {}
    cache.generation = 4
    return cache


class CacheTests(unittest.TestCase):
    def test_lru_retires_oldest_generations_and_keeps_active_entries(self):
        """验证活跃条目允许超过上限，并保留其插入顺序。"""
        native = _native.lru_eviction_indices if _native is not None else None
        eviction_indices = Mock(side_effect=native) if native is not None else None
        selectors = (None, eviction_indices) if eviction_indices is not None else (None,)
        for selector in selectors:
            with self.subTest(rust=selector is not None):
                cache = load_methods(
                    "caching.py", "LRUCache", ("clean_unused", "_keys_before_generation"),
                    _lru_eviction_indices=selector,
                )
                cache.cache = dict.fromkeys(("recent-a", "old-b", "recent-b", "old-a", "recent-c"))
                cache.used_generation = {"recent-a": 3, "old-b": 1, "recent-b": 3, "old-a": 0, "recent-c": 3}
                cache.children = {"old-a": ["subgraph"], "recent-c": ["keep"]}
                cache.max_size, cache.min_generation, cache.generation = 2, 0, 3
                cache._clean_subcaches = Mock()

                cache.clean_unused()

                self.assertEqual(list(cache.cache), ["recent-a", "recent-b", "recent-c"])
                self.assertEqual(cache.used_generation, dict.fromkeys(cache.cache, 3))
                self.assertEqual(cache.children, {"recent-c": ["keep"]})
                self.assertEqual(cache.min_generation, 3)
                cache._clean_subcaches.assert_called_once()
        if eviction_indices is not None:
            self.assertEqual(eviction_indices.call_count, 3)

    def test_ram_eviction_ranks_old_entries_and_stops_at_memory_target(self):
        """检查年代及触碰时间的优先级，并在可用内存达标时停止。"""
        native = _native.ram_eviction_order if _native is not None else None
        ram_eviction_order = Mock(side_effect=native) if native is not None else None
        selectors = (None, ram_eviction_order) if ram_eviction_order is not None else (None,)
        for selector in selectors:
            with self.subTest(rust=selector is not None):
                removed = []
                cache = load_ram_cache(
                    ("old-first", "current", "middle", "old-recent"),
                    (1, 4, 2, 1), (1.0, 10.0, 2.0, 3.0), removed, selector,
                )
                self.assertAlmostEqual(cache.ram_release(100), 0.1)
                self.assertEqual(removed, ["old-recent", "old-first"])
                self.assertEqual(list(cache.cache), ["current", "middle"])
        if ram_eviction_order is not None:
            ram_eviction_order.assert_called_once()

    def test_ram_priority_tie_preserves_python_key_comparison(self):
        """评分和时间戳均相同时，继续使用 Python 的完整键比较。"""
        ram_eviction_order = Mock()
        removed = []
        cache = load_ram_cache((object(), object()), (1, 1), (2.0, 2.0), removed, ram_eviction_order)
        with self.assertRaises(TypeError):
            cache.ram_release(100)
        self.assertEqual(removed, [])
        ram_eviction_order.assert_not_called()


class GraphTests(unittest.TestCase):
    def inhibition_graph(self, planner):
        """构造带内部链接、下游依赖和缓存监听器的控制图。"""
        graph = load_methods(
            "graph.py", "ExecutionList", ("inhibit_nodes", "_get_inhibition_plan_native"),
            _graph_inhibition_plan=planner, _RUST_INHIBITION_SCANS_PER_SOURCE=0,
        )
        graph.staged_node_id = "control"
        graph.pendingNodes = dict.fromkeys(("control", "keep", "left", "right", "target"), True)
        graph.blockCount = {"control": 0, "keep": 1, "left": 2, "right": 2, "target": 4}
        graph.blocking = {
            "control": {"right": {3: True}, "left": {0: True}, "target": {0: True}},
            "keep": {"left": {0: True}, "target": {0: True}},
            "left": {"right": {2: True}, "target": {0: True}},
            "right": {"keep": {1: True}, "target": {8: True}},
            "target": {},
        }
        graph.execution_cache = {"left": {"control": 1}, "right": {"left": 2}, "target": {}}
        graph.execution_cache_listeners = {
            "control": {("left", 0), ("right", 3), ("target", 0)},
            "keep": {("left", 0), ("target", 0)},
            "left": {("target", 0)},
            "right": {("keep", 1)},
        }
        return graph

    def test_inhibition_matches_python_including_cleanup_mutation(self):
        """比较禁用结果和缓存清理顺序，清理期间的依赖变化仍然生效。"""
        native = getattr(_native, "graph_inhibition_plan", None)
        selectors = (None, Mock(wraps=native)) if native is not None else (None,)
        results = []
        for selector in selectors:
            graph = self.inhibition_graph(selector)
            cleanup_order = []

            class Cache(dict):
                def pop(self, key, default=None):
                    """清理第一个禁用节点时给后续节点增加一条依赖。"""
                    cleanup_order.append((key, graph.blockCount.copy()))
                    if len(cleanup_order) == 1:
                        other, = {"left", "right"}.intersection(graph.pendingNodes)
                        graph.blocking[other]["control"] = {0: True}
                        graph.blockCount["control"] += 1
                    return super().pop(key, default)

            graph.execution_cache = Cache(graph.execution_cache)
            graph.inhibit_nodes(["left", "right", "absent", "left"])
            self.assertEqual(list(graph.pendingNodes), ["control", "keep", "target"])
            self.assertEqual(graph.blockCount, {"control": 0, "keep": 0, "target": 2})
            self.assertEqual(graph.blocking, {"control": {"target": {0: True}}, "keep": {"target": {0: True}}, "target": {}})
            self.assertEqual(graph.execution_cache, {"target": {}})
            self.assertEqual(graph.execution_cache_listeners, {"control": {("target", 0)}, "keep": {("target", 0)}})
            results.append(cleanup_order)
            if selector is not None:
                selector.assert_called_once()
        if len(results) == 2:
            self.assertEqual(*results)

    def test_inhibition_keeps_custom_graph_mapping_on_python_path(self):
        """自定义邻接映射保持 Python 删除语义。"""
        class Links(dict):
            pass

        planner = Mock()
        graph = self.inhibition_graph(planner)
        graph.blocking["control"] = Links(graph.blocking["control"])
        graph.inhibit_nodes(["left"])
        planner.assert_not_called()
        self.assertEqual(list(graph.blocking["control"]), ["right", "target"])
        self.assertEqual(graph.blockCount["right"], 1)

    def test_ready_nodes_keep_pending_order_and_fall_back_on_overflow(self):
        """检查就绪顺序，以及阻塞计数超出原生范围时的回退。"""
        native = _native.graph_ready_indices if _native is not None else None
        ready_indices = Mock(side_effect=native) if native is not None else None
        graph = load_methods("graph.py", "TopologicalSort", ("get_ready_nodes",), _graph_ready_indices=ready_indices)
        graph.pendingNodes = dict.fromkeys(("later", "first", "negative", "second"))
        graph.blockCount = {"later": 1, "first": 0, "negative": -1, "second": 0}
        self.assertEqual(graph.get_ready_nodes(), ["first", "second"])

        graph.blockCount["later"] = 2**80
        self.assertEqual(graph.get_ready_nodes(), ["first", "second"])
        if ready_indices is not None:
            self.assertEqual(ready_indices.call_count, 2)

    def test_cycle_includes_downstream_nodes_and_ignores_inactive_sockets(self):
        """检查环检测只采用有效连接，并保留无法解锁的下游节点。"""
        native = _native.graph_cycle_indices if _native is not None else None
        cycle_indices = Mock(side_effect=native) if native is not None else None
        selectors = (None, cycle_indices) if cycle_indices is not None else (None,)
        for selector in selectors:
            with self.subTest(rust=selector is not None):
                graph = load_methods(
                    "graph.py", "ExecutionList", ("get_nodes_in_cycle", "_get_nodes_in_cycle_native"),
                    _graph_cycle_indices=selector,
                )
                graph.pendingNodes = dict.fromkeys(("downstream", "right", "independent", "left", "tail"))
                graph.blocking = {
                    "downstream": {},
                    "right": {"left": {0: True}, "downstream": {0: True}},
                    "independent": {"tail": {0: True}},
                    "left": {"right": {0: True}, "independent": {0: False}},
                    "tail": {},
                }
                self.assertEqual(graph.get_nodes_in_cycle(), ["downstream", "right", "left"])
        if cycle_indices is not None:
            cycle_indices.assert_called_once()

    def test_dependency_from_outside_pending_nodes_stays_blocked(self):
        """确认待执行列表之外的阻塞来源交给 Python 处理。"""
        cycle_indices = Mock()
        graph = load_methods(
            "graph.py", "ExecutionList", ("get_nodes_in_cycle", "_get_nodes_in_cycle_native"),
            _graph_cycle_indices=cycle_indices,
        )
        graph.pendingNodes = {"target": True}
        graph.blocking = {"external": {"target": {0: True}}}
        self.assertEqual(graph.get_nodes_in_cycle(), ["target"])
        cycle_indices.assert_not_called()


if __name__ == "__main__":
    unittest.main()
