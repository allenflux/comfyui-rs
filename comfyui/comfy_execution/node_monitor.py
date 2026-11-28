"""在执行器边界观测第三方节点的时延和内存。"""

import asyncio
from contextlib import contextmanager
from copy import deepcopy
from itertools import count
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import socket
import threading
import time
from uuid import uuid4

import psutil
import torch

from comfy import model_management, system_memory


_LOGGER = logging.getLogger("comfyui_rs.node_monitor")
_POLL_SECONDS = 0.25
_HEARTBEAT_SECONDS = 2.0
_FILE_MAX_BYTES = 8 * 1024**2
_FILE_BACKUPS = 4
_FILE_RETRY_SECONDS = 30.0
_WORKER_RUN_ID = uuid4().hex
_CALL_IDS = count(1)
_ACTIVE = {}
_CONDITION = threading.Condition()
_FILE_LOCK = threading.Lock()
_sampler = None
_process = None
_file_handler = None
_file_retry_at = 0.0


def custom_node_source(obj):
    """用加载器登记的包名识别第三方节点，兼容旧版实例和新版类。"""
    cls = obj if isinstance(obj, type) else type(obj)
    module = _read_metric(lambda: getattr(cls, "RELATIVE_PYTHON_MODULE", None))
    if isinstance(module, str) and module.startswith("custom_nodes."):
        pack = module.removeprefix("custom_nodes.")
        if pack:
            return _read_metric(lambda: (cls.__name__, pack))
    return None


def _read_metric(read):
    """采集失败时只丢弃该指标，保持节点的结果和异常。"""
    try:
        return read()
    except Exception:
        return None


def _rss_bytes():
    """重复使用本进程句柄，读取居民内存字节数。"""
    global _process
    if _process is None:
        _process = psutil.Process()
    return _process.memory_info().rss


class _JsonlHandler(RotatingFileHandler):
    def handleError(self, record):
        """把磁盘错误交给可恢复的写入路径，避免日志系统输出回溯。"""
        raise


def _new_file_handler():
    """首次写入时定位有效的用户目录，并为当前 worker 创建滚动日志。"""
    override = os.getenv("COMFYUI_RS_NODE_MONITOR_LOG_DIR")
    if override:
        directory = Path(override).expanduser()
    else:
        import folder_paths

        directory = Path(folder_paths.get_user_directory()) / "node-monitor"
    hostname = re.sub(r"[^A-Za-z0-9._-]", "_", socket.gethostname())[:120] or "unknown"
    directory.mkdir(parents=True, exist_ok=True)
    return _JsonlHandler(
        directory / f"node-monitor-{hostname}-{os.getpid()}.jsonl",
        maxBytes=_FILE_MAX_BYTES,
        backupCount=_FILE_BACKUPS,
        encoding="utf-8",
    )


def _write_jsonl(payload):
    """独立写入纯 JSON 行，文件或挂载异常时稍后重试。"""
    global _file_handler, _file_retry_at
    with _FILE_LOCK:
        if time.monotonic() < _file_retry_at:
            return
        try:
            if _file_handler is None:
                _file_handler = _new_file_handler()
            record = logging.LogRecord(_LOGGER.name, logging.INFO, "", 0, payload, (), None)
            _file_handler.handle(record)
        except Exception:
            if _file_handler is not None:
                _read_metric(_file_handler.close)
                _file_handler = None
            _file_retry_at = time.monotonic() + _FILE_RETRY_SECONDS


def _cuda_memory():
    """只读已初始化设备的 PyTorch 分配器，无需同步或重置计数器。"""
    version = torch.version
    backend = "cuda" if getattr(version, "cuda", None) else "hip" if getattr(version, "hip", None) else None
    if backend is None or not torch.cuda.is_initialized():
        return None
    devices = {}
    for index in range(torch.cuda.device_count()):
        stats = _read_metric(lambda: torch.cuda.memory_stats(index))
        if stats is not None:
            devices[str(index)] = {
                "allocated_bytes": stats.get("allocated_bytes.all.current"),
                "reserved_bytes": stats.get("reserved_bytes.all.current"),
            }
    return {
        "backend": backend,
        "allocator_backend": _read_metric(lambda: torch.cuda.get_allocator_backend()),
        "devices": devices,
    }


def _snapshot():
    """区分进程 RSS、容器有效内存余量以及设备分配器计数。"""
    return {
        "rss_bytes": _read_metric(_rss_bytes),
        "ram_available_bytes": _read_metric(lambda: system_memory.virtual_memory_available()),
        "cgroup_limit_bytes": _read_metric(lambda: system_memory.cgroup_memory_limit()),
        "cuda": _read_metric(_cuda_memory),
    }


def _outcome(error):
    """识别可观测的 OOM 信号，保留异常类型而不记录文本。"""
    if error is None:
        return "success", None, None

    kind = type(error)
    name = f"{kind.__module__}.{kind.__qualname__}"
    interrupt = getattr(model_management, "InterruptProcessingException", None)
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) or (
        isinstance(interrupt, type) and isinstance(error, interrupt)
    ):
        return "interrupted", None, name
    if isinstance(error, MemoryError):
        return "oom", "python_memory", name
    cuda = getattr(torch, "cuda", None)
    for target in (getattr(torch, "OutOfMemoryError", None), getattr(cuda, "OutOfMemoryError", None)):
        if isinstance(target, type) and isinstance(error, target):
            return "oom", "torch_allocator", name
    if isinstance(error, RuntimeError) and error.args and type(error.args[0]) is str:
        text = error.args[0][:1024].lower()
        if any(marker in text for marker in ("cuda out of memory", "cuda error: out of memory", "cuda_error_out_of_memory")):
            return "oom", "cuda_runtime", name
        if "defaultcpuallocator" in text and ("can't allocate memory" in text or "cannot allocate memory" in text):
            return "oom", "cpu_allocator", name
    return "error", None, name


def _extreme(previous, observed, choose):
    """更新已采样的极值，缺失样本不清除已有数据。"""
    if observed is None:
        return previous
    return observed if previous is None else choose(previous, observed)


class _NodeCall:
    """保存单次调用的采样极值，不保留历史采样列表。"""

    def __init__(self, source, prompt_id, node_id, list_index, function):
        """为同步调用和异步调用分配独立的记录身份。"""
        node_class, pack = source
        self.call_id = next(_CALL_IDS)
        self.identity = {
            "schema": "comfyui-rs.node-memory.v1",
            "prompt_id": prompt_id,
            "node_id": node_id,
            "call_id": self.call_id,
            "node_class": node_class,
            "pack": pack,
            "function": function,
            "list_index": list_index,
            "pid": os.getpid(),
            "worker_run_id": _WORKER_RUN_ID,
        }
        self.started = 0.0
        self.last_event = 0.0
        self.samples = 0
        self.start_memory = None
        self.peak = {"rss_bytes": None, "min_ram_available_bytes": None, "cuda": None}
        self.state_lock = threading.Lock()
        self.emit_lock = threading.RLock()
        self.finished = threading.Event()

    def _update_peak(self, snapshot):
        """更新进程和各卡采样峰值，以及采样到的最小可用 RAM。"""
        self.peak["rss_bytes"] = _extreme(self.peak["rss_bytes"], snapshot["rss_bytes"], max)
        minimum = self.peak["min_ram_available_bytes"]
        self.peak["min_ram_available_bytes"] = _extreme(minimum, snapshot["ram_available_bytes"], min)
        cuda = snapshot["cuda"]
        if cuda is not None:
            if self.peak["cuda"] is None:
                self.peak["cuda"] = {
                    "backend": cuda["backend"], "allocator_backend": cuda["allocator_backend"], "devices": {},
                }
            peaks = self.peak["cuda"]["devices"]
            for index, values in cuda["devices"].items():
                device = peaks.setdefault(index, {})
                for metric, value in values.items():
                    device[metric] = _extreme(device.get(metric), value, max)

    def record(self, event, snapshot, now, error=None):
        """在锁内构造独立快照，避免日志处理器与采样状态相互等待。"""
        with self.state_lock:
            if event == "sample" and self.finished.is_set():
                return None
            if event == "start":
                self.started = now
                self.start_memory = snapshot
            self.samples += 1
            self._update_peak(snapshot)
            if event == "sample" and now - self.last_event < _HEARTBEAT_SECONDS:
                return None
            self.last_event = now
            status, oom_kind, exc_type = _outcome(error) if event == "finish" else ("running", None, None)
            return {
                **self.identity,
                "event": event,
                "timestamp_ms": time.time_ns() // 1_000_000,
                "duration_ms": (now - self.started) * 1000,
                "samples": self.samples,
                "status": status,
                "oom_kind": oom_kind,
                "exception_type": exc_type,
                "memory": deepcopy({"start": self.start_memory, "current": snapshot, "peak": self.peak}),
            }

    def emit(self, record):
        """同一次调用的日志串行输出，完成后丢弃过期心跳。"""
        if record is None:
            return
        with self.emit_lock:
            if record["event"] != "sample" or not self.finished.is_set():
                payload = _read_metric(lambda: json.dumps(record, separators=(",", ":")))
                if payload is not None:
                    _read_metric(lambda: _LOGGER.info("comfyui_rs_node %s", payload))
                    _read_metric(lambda: _write_jsonl(payload))


def _poll():
    """用同一份快照更新当前所有调用，没有节点在跑时休眠。"""
    while True:
        with _CONDITION:
            _CONDITION.wait_for(lambda: bool(_ACTIVE))
        time.sleep(_POLL_SECONDS)
        with _CONDITION:
            calls = tuple(_ACTIVE.values())
        if calls:
            snapshot = _snapshot()
            now = time.perf_counter()
            for call in calls:
                _read_metric(lambda: call.emit(call.record("sample", snapshot, now)))


def _register(call):
    """首次有节点运行时启动采样线程，之后复用同一个守护线程。"""
    global _sampler
    with _CONDITION:
        if _sampler is None:
            thread = threading.Thread(target=_poll, name="comfyui-node-monitor", daemon=True)
            thread.start()
            _sampler = thread
        _ACTIVE[call.call_id] = call
        _CONDITION.notify()


def _finish(call, error):
    """先退出活动集合，再记录结束快照，避免结束之后继续采样。"""
    now = time.perf_counter()
    call.finished.set()
    with _CONDITION:
        _ACTIVE.pop(call.call_id, None)
    call.emit(call.record("finish", _snapshot(), now, error))


@contextmanager
def monitor_node_call(source, prompt_id, node_id, list_index, function):
    """包住真实节点方法，监测故障不影响调用本身。"""
    call = None
    if source is not None:
        candidate = _read_metric(lambda: _NodeCall(source, prompt_id, node_id, list_index, function))
        if candidate is not None:
            try:
                candidate.emit(candidate.record("start", _snapshot(), time.perf_counter()))
                call = candidate
                # 起始日志写出后开始计时，耗时不含初始采样和日志阻塞。
                call.started = time.perf_counter()
                _read_metric(lambda: _register(call))
            except Exception:
                pass
    try:
        yield
    except BaseException as error:
        if call is not None:
            _read_metric(lambda: _finish(call, error))
        raise
    else:
        if call is not None:
            _read_metric(lambda: _finish(call, None))
