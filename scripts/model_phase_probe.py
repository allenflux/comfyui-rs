"""为 GPU 压测记录 ComfyUI 模型阶段耗时，由两个镜像加载同一份探针。"""

from contextvars import ContextVar
from functools import wraps
import json
import logging
import math
import os
from time import perf_counter_ns
import weakref


NODE_CLASS_MAPPINGS = {}
_SCHEMA = 1
_CUDA_EVENT_LIMIT = 256
_REPORT_LIMIT = 32 * 1024 ** 2
_PHASES = (
    "checkpoint_read", "model_setup", "model_load", "load_prepare", "model_request",
    "model_switch", "model_release", "vram_reclaim", "allocator_trim", "weight_move_host",
)


class _PromptTiming:
    """在单个工作流内保留主机计时和有上限的 CUDA Event 对。"""

    def __init__(self):
        self.host = {name: [0, 0] for name in _PHASES}
        self.active = set()
        self.cuda = False
        self.events = []
        self.event_attempts = 0
        self.events_dropped = 0
        self.events_failed = 0
        self.errors = 0

    def add(self, phase, elapsed):
        """累计主机调用次数和已用纳秒。"""
        self.host[phase][0] += 1
        self.host[phase][1] += elapsed

    def report(self, prompt_id, elapsed):
        """读取已完成的事件，保留未完成、超限以及读取失败的计数。"""
        intervals = []
        incomplete = 0
        for start, end in self.events:
            try:
                if not end.query():
                    incomplete += 1
                    continue
                interval = start.elapsed_time(end)
                if not math.isfinite(interval) or interval < 0:
                    raise ValueError("invalid CUDA event interval")
                intervals.append(interval)
            except Exception:
                self.events_failed += 1
        return {
            "schema": _SCHEMA,
            "prompt_id": prompt_id,
            "workflow_host_ms": elapsed / 1e6,
            "phases": {name: {"calls": count, "host_ms": duration / 1e6}
                       for name, (count, duration) in self.host.items()},
            "weight_move_cuda": {
                "events_complete": len(intervals),
                "events_incomplete": incomplete,
                "events_dropped": self.events_dropped,
                "events_failed": self.events_failed,
                "stream_ms": math.fsum(intervals) if intervals else None,
            },
            "probe_errors": self.errors,
        }


class _Report:
    """将有界的结构化记录写到独立文件，避免读取工作流日志和敏感参数。"""

    def __init__(self, path):
        self.path = path
        self.size = 0
        self.full = False

    def write(self, row):
        """附加一条记录，达到文件上限时写入截断标记。"""
        if self.full:
            return
        line = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if self.size + len(line) > _REPORT_LIMIT - 1024:
            line = b'{"schema":1,"probe_truncated":true}\n'
            self.full = True
        with open(self.path, "ab" if self.size else "wb") as report:
            report.write(line)
        self.size += len(line)


def _argument(args, kwargs, index, name):
    """读取已知入口的位置参数或同名关键字参数。"""
    return args[index] if len(args) > index else kwargs.get(name)


def _warn(message):
    """探针告警不能盖过工作流的返回值或异常。"""
    try:
        logging.getLogger(__name__).warning(message)
    except Exception:
        pass


def _install(path):
    """挂接改前和改后版本共用的入口，阶段时长为主机含子调用的耗时。"""
    import torch
    import execution
    from comfy import model_management, model_patcher, sd, utils

    executor = execution.PromptExecutor
    execute_async = executor.execute_async
    if getattr(execute_async, "_model_phase_probe", False):
        return
    report = _Report(path)
    current_timing = ContextVar("comfyui_model_phase_prompt", default=None)
    last_request = None
    installations = []
    unavailable = []

    def install_hook(owner, name, label, wrap):
        """保存回退用的属性值并登记已覆盖的入口。"""
        original = getattr(owner, name, None)
        if original is None:
            unavailable.append(label)
            return
        local = name in vars(owner)
        setattr(owner, name, wrap(original))
        installations.append((owner, name, original, local, label))

    def host_timer(phase, cuda_only=False):
        """同一阶段嵌套时只记录最外层调用，阶段之间可重叠。"""
        def wrap(original):
            @wraps(original)
            def timed(*args, **kwargs):
                state = current_timing.get()
                if state is None or phase in state.active:
                    return original(*args, **kwargs)
                if cuda_only and getattr(_argument(args, kwargs, 1, "device"), "type", None) != "cuda":
                    return original(*args, **kwargs)
                state.active.add(phase)
                started = perf_counter_ns()
                try:
                    return original(*args, **kwargs)
                finally:
                    state.add(phase, perf_counter_ns() - started)
                    state.active.remove(phase)
            return timed
        return wrap

    def requested_models(original):
        """用弱引用比较相邻的模型请求集合，避免探针延长模型生命周期。"""
        @wraps(original)
        def timed(*args, **kwargs):
            nonlocal last_request
            state = current_timing.get()
            if state is None or "model_request" in state.active:
                return original(*args, **kwargs)
            current = None
            switched = False
            try:
                models = _argument(args, kwargs, 0, "models")
                if type(models) in (list, tuple) and 0 < len(models) <= 128:
                    current = {id(model): weakref.ref(model) for model in models}
                    switched = last_request is not None and (
                        last_request.keys() != current.keys() or any(
                            last_request[key]() is not reference() for key, reference in current.items()
                        )
                    )
            except Exception:
                state.errors += 1
            state.active.add("model_request")
            started = perf_counter_ns()
            try:
                result = original(*args, **kwargs)
                last_request = current
                return result
            finally:
                elapsed = perf_counter_ns() - started
                state.add("model_request", elapsed)
                if switched:
                    state.add("model_switch", elapsed)
                state.active.remove("model_request")
        return timed

    def model_placement(args, kwargs):
        """识别在模型生命周期中可见的模块设备转换。"""
        target = _argument(args, kwargs, 1, "device")
        if isinstance(target, torch.Tensor):
            target = target.device
        elif isinstance(target, (str, int)):
            target = torch.device(target)
        if not isinstance(target, torch.device):
            return None
        first = next(args[0].parameters(), None)
        if first is None or first.device == target or first.device.type == "meta":
            return None
        return (target if target.type == "cuda" else first.device), None

    def single_cast(args, kwargs):
        """定位权重转换所在设备，记录含拷贝或类型转换的调用。"""
        tensor = _argument(args, kwargs, 0, "tensor")
        target = _argument(args, kwargs, 1, "device")
        if target is None or not hasattr(tensor, "device"):
            return None
        if not isinstance(target, torch.device):
            target = torch.device(target)
        dtype = _argument(args, kwargs, 2, "dtype")
        if (tensor.device == target and dtype in (None, tensor.dtype)
                and not _argument(args, kwargs, 3, "copy")):
            return None
        return (target if target.type == "cuda" else tensor.device), None

    def gathered_cast(args, kwargs):
        """使用批量权重转换自己的目标缓冲区和流。"""
        destination2 = _argument(args, kwargs, 4, "r2")
        destination = destination2 if destination2 is not None else _argument(args, kwargs, 1, "r")
        stream = _argument(args, kwargs, 3, "stream")
        device = getattr(stream, "device", None) if stream is not None else getattr(destination, "device", None)
        # AIMDO 文件直读在没有显式流时可能使用默认流。
        return (device, stream, stream is None) if device is not None else None

    def start_event(state, device, stream, require_default=False):
        """只在可识别的 CUDA 流记录事件，避免图捕获和额外同步。"""
        if state.event_attempts >= _CUDA_EVENT_LIMIT:
            state.events_dropped += 1
            return None
        state.event_attempts += 1
        try:
            if torch.cuda.is_current_stream_capturing():
                state.events_dropped += 1
                return None
            if stream is None:
                stream = torch.cuda.current_stream(device)
                if require_default and stream != torch.cuda.default_stream(device):
                    state.events_dropped += 1
                    return None
            elif not isinstance(stream, torch.cuda.Stream):
                state.events_dropped += 1
                return None
            with torch.cuda.stream(stream):
                if torch.cuda.is_current_stream_capturing():
                    state.events_dropped += 1
                    return None
                start = torch.cuda.Event(enable_timing=True)
                start.record(stream)
            return start, stream
        except Exception:
            state.events_failed += 1
            return None

    def weight_timer(select, requires_model_scope):
        """区分主机提交耗时和探针覆盖的 CUDA 流内时段。"""
        model_scopes = {"checkpoint_read", "model_setup", "model_load", "model_request", "model_release"}

        def wrap(original):
            @wraps(original)
            def timed(*args, **kwargs):
                if torch.compiler.is_compiling():
                    return original(*args, **kwargs)
                state = current_timing.get()
                if (state is None or not state.cuda or "weight_move_host" in state.active
                        or (requires_model_scope and model_scopes.isdisjoint(state.active))):
                    return original(*args, **kwargs)
                try:
                    selected = select(args, kwargs)
                except Exception:
                    state.errors += 1
                    selected = None
                if selected is None or getattr(selected[0], "type", None) != "cuda":
                    return original(*args, **kwargs)
                span = start_event(state, *selected)
                state.active.add("weight_move_host")
                started = perf_counter_ns()
                try:
                    return original(*args, **kwargs)
                finally:
                    state.add("weight_move_host", perf_counter_ns() - started)
                    state.active.remove("weight_move_host")
                    if span is not None:
                        try:
                            start, stream = span
                            end = torch.cuda.Event(enable_timing=True)
                            end.record(stream)
                            state.events.append((start, end))
                        except Exception:
                            state.events_failed += 1
            return timed
        return wrap

    @wraps(execute_async)
    async def timed_execute(self, prompt, prompt_id, *args, **kwargs):
        """汇总同一工作流的阶段记录，读取 CUDA 事件时不等待未完成的搬运。"""
        if not isinstance(prompt_id, str):
            return await execute_async(self, prompt, prompt_id, *args, **kwargs)
        state = _PromptTiming()
        try:
            state.cuda = torch.version.cuda is not None and model_management.get_torch_device().type == "cuda"
        except Exception:
            state.errors += 1
        token = current_timing.set(state)
        started = perf_counter_ns()
        try:
            return await execute_async(self, prompt, prompt_id, *args, **kwargs)
        finally:
            elapsed = perf_counter_ns() - started
            current_timing.reset(token)
            try:
                report.write(state.report(prompt_id, elapsed))
            except Exception:
                _warn("ComfyUI 模型阶段探针无法写入记录")

    timed_execute._model_phase_probe = True
    try:
        install_hook(utils, "load_torch_file", "comfy.utils.load_torch_file", host_timer("checkpoint_read"))
        for name in ("load_state_dict_guess_config", "load_diffusion_model_state_dict",
                     "load_text_encoder_state_dicts", "load_vae_patcher"):
            install_hook(sd, name, f"comfy.sd.{name}", host_timer("model_setup"))
        for name, phase in (("model_load", "model_load"), ("model_unload", "model_release")):
            install_hook(model_management.LoadedModel, name, f"LoadedModel.{name}", host_timer(phase))
        for name, phase in (("load", "model_load"), ("detach", "model_release"), ("_load_list", "load_prepare")):
            install_hook(model_patcher.ModelPatcher, name, f"ModelPatcher.{name}", host_timer(phase))
        if hasattr(model_patcher, "ModelPatcherDynamic"):
            install_hook(model_patcher.ModelPatcherDynamic, "load", "ModelPatcherDynamic.load", host_timer("model_load"))
        install_hook(model_management, "load_models_gpu", "model_management.load_models_gpu", requested_models)
        install_hook(model_management, "free_memory", "model_management.free_memory", host_timer("vram_reclaim", cuda_only=True))
        install_hook(model_management, "soft_empty_cache", "model_management.soft_empty_cache", host_timer("allocator_trim"))
        install_hook(torch.nn.Module, "to", "torch.nn.Module.to", weight_timer(model_placement, True))
        install_hook(model_management, "cast_to_device", "model_management.cast_to_device", weight_timer(single_cast, True))
        install_hook(model_management, "cast_to_gathered", "model_management.cast_to_gathered", weight_timer(gathered_cast, False))
        install_hook(executor, "execute_async", "PromptExecutor.execute_async", lambda _: timed_execute)
        report.write({
            "schema": _SCHEMA, "probe_ready": True,
            "hooks": [item[-1] for item in installations], "unavailable_hooks": unavailable,
            "cuda_event_limit": _CUDA_EVENT_LIMIT, "report_limit_bytes": _REPORT_LIMIT,
        })
    except Exception:
        for owner, name, original, local, _ in reversed(installations):
            if local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        raise


_path = os.getenv("COMFYUI_RS_MODEL_PHASE_PATH")
if _path:
    try:
        _install(_path)
    except Exception:
        _warn("ComfyUI 模型阶段探针初始化失败")
