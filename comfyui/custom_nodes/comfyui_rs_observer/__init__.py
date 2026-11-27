"""为专用 ComfyUI 测试 worker 提供可选的工作流观测。"""

from functools import wraps
import json
import logging
import os
import time


NODE_CLASS_MAPPINGS = {}
_logger = logging.getLogger("comfyui_rs.observer")


def _warn(message):
    """尽力发出观测告警，即使日志处理器失败也继续推理。"""
    try:
        _logger.warning(message)
    except Exception:
        pass


def _budget_config():
    """加载以无符号 64 位字节数配置的 Rust 预算评估，无法启用时返回 None。"""
    expected = os.getenv("COMFYUI_RS_EXPECTED_EXTRA_BYTES")
    if expected is None:
        return None
    try:
        expected = int(expected)
        headroom = int(os.getenv("COMFYUI_RS_HEADROOM_BYTES", "0"))
        if not (0 <= expected < 2**64 and 0 <= headroom < 2**64):
            raise ValueError
    except ValueError:
        _warn("ComfyUI Rust observer budget values must be unsigned 64-bit byte counts")
        return None
    try:
        from comfyui_rs import assess_budget
    except Exception:
        _warn("ComfyUI Rust observer could not load the native budget extension")
        return None
    return assess_budget, expected, headroom


def _execution_status(executor, prompt_id, escaped_error, model_management, torch):
    """优先识别执行抛出的异常，再从当前工作流的最近事件判定结果。"""
    if escaped_error is not None:
        if isinstance(escaped_error, model_management.InterruptProcessingException):
            return "interrupted"
        if isinstance(escaped_error, torch.cuda.OutOfMemoryError):
            return "oom"
        return "error"

    for event, data in reversed(executor.status_messages):
        if data.get("prompt_id") != prompt_id:
            continue
        if event == "execution_success":
            return "success"
        if event == "execution_interrupted":
            return "interrupted"
        if event == "execution_error":
            exception_type = data.get("exception_type", "").rsplit(".", 1)[-1]
            # ComfyUI 识别到加速器显存不足时会附加这段提示。
            oom_marker = "\nThis error means you ran out of memory on your GPU.\n\nTIPS:"
            if exception_type == "OutOfMemoryError" or oom_marker in data.get("exception_message", ""):
                return "oom"
            return "error"
    return "success" if executor.success else "error"


def _install():
    """首次加载时为工作流执行方法安装观测包装，重复加载时复用原包装。"""
    import execution
    import torch
    from comfy import model_management

    original = execution.PromptExecutor.execute
    if getattr(original, "_comfyui_rs_observer", False):
        return
    budget_config = _budget_config()

    @wraps(original)
    def observe(self, *args, **kwargs):
        """记录工作流耗时、结果与可用的 CUDA 指标，保留原调用的返回值和异常。"""
        prompt_id = args[1] if len(args) > 1 else kwargs.get("prompt_id")
        if not isinstance(prompt_id, str):
            return original(self, *args, **kwargs)

        cuda = None
        counters_started = False
        try:
            device = model_management.get_torch_device()
            if device.type == "cuda" and torch.version.cuda is not None:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                cuda = {"device_index": device.index, "driver_free_bytes": free_bytes}
                cuda["torch_reserved_start_bytes"] = torch.cuda.memory_reserved(device)
                # 从本次工作流开始重新统计显存峰值。
                torch.cuda.reset_peak_memory_stats(device)
                counters_started = True
        except Exception:
            _warn("ComfyUI Rust observer could not read initial CUDA metrics")

        escaped_error = None
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        except BaseException as error:
            escaped_error = error
            raise
        finally:
            try:
                duration_ms = (time.perf_counter() - started) * 1000
                record = {"prompt_id": prompt_id, "duration_ms": duration_ms}
                record["status"] = _execution_status(self, prompt_id, escaped_error, model_management, torch)
                if cuda is not None:
                    record["cuda"] = cuda
                    if counters_started:
                        try:
                            peak_reserved = torch.cuda.max_memory_reserved(device)
                            cuda["torch_peak_reserved_bytes"] = peak_reserved
                            cuda["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                            cuda["torch_reserved_growth_bytes"] = max(0, peak_reserved - cuda["torch_reserved_start_bytes"])
                        except Exception:
                            _warn("ComfyUI Rust observer could not read CUDA peak metrics")
                    if budget_config is not None:
                        try:
                            assess_budget, expected, headroom = budget_config
                            fits, available, shortfall = assess_budget(cuda["driver_free_bytes"], expected, headroom)
                            cuda["budget"] = {
                                "fits": fits,
                                "available_bytes": available,
                                "shortfall_bytes": shortfall,
                                "expected_extra_bytes": expected,
                                "headroom_bytes": headroom,
                            }
                        except Exception:
                            _warn("ComfyUI Rust observer could not calculate the budget advisory")
                _logger.info("comfyui_rs_observation %s", json.dumps(record, separators=(",", ":")))
            except Exception:
                _warn("ComfyUI Rust observer could not record prompt metrics")

    observe._comfyui_rs_observer = True
    execution.PromptExecutor.execute = observe


if os.getenv("COMFYUI_RS_OBSERVE") == "1":
    try:
        _install()
    except Exception:
        _warn("ComfyUI Rust observer could not attach to the prompt executor")
