"""ComfyUI 的 Rust 显存管理扩展。"""

import logging
import os

from ._native import assess_budget as assess_budget, eviction_order as _eviction_order


_logger = logging.getLogger("comfyui_rs.eviction")
_mode = os.getenv("COMFYUI_RS_EVICTION_MODE", "rust")
if _mode not in ("python", "shadow", "rust"):
    _logger.warning("无效的 COMFYUI_RS_EVICTION_MODE，使用 Python 排序")
    _mode = "python"


def _sort_in_rust(candidates):
    """将 Rust 返回的下标映射成原候选对象，无法完成排序时返回 None。"""
    try:
        indices = _eviction_order(candidates)
        if indices is not None:
            # 只移动引用，排序后仍是调用方提供的元组对象。
            return [candidates[index] for index in indices]
    except Exception:
        _logger.warning("Rust 显存回收排序失败，使用 Python 排序")
    return None


def sort_eviction_candidates(candidates):
    """沿用 ComfyUI 的稳定排序语义，按配置使用 Python、Rust 或影子对照。"""
    if _mode == "python":
        return sorted(candidates)

    if _mode == "shadow":
        # 先运行 Python，保留原有比较异常。对照时按对象身份检查顺序。
        reference = sorted(candidates)
        rust_result = _sort_in_rust(candidates)
        if rust_result is not None and (
            len(reference) != len(rust_result)
            or any(left is not right for left, right in zip(reference, rust_result))
        ):
            _logger.warning("Rust 显存回收排序与 Python 结果不一致，使用 Python 排序")
        return reference

    rust_result = _sort_in_rust(candidates)
    return sorted(candidates) if rust_result is None else rust_result
