import math
import ctypes
import dataclasses
import os
import torch
from typing import NamedTuple

import comfy_aimdo.host_buffer
from comfy.quant_ops import QuantizedTensor, ck


_rust_gathered_byte_layout = None
if os.getenv("COMFYUI_RS_NATIVE_MODE", "rust") != "python":
    try:
        from comfyui_rs._native import gathered_byte_layout as _rust_gathered_byte_layout
    except ImportError:
        pass


class TensorFileSlice(NamedTuple):
    file_ref: object
    lock: object
    offset: int
    size: int


def write_tensor_file_slice(tensor, source, stream=None):
    """Write one complete tensor; missing or incompatible file backing is an error."""
    info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
    size = source.numel() * source.element_size()
    if (info is None or tensor.storage_offset() != 0 or not tensor.is_contiguous()
            or tensor.numel() * tensor.element_size() != info.size or not source.is_contiguous()
            or size != info.size):
        raise ValueError("Writable tensors require an exact file descriptor and a complete contiguous source")

    with info.lock:
        if source.device.type == "cuda":
            stream = torch.cuda.current_stream(source.device) if stream is None else stream
            comfy_aimdo.host_buffer.write_device_to_file(info.file_ref, info.offset, size,
                                                        stream.cuda_stream, source.data_ptr(), source.device.index)
        elif source.device.type == "cpu":
            comfy_aimdo.host_buffer.write_host_to_file(info.file_ref, info.offset, size, source.data_ptr())
        else:
            raise ValueError("Tensor file writes require a CPU or CUDA source")


def read_tensor_file_slice_into(tensor, destination, stream=None, destination2=None):

    if isinstance(tensor, QuantizedTensor):
        if not read_tensor_file_slice_into(tensor._qdata,
                                           destination._qdata if destination is not None else None, stream=stream,
                                           destination2=(destination2._qdata if destination2 is not None else None)):
            return False

        if destination is not None:
            dst_orig_dtype = destination._params.orig_dtype
            destination._params.copy_from(tensor._params, non_blocking=False)
            destination._params = dataclasses.replace(destination._params, orig_dtype=dst_orig_dtype)
        if destination2 is not None:
            dst_orig_dtype = destination2._params.orig_dtype
            destination2._params.copy_from(destination._params if destination is not None else tensor._params, non_blocking=True)
            destination2._params = dataclasses.replace(destination2._params, orig_dtype=dst_orig_dtype)
        return True

    info = getattr(tensor.untyped_storage(), "_comfy_tensor_file_slice", None)
    if info is None:
        return False

    if destination is not None and destination.device.type != "cpu" and destination2 is None:
        destination2 = destination
        destination = None

    file_obj = info.file_ref
    if (file_obj is None
            or (destination is None and destination2 is None)
            or (destination is not None and (destination.device.type != "cpu" or destination.numel() * destination.element_size() < info.size))
            or (destination2 is not None and (destination2.device.type == "cpu" or destination2.numel() * destination2.element_size() < info.size))
            or tensor.numel() * tensor.element_size() != info.size
            or tensor.storage_offset() != 0
            or not tensor.is_contiguous()):
        return False

    if info.size == 0:
        return True

    if destination is None:
        stream_ptr = getattr(stream, "cuda_stream", 0) if stream is not None else 0
        comfy_aimdo.host_buffer.read_file_to_device(file_obj, info.offset, info.size,
                                                    stream_ptr, destination2.data_ptr(),
                                                    destination2.device.index,
                                                    mark_cold=False)
        return True

    hostbuf = getattr(destination.untyped_storage(), "_comfy_hostbuf", None)
    if hostbuf is not None:
        stream_ptr = getattr(stream, "cuda_stream", 0) if stream is not None else 0
        device_ptr = destination2.data_ptr() if destination2 is not None else 0
        with info.lock:
            hostbuf.read_file_slice(file_obj, info.offset, info.size,
                                    offset=destination.data_ptr() - hostbuf.get_raw_address(),
                                    stream=stream_ptr,
                                    device_ptr=device_ptr,
                                    device=None if destination2 is None else destination2.device.index)
        return True

    if not hasattr(file_obj, "seek") or not hasattr(file_obj, "readinto"):
        return False

    buf_type = ctypes.c_ubyte * info.size
    view = memoryview(buf_type.from_address(destination.data_ptr()))

    try:
        with info.lock:
            file_obj.seek(info.offset)
            done = 0
            while done < info.size:
                try:
                    n = file_obj.readinto(view[done:])
                except OSError:
                    return False
                if n <= 0:
                    return False
                done += n
        return True
    finally:
        view.release()

class TensorGeometry(NamedTuple):
    shape: any
    dtype: torch.dtype

    def element_size(self):
        info = torch.finfo(self.dtype) if self.dtype.is_floating_point else torch.iinfo(self.dtype)
        return info.bits // 8

    def numel(self):
        """直接计算单个形状的元素数，省去跨语言调用。"""
        return math.prod(self.shape)

def tensors_to_geometries(tensors, dtype=None):
    geometries = []
    for t in tensors:
        if t is None or isinstance(t, QuantizedTensor):
            geometries.append(t)
            continue
        tdtype = t.dtype
        if hasattr(t, "_model_dtype"):
            tdtype = t._model_dtype
        if dtype is not None:
            tdtype = dtype
        geometries.append(TensorGeometry(shape=t.shape, dtype=tdtype))
    return geometries

_RUST_LAYOUT_MIN_TEMPLATES = 16
_TENSOR_LAYOUT_METADATA = frozenset({
    "dim", "dtype", "element_size", "is_nested", "is_quantized", "layout", "nbytes", "numel", "shape",
})


def _has_plain_tensor_metadata(tensor):
    """识别可提前读取元数据的普通 Torch 张量。"""
    return (
        type(tensor) in (torch.Tensor, torch.nn.Parameter)
        and _TENSOR_LAYOUT_METADATA.isdisjoint(tensor.__dict__)
        and tensor.layout is torch.strided and not tensor.is_nested and not tensor.is_quantized
    )


def _gathered_layout_entries(tensors):
    """将足够大的普通模板组转换为字节数或几何描述，保留空槽位。"""
    entries = []
    template_count = 0
    for tensor in tensors:
        if tensor is None:
            entries.append(0)
            continue
        if type(tensor) is TensorGeometry:
            shape = tensor.shape
            if type(shape) is torch.Size:
                shape = tuple(shape)
            if type(shape) is not tuple or type(tensor.dtype) is not torch.dtype:
                return None
            try:
                entry = (shape, tensor.element_size())
            except (TypeError, RuntimeError, ValueError):
                # 留到创建当前视图时处理，保留前面视图的异常顺序。
                return None
        elif _has_plain_tensor_metadata(tensor):
            try:
                entry = tensor.nbytes
            except (TypeError, RuntimeError, ValueError):
                return None
            if type(entry) is not int:
                return None
        else:
            return None
        entries.append(entry)
        template_count += 1
    return entries if template_count >= _RUST_LAYOUT_MIN_TEMPLATES else None


def vram_aligned_size(tensor):
    """在 Python 中计算各张量分别按 1024 字节对齐后的总大小。"""
    if isinstance(tensor, list):
        return sum([vram_aligned_size(t) for t in tensor])

    if isinstance(tensor, QuantizedTensor):
        inner_tensors, _ = tensor.__tensor_flatten__()
        return vram_aligned_size([ getattr(tensor, attr) for attr in inner_tensors ])

    if tensor is None:
        return 0

    size = tensor.numel() * tensor.element_size()
    alignment = 1024
    return (size + alignment - 1) // alignment * alignment


def _gathered_view(template, gathered, start, end):
    """按原始字节数检查范围并恢复张量形状，末尾填充不占视图范围。"""
    if end > gathered.numel():
        raise ValueError(f"Buffer too small: needs {end} bytes, but only has {gathered.numel()}. ")
    if getattr(torch, "__candle_torch__", False) and (gathered.storage_offset() + start) % template.element_size():
        raise RuntimeError(f"Buffer offset must be aligned to the size of {template.dtype}")
    return gathered[start:end].view(template.dtype).view(template.shape)


def interpret_gathered_like(tensors, gathered):
    """根据模板恢复连续字节缓冲区中的张量视图和空槽位。"""
    offset = 0
    dest_views = []

    if gathered.dim() != 1 or gathered.element_size() != 1:
        raise ValueError(f"Buffer must be 1D and single-byte (got {gathered.dim()}D {gathered.dtype})")

    if (
        _rust_gathered_byte_layout is not None
        and type(tensors) in (list, tuple)
        and len(tensors) >= _RUST_LAYOUT_MIN_TEMPLATES
        and not torch.compiler.is_compiling()
        and _has_plain_tensor_metadata(gathered)
    ):
        entries = _gathered_layout_entries(tensors)
        if entries is not None:
            layout = _rust_gathered_byte_layout(entries)
            if layout is not None:
                return [
                    None if tensor is None else _gathered_view(tensor, gathered, start, end)
                    for tensor, (start, end) in zip(tensors, layout)
                ]

    for tensor in tensors:

        if tensor is None:
            dest_views.append(None)
            continue

        if not isinstance(tensor, QuantizedTensor):
            size = tensor.numel() * tensor.element_size()
            dest_views.append(_gathered_view(tensor, gathered, offset, offset + size))
            offset += vram_aligned_size(tensor)
            continue

        inner_tensors, qt_ctx = tensor.__tensor_flatten__()
        templates = {attr: getattr(tensor, attr) for attr in inner_tensors}
        actuals = {}
        for attr, template in templates.items():
            size = template.numel() * template.element_size()
            actuals[attr] = _gathered_view(template, gathered, offset, offset + size)
            offset += vram_aligned_size(template)

        dest_views.append(QuantizedTensor.__tensor_unflatten__(actuals, qt_ctx, 0, 0))

    return dest_views

def cat_gathered(tensors, gathered):
    """Concatenate quantized tensors laid out in order in gathered into one, in place.

    Per-row fields are concatenated; every other field and parameter must match and is
    shared. The codes are moved down over the earlier tensors' metadata, which is saved
    first and rewritten after the last codes, so the result is laid out exactly like
    interpret_gathered_like would lay out one tensor of the summed rows. Returns None
    when the tensors cannot be concatenated."""
    inner_tensors, ctx = tensors[0].__tensor_flatten__()
    rows = [t._params.orig_shape[0] for t in tensors]
    orig_shape = (sum(rows), *tensors[0]._params.orig_shape[1:])
    for t in tensors[1:]:
        other_inner, other_ctx = t.__tensor_flatten__()
        other_ctx["non_tensor_fields"]["orig_shape"] = (rows[0], *t._params.orig_shape[1:])
        if (other_inner != inner_tensors or other_ctx["layout_cls"] != ctx["layout_cls"]
                or other_ctx["non_tensor_fields"] != ctx["non_tensor_fields"]):
            return None
    fields = {}
    for attr in inner_tensors[1:]:
        values = [getattr(t, attr) for t in tensors]
        if all(v.ndim > 0 and v.shape[0] == n and v.shape[1:] == values[0].shape[1:] and v.dtype == values[0].dtype for v, n in zip(values, rows)):
            fields[attr] = torch.cat(values)
        elif all(torch.equal(v, values[0]) for v in values[1:]):
            fields[attr] = values[0].clone()
        else:
            return None

    offset = 0
    for t in tensors:
        qdata = t._qdata
        if qdata.data_ptr() < gathered.data_ptr() + offset:
            raise RuntimeError("Tensors must sit in order within gathered")
        ck.memmove(gathered[offset:offset + qdata.nbytes], qdata.view(torch.uint8).view(-1))
        offset += qdata.nbytes
    qdata = tensors[0]._qdata
    actuals = {"_qdata": gathered[:offset].view(qdata.dtype).view(orig_shape[0], *qdata.shape[1:])}
    offset = vram_aligned_size(actuals["_qdata"])
    for attr, value in fields.items():
        actuals[attr] = gathered[offset:offset + value.nbytes].view(value.dtype).view(value.shape)
        actuals[attr].copy_(value)
        offset += vram_aligned_size(value)
    ctx["non_tensor_fields"]["orig_shape"] = orig_shape
    return QuantizedTensor.__tensor_unflatten__(actuals, ctx, 0, 0)

aimdo_enabled = False

extra_ram_release_callback = None
RAM_CACHE_HEADROOM = 0

def set_ram_cache_release_state(callback, headroom):
    global extra_ram_release_callback
    global RAM_CACHE_HEADROOM
    extra_ram_release_callback = callback
    RAM_CACHE_HEADROOM = max(0, int(headroom))

def extra_ram_release(target, free_active=False):
    if extra_ram_release_callback is None:
        return 0
    return extra_ram_release_callback(target, free_active=free_active)
