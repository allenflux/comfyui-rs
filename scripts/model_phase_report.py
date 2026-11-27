"""按任务匹配 GPU 模型探针记录，汇总成功任务的阶段耗时。"""

import json
import math
import statistics


PROBE_PATH = "/tmp/comfyui-rs-model-phases.jsonl"
PROBE_FILENAME = "_comfyui_rs_model_phase_probe.py"
HOST_PHASES = (
    "checkpoint_read", "model_setup", "model_request", "model_load", "load_prepare", "model_switch",
    "model_release", "vram_reclaim", "allocator_trim", "weight_move_host",
)
CUDA_COUNTERS = ("events_complete", "events_incomplete", "events_dropped", "events_failed")
EXECUTION_TERMINAL = {"success", "oom", "execution_error", "interrupted"}


def probe_ready(line):
    """确认容器确实加载了兼容版本的探针。"""
    try:
        marker = json.loads(line)
    except ValueError:
        return False
    return isinstance(marker, dict) and marker.get("schema") == 1 and marker.get("probe_ready") is True


def nonnegative_int(value):
    """排除布尔值、负数及不可用的计数。"""
    return type(value) is int and value >= 0


def valid_ms(value):
    """接受可用于计算分位数的有限非负耗时。"""
    return type(value) in (int, float) and 0 <= value < math.inf


def parse_probe_row(line):
    """校验一条任务记录，跳过格式异常的遥测数据。"""
    try:
        row = json.loads(line)
        phases, cuda = row["phases"], row["weight_move_cuda"]
        if not (row["schema"] == 1 and isinstance(row["prompt_id"], str) and row["prompt_id"]
                and valid_ms(row["workflow_host_ms"]) and nonnegative_int(row["probe_errors"])
                and isinstance(phases, dict) and isinstance(cuda, dict)):
            return None
        for phase in phases.values():
            if not nonnegative_int(phase["calls"]) or not valid_ms(phase["host_ms"]):
                return None
        if not all(nonnegative_int(cuda[key]) for key in CUDA_COUNTERS):
            return None
        if cuda["events_complete"]:
            if not valid_ms(cuda["stream_ms"]):
                return None
        elif cuda["stream_ms"] is not None:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return row


def latency_statistics(values, metric):
    """按成功且命中阶段的任务计算中位数和最近秩 P99。"""
    values = sorted(values)
    return {
        "jobs": len(values),
        f"median_{metric}_ms": statistics.median(values) if values else None,
        f"p99_{metric}_ms": values[math.ceil(0.99 * len(values)) - 1] if values else None,
    }


def summarize_model_phases(raw, jobs, prompt_ids):
    """只汇总本轮测量任务，记录缺失、重复和失败任务以便核对。"""
    lines = raw.splitlines()
    ready = bool(lines) and probe_ready(lines[0])
    rows, duplicates = {}, set()
    invalid = 0
    truncated = False
    for line in lines[1:] if ready else lines:
        if line == '{"schema":1,"probe_truncated":true}':
            truncated = True
            continue
        row = parse_probe_row(line)
        if row is None:
            invalid += 1
            continue
        prompt_id = row["prompt_id"]
        if prompt_id in rows:
            duplicates.add(prompt_id)
        rows[prompt_id] = row

    matched, missing = [], []
    for job in jobs:
        prompt_id = prompt_ids.get(job["sequence"])
        row = rows.get(prompt_id)
        if row is None or prompt_id in duplicates:
            if job["status"] in EXECUTION_TERMINAL:
                missing.append(job["sequence"])
            continue
        matched.append({**{key: job[key] for key in ("workflow_index", "sequence", "status")},
                        **{key: row[key] for key in ("workflow_host_ms", "phases", "weight_move_cuda", "probe_errors")}})

    successes = [row for row in matched if row["status"] == "success"]
    phases = {}
    for name in HOST_PHASES:
        samples = [sample for row in successes if (sample := row["phases"].get(name)) and sample["calls"]]
        phases[name] = {**latency_statistics([sample["host_ms"] for sample in samples], "host"),
                        "calls": sum(sample["calls"] for sample in samples)}
    cuda = [row["weight_move_cuda"] for row in successes]
    event_counts = {name: sum(row[name] for row in cuda) for name in CUDA_COUNTERS}
    probe_errors = sum(row["probe_errors"] for row in matched)
    return {
        "complete": ready and not truncated and not missing and invalid == 0 and not duplicates and probe_errors == 0,
        "probe": json.loads(lines[0]) if ready else None, "truncated": truncated,
        "measured_records": len(matched), "successful_records": len(successes),
        "missing_measured_sequences": missing, "invalid_records": invalid,
        "duplicate_records": len(duplicates), "probe_errors": probe_errors,
        "host_phases": phases,
        "weight_move_cuda": {
            **latency_statistics([row["stream_ms"] for row in cuda if row["stream_ms"] is not None], "stream"),
            **event_counts,
            "partial": any(event_counts[name] for name in CUDA_COUNTERS if name != "events_complete"),
        },
        "records": matched,
    }
