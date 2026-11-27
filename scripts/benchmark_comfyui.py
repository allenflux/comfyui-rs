"""用同一组 API 工作流衡量原版和 Rust 混合版 ComfyUI。"""

import argparse
import hashlib
import http.client
import json
import math
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


TERMINAL = {"success", "oom", "execution_error", "interrupted"}
OUTSTANDING_UNKNOWN = {"submission_unknown", "timed_out"}
INCOMPLETE = OUTSTANDING_UNKNOWN | {"history_unknown", "not_sent", "partial_validation", "rejected"}
TRANSPORT_ERRORS = (OSError, ValueError, http.client.HTTPException)
SEED_FIELDS = {
    "KSampler": "seed",
    "KSamplerAdvanced": "noise_seed",
    "SamplerCustom": "noise_seed",
    "RandomNoise": "noise_seed",
}


@dataclass(frozen=True)
class Job:
    workflow_index: int
    sequence: int
    prompt: dict


@dataclass(frozen=True)
class JobResult:
    workflow_index: int
    sequence: int
    status: str
    latency_seconds: float | None = None
    http_status: int | None = None
    poll_errors: int = 0


class HttpStatusError(Exception):
    def __init__(self, status):
        """只保留 HTTP 状态码，避免把响应正文或请求信息写进报告。"""
        self.status = status
        super().__init__(status)


def is_api_prompt(prompt):
    """区分 API 节点映射与浏览器里保存的画布工作流。"""
    return isinstance(prompt, dict) and bool(prompt) and all(
        isinstance(node, dict) and isinstance(node.get("class_type"), str)
        and isinstance(node.get("inputs"), dict)
        for node in prompt.values()
    )


def load_workflow(path):
    """读取原始 API 导出，也接受只有 prompt 字段被用于执行的请求封装。"""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if is_api_prompt(value):
        return value
    if isinstance(value, dict) and is_api_prompt(value.get("prompt")):
        return value["prompt"]
    raise ValueError("需要 ComfyUI API 格式的工作流")


def has_mutable_seed(prompt):
    """只处理内置采样节点的字面整数种子，保留自定义字段和连线。"""
    return any(
        type(node["inputs"].get(SEED_FIELDS.get(node["class_type"]))) is int
        for node in prompt.values()
    )


def seeded_prompt(prompt, rng):
    """复制需要改动的输入，并保持一份工作流内相同种子的一致性。"""
    result = prompt.copy()
    seeds = {}
    for node_id, node in prompt.items():
        key = SEED_FIELDS.get(node["class_type"])
        inputs = node["inputs"]
        seed = inputs.get(key)
        if type(seed) is int:
            if seed not in seeds:
                seeds[seed] = rng.getrandbits(32)
            result[node_id] = {**node, "inputs": {**inputs, key: seeds[seed]}}
    return result


def make_workload(workflows, repeats, warmups, seed, keep_seeds=False):
    """轮流提交各工作流，所有版本复用同一份预热和测量任务计划。"""
    jobs = []
    for phase_count, rng in ((warmups, random.Random(seed ^ 0xFFFFFFFF)), (repeats, random.Random(seed))):
        for _ in range(phase_count):
            for index, prompt in enumerate(workflows, start=1):
                payload = prompt if keep_seeds else seeded_prompt(prompt, rng)
                jobs.append(Job(index, len(jobs) + 1, payload))
    warmup_count = warmups * len(workflows)
    return jobs[:warmup_count], jobs[warmup_count:]


def request_json(base_url, route, timeout, payload=None):
    """在剩余时间内请求 ComfyUI，并让失败信息保持可分类。"""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(base_url.rstrip("/") + "/" + route.lstrip("/"), data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        error.close()
        raise HttpStatusError(error.code) from None


def classify_history(entry):
    """从服务端结构化终态识别成功、OOM、中断和普通执行错误。"""
    status = entry.get("status") if isinstance(entry, dict) else None
    if not isinstance(status, dict):
        return "history_unknown"
    if status.get("status_str") == "success" and status.get("completed", True) is True:
        return "success"
    messages = status.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, (tuple, list)) or len(message) != 2:
                continue
            event, details = message
            if event == "execution_interrupted":
                return "interrupted"
            if event == "execution_error" and isinstance(details, dict):
                error_type = str(details.get("exception_type", "")).rsplit(".", 1)[-1]
                error_message = str(details.get("exception_message", "")).lower()
                if error_type in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or any(
                    marker in error_message for marker in (
                        "this error means you ran out of memory on your gpu",
                        "cuda out of memory", "cuda error: out of memory", "cuda_error_out_of_memory",
                    )
                ):
                    return "oom"
                return "execution_error"
    return "execution_error" if status.get("status_str") == "error" else "history_unknown"


def run_job(base_url, job, timeout, poll_interval):
    """从提交开始计时，只查询已接收任务的历史，不重发含糊的提交。"""
    started = time.perf_counter()
    deadline = started + timeout

    def finish(status, http_status=None, poll_errors=0):
        """记录基准结果，丢弃服务端任务 ID 和正文。"""
        return JobResult(job.workflow_index, job.sequence, status, time.perf_counter() - started, http_status, poll_errors)

    try:
        accepted = request_json(base_url, "/prompt", timeout, {"prompt": job.prompt})
    except HttpStatusError as error:
        rejected = 400 <= error.status < 500 and error.status != 408
        return finish("rejected" if rejected else "submission_unknown", error.status)
    except TRANSPORT_ERRORS:
        return finish("submission_unknown")
    prompt_id = accepted.get("prompt_id") if isinstance(accepted, dict) else None
    if not isinstance(prompt_id, str) or not prompt_id:
        return finish("submission_unknown")

    route = "/history/" + quote(prompt_id, safe="")
    partial = bool(accepted.get("node_errors"))
    poll_errors = 0
    while (remaining := deadline - time.perf_counter()) > 0:
        try:
            history = request_json(base_url, route, remaining)
            if not isinstance(history, dict):
                poll_errors += 1
            elif prompt_id in history and time.perf_counter() < deadline:
                # 有节点校验失败时服务端仍可能排入部分任务，需要等待但不能计入完整工作流指标。
                status = "partial_validation" if partial else classify_history(history[prompt_id])
                return finish(status, poll_errors=poll_errors)
        except (HttpStatusError, *TRANSPORT_ERRORS):
            poll_errors += 1
        if (remaining := deadline - time.perf_counter()) > 0:
            time.sleep(min(poll_interval, remaining))
    return finish("timed_out", poll_errors=poll_errors)


def not_sent(jobs):
    """对尚未提交的工作保留计划序号，便于判断比较是否完整。"""
    return [JobResult(job.workflow_index, job.sequence, "not_sent") for job in jobs]


def run_phase(base_url, jobs, concurrency, timeout, poll_interval):
    """并发数限制未结束的请求，结果不明时停止发出后续任务。"""
    if not jobs:
        return [], 0.0
    stop = threading.Event()

    def execute(job):
        """其它任务已无法确认时，避免把不确定的在途任务排除出并发数。"""
        if stop.is_set():
            return JobResult(job.workflow_index, job.sequence, "not_sent")
        result = run_job(base_url, job, timeout, poll_interval)
        if result.status in OUTSTANDING_UNKNOWN:
            stop.set()
        return result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
        results = list(pool.map(execute, jobs))
    return results, time.perf_counter() - started


def summarize(results, elapsed_seconds):
    """用成功数计算吞吐和 P99，用已知执行终态计算 OOM 比例。"""
    counts = Counter(result.status for result in results)
    latencies = sorted(result.latency_seconds for result in results if result.status == "success")
    known = sum(counts[status] for status in TERMINAL)
    return {
        "elapsed_seconds": elapsed_seconds,
        "status_counts": dict(counts),
        "known_terminal_count": known,
        "success_per_second": counts["success"] / elapsed_seconds if elapsed_seconds else None,
        "success_p99_seconds": latencies[math.ceil(len(latencies) * 0.99) - 1] if latencies else None,
        "oom_rate": counts["oom"] / known if known else None,
    }


class GpuSampler:
    def __init__(self, selector, interval):
        """采样本机 NVIDIA 整卡显存，CPU 模式用空设备标识。"""
        self.selector = selector
        self.interval = interval
        self.executable = shutil.which("nvidia-smi") if selector is not None else None
        self.thread = None
        self.stop_event = threading.Event()
        self.uuid = None
        self.samples = []
        self.baseline = None
        self.errors = 0

    def _sample(self):
        """只统计同一 UUID 的有效采样，错误次数单独记录。"""
        try:
            response = subprocess.run([
                self.executable, f"--id={self.selector}", "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ], check=True, capture_output=True, text=True, timeout=3)
            uuid, memory = (item.strip() for item in response.stdout.strip().split(","))
            used_mib = int(memory)
            if not uuid.startswith("GPU-") or used_mib < 0 or self.uuid not in (None, uuid):
                raise ValueError
        except (OSError, ValueError, subprocess.SubprocessError):
            self.errors += 1
            return
        self.uuid = uuid
        self.samples.append(used_mib)

    def _run(self):
        """测量期间定时采样，等待时也可立即结束。"""
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        """在测量前先尽量采集已预热的基线。"""
        if self.executable is not None:
            self._sample()
            self.baseline = self.samples[0] if self.samples else None
            self.thread = threading.Thread(target=self._run)
            self.thread.start()
        elif self.selector is not None:
            self.errors += 1

    def stop(self):
        """保留测量期和终点的有效峰值，缺失数据返回空值。"""
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self._sample()
        return {
            "enabled": self.selector is not None,
            "device_uuid": self.uuid,
            "sample_count": len(self.samples),
            "sample_error_count": self.errors,
            "baseline_mib": self.baseline,
            "peak_mib": max(self.samples, default=None),
        }


def benchmark_variant(base_url, warming, measured, args, selector):
    """隔离预热和测量时间，避免把上一个阶段未确认的任务带入测量。"""
    sampler = GpuSampler(selector, args.gpu_sample_interval)
    if base_url is None:
        warmup, warmup_seconds = not_sent(warming), None
    else:
        warmup, warmup_seconds = run_phase(base_url, warming, args.concurrency, args.timeout, args.poll_interval)
    unknown = any(result.status in OUTSTANDING_UNKNOWN for result in warmup)
    if base_url is None or unknown:
        jobs, elapsed = not_sent(measured), None
        gpu = sampler.stop()
    else:
        sampler.start()
        try:
            jobs, elapsed = run_phase(base_url, measured, args.concurrency, args.timeout, args.poll_interval)
        finally:
            gpu = sampler.stop()
    return {
        "warmup": summarize(warmup, warmup_seconds),
        "measured": summarize(jobs, elapsed),
        "jobs": [asdict(result) for result in jobs],
        "gpu": gpu,
    }


def endpoint(value):
    """只接收常规 HTTP(S) 地址，避免参数里混入凭据或查询信息。"""
    try:
        url = urlsplit(value)
        if url.scheme in {"http", "https"} and url.hostname and not (
            url.username or url.password or url.query or url.fragment
        ):
            return value.rstrip("/")
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("请输入不含凭据、查询或片段的 HTTP(S) 地址")


def print_summary(report):
    """用终端表格显示主指标，详细失败与采样信息保留在 JSON 中。"""
    print("版本       成功/计划   吞吐(成功/s)   P99(s)   显存峰值(MiB)   OOM/已知终态", file=sys.stderr)
    for name, variant in report["variants"].items():
        metrics = variant["measured"]
        counts = metrics["status_counts"]
        throughput, p99 = metrics["success_per_second"], metrics["success_p99_seconds"]
        peak, known = variant["gpu"]["peak_mib"], metrics["known_terminal_count"]
        rate = f"{counts.get('oom', 0)}/{known} ({metrics['oom_rate']:.1%})" if known else "—"
        print(f"{name:<10} {counts.get('success', 0)}/{len(variant['jobs']):<7} "
              f"{format(throughput, '.3f') if throughput is not None else '—':<14} "
              f"{format(p99, '.3f') if p99 is not None else '—':<8} "
              f"{str(peak) if peak is not None else '—':<15} {rate}", file=sys.stderr)
    if report["settings"]["workflows_without_mutable_seed"] or report["settings"]["keep_seeds"]:
        print("部分工作流使用固定种子或无可改种子。若要测每次完整执行，两端使用相同的 --cache-none。", file=sys.stderr)
    if report["shared_gpu_in_one_run"]:
        print("两个实例记录了同一 GPU。请分别启动并采集，再合并报告，避免模型驻留影响对照。", file=sys.stderr)
    if not report["gpu_metrics_complete"]:
        print("显存采样缺失，报告保留已采集的其它指标。", file=sys.stderr)


def make_report(settings, variants, *, separate_runs=False):
    """标记任务、采样和设备隔离是否完整，同时保留各自的测量值。"""
    incomplete = any(
        INCOMPLETE.intersection(variant[phase]["status_counts"])
        for variant in variants.values() for phase in ("warmup", "measured")
    )
    gpu_ids = [variant["gpu"]["device_uuid"] for variant in variants.values()]
    shared_gpu = (
        not separate_runs and len(gpu_ids) == 2 and gpu_ids[0] is not None and gpu_ids[0] == gpu_ids[1]
    )
    gpu_complete = settings["no_gpu_metrics"] or all(
        variant["gpu"]["peak_mib"] is not None for variant in variants.values()
    )
    return {
        "settings": settings,
        "variants": variants,
        "comparable": len(variants) == 2 and not incomplete and not shared_gpu and gpu_complete,
        "incomplete": incomplete,
        "gpu_metrics_complete": gpu_complete,
        "shared_gpu_in_one_run": shared_gpu,
    }


def compare_reports(original_path, hybrid_path):
    """合并单独采集的两份报告，同一张卡可以按顺序复用。"""
    original, hybrid = [
        json.loads(path.read_text(encoding="utf-8")) for path in (original_path, hybrid_path)
    ]
    settings = original["settings"]
    if not settings.get("workload_fingerprint") or settings != hybrid["settings"]:
        raise ValueError("两份报告的工作流、种子或采集参数不一致")
    if set(original["variants"]) != {"original"} or set(hybrid["variants"]) != {"hybrid"}:
        raise ValueError("需要依次提供单独采集的原版和混合版报告")
    return make_report(settings, {**original["variants"], **hybrid["variants"]}, separate_runs=True)


def save_report(report, output):
    """保存完整 JSON，并在标准错误流显示方便阅读的对照。"""
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        output.write_text(content, encoding="utf-8")
    print_summary(report)


def main(argv=None):
    """生成可单独采集或成对比较的基准报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-url", type=endpoint, help="原版 ComfyUI API 根地址")
    parser.add_argument("--hybrid-url", type=endpoint, help="Rust 混合版 ComfyUI API 根地址")
    parser.add_argument("--workflow", type=Path, action="append", help="API 工作流 JSON，可重复指定")
    parser.add_argument("--compare-reports", type=Path, nargs=2, metavar=("ORIGINAL", "HYBRID"),
                        help="离线合并两份分别采集的 JSON 报告")
    parser.add_argument("--repeats", type=int, default=5, help="每个工作流的测量次数")
    parser.add_argument("--warmups", type=int, default=1, help="每个工作流的预热次数")
    parser.add_argument("--concurrency", type=int, default=1, help="同时提交并等待的任务上限")
    parser.add_argument("--seed", type=int, default=2711, help="任务计划的随机种子")
    parser.add_argument("--keep-seeds", action="store_true", help="沿用工作流中的原始种子")
    parser.add_argument("--timeout", type=float, default=600.0, help="每个任务的总超时秒数")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="历史状态轮询间隔秒数")
    parser.add_argument("--original-gpu", default="0", help="原版所用本机 GPU UUID 或索引")
    parser.add_argument("--hybrid-gpu", default="0", help="混合版所用本机 GPU UUID 或索引")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5, help="显存采样间隔秒数")
    parser.add_argument("--no-gpu-metrics", action="store_true", help="跳过 NVIDIA 显存采样")
    parser.add_argument("--output", type=Path, help="写入 JSON 文件，省略时输出到标准输出")
    args = parser.parse_args(argv)
    inputs = args.compare_reports or args.workflow or ()
    if args.output is not None and any(path.resolve() == args.output.resolve() for path in inputs):
        parser.error("输出文件不能覆盖输入")
    if args.compare_reports:
        if args.original_url or args.hybrid_url or args.workflow:
            parser.error("合并已有报告时只需要指定两份报告和可选的输出路径")
        try:
            report = compare_reports(*args.compare_reports)
        except (KeyError, TypeError, AttributeError, OSError, UnicodeError, json.JSONDecodeError):
            parser.error("无法读取两份有效的基准报告")
        except ValueError as error:
            parser.error(str(error))
        try:
            save_report(report, args.output)
        except OSError:
            parser.error("无法写入基准报告")
        return 0 if report["comparable"] else 1

    if not (args.original_url or args.hybrid_url):
        parser.error("至少指定一个 ComfyUI API 地址")
    if not args.workflow:
        parser.error("请指定至少一份 API 工作流")
    if args.warmups < 0 or args.repeats <= 0 or args.concurrency <= 0:
        parser.error("预热次数可为零，测量次数和并发数必须为正整数")
    if not all(math.isfinite(value) and value > 0 for value in (
        args.timeout, args.poll_interval, args.gpu_sample_interval,
    )):
        parser.error("超时和采样间隔必须是有限的正数")

    workflows = []
    for index, path in enumerate(args.workflow, start=1):
        try:
            workflows.append(load_workflow(path))
        except (OSError, ValueError):
            parser.error(f"第 {index} 个工作流无法读取或不是 API 格式")
    warming, measured = make_workload(workflows, args.repeats, args.warmups, args.seed, args.keep_seeds)
    settings = {key: getattr(args, key) for key in (
        "repeats", "warmups", "concurrency", "seed", "keep_seeds", "timeout", "poll_interval",
        "gpu_sample_interval", "no_gpu_metrics",
    )}
    settings["workflow_count"] = len(workflows)
    settings["workflows_without_mutable_seed"] = [
        index for index, workflow in enumerate(workflows, start=1) if not has_mutable_seed(workflow)
    ]
    digest = hashlib.sha256()
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for job in warming + measured:
        # 节点和输入的字典顺序会影响执行顺序，指纹使用实际提交的顺序。
        digest.update(json.dumps(job.prompt, separators=(",", ":")).encode("utf-8"))
    settings["workload_fingerprint"] = digest.hexdigest()

    variants = {}
    outstanding_unknown = False
    for name, url, selector in (
        ("original", args.original_url, args.original_gpu), ("hybrid", args.hybrid_url, args.hybrid_gpu),
    ):
        if url is None:
            continue
        result = benchmark_variant(
            None if outstanding_unknown else url, warming, measured, args,
            None if args.no_gpu_metrics else selector,
        )
        variants[name] = result
        outstanding_unknown = outstanding_unknown or any(
            OUTSTANDING_UNKNOWN.intersection(result[phase]["status_counts"])
            for phase in ("warmup", "measured")
        )
    report = make_report(settings, variants)
    try:
        save_report(report, args.output)
    except OSError:
        parser.error("无法写入基准报告")
    if outstanding_unknown:
        print("有任务交付或终态未确认，后续发送已暂停。请确认服务队列后再比较。", file=sys.stderr)
    return int(report["incomplete"] or report["shared_gpu_in_one_run"] or not report["gpu_metrics_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
