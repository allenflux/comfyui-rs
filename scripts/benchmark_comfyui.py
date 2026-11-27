"""用同一组 API 工作流比较原版和 Rust 混合版 ComfyUI。"""

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

if __package__:
    from . import model_phase_report as phase_report
else:
    import model_phase_report as phase_report


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
DOCKER_VARIANTS = ("original", "hybrid_rust", "hybrid_python")
IMAGE_VARIANTS = ("before", "after")
POD_VARIANTS = ("rust", "python")
DOCKER_PYTHON = "/app/.venv/bin/python"
MODEL_PHASE_PROBE = Path(__file__).with_name("model_phase_probe.py")


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
    prompt_id: str | None = None


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
    prompt_id = None

    def finish(status, http_status=None, poll_errors=0):
        """保留用于匹配探针的任务 ID，报告生成时丢弃该标识。"""
        return JobResult(job.workflow_index, job.sequence, status, time.perf_counter() - started,
                         http_status, poll_errors, prompt_id)

    try:
        accepted = request_json(base_url, "/prompt", timeout, {"prompt": job.prompt})
    except HttpStatusError as error:
        rejected = 400 <= error.status < 500 and error.status != 408
        return finish("rejected" if rejected else "submission_unknown", error.status)
    except TRANSPORT_ERRORS:
        return finish("submission_unknown")
    accepted_id = accepted.get("prompt_id") if isinstance(accepted, dict) else None
    if not isinstance(accepted_id, str) or not accepted_id:
        return finish("submission_unknown")
    prompt_id = accepted_id

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


def benchmark_variant(base_url, warming, measured, args, selector, *, prompt_ids=None):
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
    if prompt_ids is not None:
        prompt_ids.update((result.sequence, result.prompt_id) for result in jobs if result.prompt_id is not None)
    job_rows = [asdict(result) for result in jobs]
    for row in job_rows:
        del row["prompt_id"]
    return {
        "warmup": summarize(warmup, warmup_seconds),
        "measured": summarize(jobs, elapsed),
        "jobs": job_rows,
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


def command_output(command, timeout=30):
    """以参数列表执行宿主命令，并保留启动失败的诊断信息。"""
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{command[0]} {command[1]}：{error.stderr.strip()}") from error


def selected_gpu(selector):
    """把索引固定为物理卡 UUID，容器分配和显存采样都使用该 UUID。"""
    row = command_output([
        "nvidia-smi", f"--id={selector}", "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], timeout=3)
    uuid, name, total, driver = (item.strip() for item in row.split(","))
    if not re.fullmatch(r"GPU-[a-fA-F0-9-]+", uuid) or "\n" in row:
        raise ValueError("请选择一张完整的 NVIDIA GPU，使用索引或 GPU UUID")
    return {"uuid": uuid, "name": name, "total_mib": int(total), "driver_version": driver}


def wait_for_idle_gpu(gpu_uuid):
    """等待前一容器释放 GPU 上下文，有其它计算进程时停止比较。"""
    deadline = time.monotonic() + 15
    while True:
        active = command_output([
            "nvidia-smi", f"--id={gpu_uuid}", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], timeout=3)
        if active in {"", "No running processes found"}:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{gpu_uuid} 仍有计算进程，请使用空闲的 GPU")
        time.sleep(0.5)


def comfy_args_with_probe(arguments):
    """停用自定义节点时只保留指定节点和同一份计时探针。"""
    arguments = arguments.copy()
    if "--disable-all-custom-nodes" not in arguments:
        return arguments
    option = "--whitelist-custom-nodes"
    for index in range(len(arguments) - 1, -1, -1):
        argument = arguments[index]
        if argument == option:
            arguments.insert(index + 1, phase_report.PROBE_FILENAME)
            return arguments
        if argument.startswith(option + "="):
            arguments[index:index + 1] = [option, phase_report.PROBE_FILENAME, argument.partition("=")[2]]
            return arguments
    return [*arguments, option, phase_report.PROBE_FILENAME]


def docker_run_args(args, image, gpu_uuid, container, variant, *, probe_source=None):
    """各组共用 GPU、数据和启动参数，双镜像对照另外挂载同一份探针。"""
    original = variant == "original"
    profiled = variant in IMAGE_VARIANTS
    command = ["docker", "run", "--detach", "--name", container, "--pull", "never",
               "--gpus", f"device={gpu_uuid}", "--publish", "127.0.0.1::8188",
               "--entrypoint", DOCKER_PYTHON]
    for name in ("models", "custom_nodes", "input", "extra_model_paths.yaml"):
        source = args.data_dir / name
        if source.exists():
            command.extend(["--volume", f"{source}:/app/{name}:ro"])
    if original:
        command.extend(["--volume", f"{args.original_source}:/benchmark/original:ro"])
    else:
        mode = "python" if variant == "hybrid_python" else "rust"
        command.extend(["--env", f"COMFYUI_RS_NATIVE_MODE={mode}", "--env", "COMFYUI_RS_EVICTION_MODE=rust"])
    if profiled:
        probe = probe_source or MODEL_PHASE_PROBE.resolve()
        command.extend([
            "--volume", f"{probe}:/app/comfyui/custom_nodes/{phase_report.PROBE_FILENAME}:ro",
            "--env", f"COMFYUI_RS_MODEL_PHASE_PATH={phase_report.PROBE_PATH}",
        ])

    entry = "/benchmark/original/main.py" if original else "/app/main.py"
    command.extend([image, entry, "--listen", "0.0.0.0", "--port", "8188",
                    "--base-directory", "/app", "--disable-assets"])
    # 混合版从安装根目录自动加载 YAML，原版从挂载的源码目录启动时需显式指定。
    if original and (args.data_dir / "extra_model_paths.yaml").is_file():
        command.extend(["--extra-model-paths-config", "/app/extra_model_paths.yaml"])
    if not args.use_comfy_cache:
        command.append("--cache-none")
    command.extend(comfy_args_with_probe(args.comfy_arg) if profiled else args.comfy_arg)
    return command


def wait_for_docker_worker(container, timeout):
    """等待新实例启动并检查 CUDA 设备，避免把 CPU 回退算进 GPU 对照。"""
    address = command_output(["docker", "port", container, "8188/tcp"])
    if not re.fullmatch(r"127\.0\.0\.1:[0-9]+", address):
        raise RuntimeError(f"{container} 没有独立的本机 API 端口")
    url = f"http://{address}"
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            stats = request_json(url, "/system_stats", min(2, remaining))
        except (HttpStatusError, *TRANSPORT_ERRORS):
            state = command_output(["docker", "inspect", "--format", "{{.State.Status}}", container])
            if state != "running":
                raise RuntimeError(f"{container} 启动失败，容器状态为 {state}") from None
            time.sleep(0.5)
            continue
        devices = stats["devices"]
        if len(devices) != 1 or devices[0]["type"] != "cuda":
            raise RuntimeError(f"{container} 应使用一张 CUDA GPU")
        runtime = {key: stats["system"][key] for key in ("comfyui_version", "python_version", "pytorch_version")}
        return url, runtime
    raise RuntimeError(f"{container} 在 {timeout:g} 秒内未就绪")


def rss_peak(status):
    """从 Linux 进程状态中读取 worker 的常驻内存高水位。"""
    match = re.search(r"^VmHWM:[ \t]+([0-9]+)[ \t]+kB$", status, re.MULTILINE)
    return int(match[1]) / 1024 if match else None


def peak_worker_rss_mib(container):
    """读取容器 worker 的常驻内存峰值，读取失败时保留空值。"""
    try:
        status = command_output(["docker", "exec", container, "cat", "/proc/1/status"], timeout=5)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None
    return rss_peak(status)


def assert_probe_ready(container):
    """提交任务前验证双镜像都成功启用了同版本的探针。"""
    try:
        marker = command_output(["docker", "exec", container, "head", "-n", "1", phase_report.PROBE_PATH], timeout=5)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise RuntimeError("未找到模型计时探针的就绪记录，请检查自定义节点加载设置") from error
    if not phase_report.probe_ready(marker):
        raise RuntimeError("模型计时探针的就绪记录不兼容")


def read_model_phases(container, jobs, prompt_ids):
    """在容器清理前读取独立遥测文件，保留读取失败时的其它指标。"""
    try:
        raw = command_output(["docker", "exec", container, "cat", phase_report.PROBE_PATH])
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        result = phase_report.summarize_model_phases("", jobs, prompt_ids)
        result["read_error"] = str(error)
        return result
    return phase_report.summarize_model_phases(raw, jobs, prompt_ids)


def benchmark_docker(warming, measured, args, gpu, *, probe_source=None):
    """逐组运行固定镜像 ID 并清理容器，保留已完成组的测量。"""
    profiled = bool(args.before_image)
    images = ({"before": args.before_image, "after": args.after_image} if profiled
              else dict.fromkeys(DOCKER_VARIANTS, args.docker_image))
    image_ids = {tag: command_output(["docker", "image", "inspect", "--format", "{{.Id}}", tag])
                 for tag in dict.fromkeys(images.values())}
    metadata = {"gpu": gpu, "order": list(images), "comfy_args": args.comfy_arg, "data_dir": str(args.data_dir),
                "cache_none": not args.use_comfy_cache, "runtime": {}}
    if profiled:
        metadata["images"] = {variant: {"image": tag, "image_id": image_ids[tag]} for variant, tag in images.items()}
        probe_source = probe_source or MODEL_PHASE_PROBE.resolve()
        metadata["model_phase_probe_sha256"] = hashlib.sha256(probe_source.read_bytes()).hexdigest()
    else:
        metadata.update(image=args.docker_image, image_id=image_ids[args.docker_image],
                        original_source=str(args.original_source))
    reference_runtime = None
    variants = {}
    for variant, tag in images.items():
        try:
            wait_for_idle_gpu(gpu["uuid"])
            print(f"开始测量 {variant}，计划执行 {len(warming)} 次预热和 {len(measured)} 次测量。", file=sys.stderr)
            container = f"comfyui-bench-{uuid4().hex[:12]}-{variant.replace('_', '-')}"
            try:
                command = docker_run_args(args, image_ids[tag], gpu["uuid"], container, variant, probe_source=probe_source)
                command_output(command, timeout=args.startup_timeout)
                url, runtime = wait_for_docker_worker(container, args.startup_timeout)
                metadata["runtime"][variant] = runtime
                if reference_runtime is None:
                    reference_runtime = runtime
                if runtime != reference_runtime:
                    raise RuntimeError("ComfyUI、Python 或 PyTorch 的运行版本不一致")
                if profiled or variant == "hybrid_rust":
                    # 多处接入在导入失败时会回退，先确认此镜像的原生扩展可加载。
                    command_output(["docker", "exec", container, DOCKER_PYTHON, "-c",
                                    "from comfyui_rs._native import PromptScheduler"])
                if profiled:
                    assert_probe_ready(container)
                    prompt_ids = {}
                    result = benchmark_variant(url, warming, measured, args, gpu["uuid"], prompt_ids=prompt_ids)
                    result["model_phases"] = read_model_phases(container, result["jobs"], prompt_ids)
                else:
                    result = benchmark_variant(url, warming, measured, args, gpu["uuid"])
                result["worker_peak_rss_mib"] = peak_worker_rss_mib(container)
            finally:
                # 容器由本次脚本创建，未知终态也必须先清理，再开始下一组。
                try:
                    command_output(["docker", "rm", "--force", container])
                except RuntimeError as error:
                    if "No such container" not in str(error):
                        raise
            variants[variant] = result
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            metadata["error"] = f"{variant}：{error}"
            print(metadata["error"], file=sys.stderr)
            break
    return variants, metadata


def pod_worker_command(args, interpreter, directory, port, probe):
    """复用 Pod 数据目录，隔离本次 worker 的写入目录和探针。"""
    output, user, previews = (directory / name for name in ("output", "user", "previews"))
    for path in (output, user, previews):
        path.mkdir()
    command = [str(interpreter), str(args.app_dir / "comfyui" / "main.py"), "--disable-assets"]
    if not args.use_comfy_cache:
        command.append("--cache-none")
    command.extend(comfy_args_with_probe(args.comfy_arg) if probe is not None else args.comfy_arg)
    config = args.data_dir / "extra_model_paths.yaml"
    if args.data_dir != args.app_dir and config.is_file():
        command.extend(["--extra-model-paths-config", str(config)])
    if probe is not None:
        nodes = directory / "probe_nodes"
        nodes.mkdir()
        (nodes / phase_report.PROBE_FILENAME).write_bytes(probe)
        config = directory / "probe_paths.yaml"
        config.write_text(json.dumps({"comfyui_benchmark": {"custom_nodes": str(nodes)}}), encoding="utf-8")
        command.extend(["--extra-model-paths-config", str(config)])
    command.extend(["--listen", "127.0.0.1", "--port", str(port), "--base-directory", str(args.data_dir),
                    "--output-directory", str(output), "--user-directory", str(user),
                    "--previews-directory", str(previews), "--temp-directory", str(directory)])
    return command


def wait_for_pod_worker(worker, url, timeout, directory):
    """确认回环端口属于新 worker，并验证实际只加载一张 CUDA 卡。"""
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        if worker.poll() is not None:
            raise RuntimeError(f"Pod 测试 worker 提前退出（{worker.returncode}）")
        try:
            stats = request_json(url, "/system_stats", min(2, remaining))
        except (HttpStatusError, *TRANSPORT_ERRORS):
            time.sleep(0.25)
            continue
        if str(directory / "output") not in stats["system"].get("argv", ()):
            raise RuntimeError("回环端口由其它进程占用")
        if len(stats["devices"]) != 1 or stats["devices"][0]["type"] != "cuda":
            raise RuntimeError("Pod 测试 worker 应使用一张 CUDA GPU")
        return {key: stats["system"][key] for key in ("comfyui_version", "python_version", "pytorch_version")}
    raise RuntimeError(f"Pod 测试 worker 在 {timeout:g} 秒内未就绪")


def stop_pod_worker(worker):
    """只终止由脚本新建的进程组，等待 worker 退出。"""
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(worker.pid, signal.SIGKILL)
        worker.wait(timeout=5)


def run_pod_worker(warming, measured, args, gpu, interpreter, mode, probe):
    """在当前镜像启动单个测试 worker，读取阶段数据后清理进程。"""
    if not args.allow_shared_gpu:
        wait_for_idle_gpu(gpu["uuid"])
    with TemporaryDirectory(prefix="comfyui-pod-bench-") as folder:
        directory = Path(folder).resolve()
        report_path = directory / "model-phases.jsonl"
        env = os.environ.copy()
        env.pop("COMFYUI_RS_MODEL_PHASE_PATH", None)
        env.pop("COMFYUI_RS_OBSERVE", None)
        env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], COMFYUI_RS_NATIVE_MODE=mode,
                   COMFYUI_RS_EVICTION_MODE="rust", COMFYUI_RS_INSTALL_ROOT=str(args.app_dir))
        if probe is not None:
            env["COMFYUI_RS_MODEL_PHASE_PATH"] = str(report_path)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = pod_worker_command(args, interpreter, directory, port, probe)
        url = f"http://127.0.0.1:{port}"
        with (directory / "worker.log").open("wb") as log:
            worker = subprocess.Popen(command, cwd=args.app_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            try:
                runtime = wait_for_pod_worker(worker, url, args.startup_timeout, directory)
                if probe is not None:
                    try:
                        with report_path.open(encoding="utf-8") as probe_report:
                            ready = phase_report.probe_ready(probe_report.readline())
                    except (OSError, UnicodeError):
                        ready = False
                    if not ready:
                        raise RuntimeError("Pod 计时探针未就绪，请检查自定义节点加载设置")
                prompt_ids = {} if probe is not None else None
                result = benchmark_variant(url, warming, measured, args, gpu["uuid"], prompt_ids=prompt_ids)
                if probe is not None:
                    try:
                        raw = report_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeError) as error:
                        result["model_phases"] = phase_report.summarize_model_phases("", result["jobs"], prompt_ids)
                        result["model_phases"]["read_error"] = str(error)
                    else:
                        result["model_phases"] = phase_report.summarize_model_phases(raw, result["jobs"], prompt_ids)
                try:
                    status = Path(f"/proc/{worker.pid}/status").read_text(encoding="utf-8")
                    result["worker_peak_rss_mib"] = rss_peak(status)
                except OSError:
                    result["worker_peak_rss_mib"] = None
                return result, runtime
            finally:
                stop_pod_worker(worker)


def phase_coverage(report):
    """要求两个版本的同一份探针具有相同入口覆盖，且任务记录完整。"""
    phases = [variant["model_phases"] for variant in report["variants"].values()]
    same = len(phases) == 2 and phases[0]["probe"] is not None and phases[0]["probe"] == phases[1]["probe"]
    report["phase_probe_coverage_matches"] = same
    report["phase_metrics_complete"] = same and all(phase["complete"] for phase in phases)
    report["comparable"] = report["comparable"] and report["phase_metrics_complete"]


def summarize_pod_pairs(rounds):
    """逐轮计算 Rust 相对 Python 的变化，再汇总成配对中位数。"""
    selectors = {
        "success_per_second": lambda row: row["measured"]["success_per_second"],
        "success_p99_seconds": lambda row: row["measured"]["success_p99_seconds"],
        "oom_rate": lambda row: row["measured"]["oom_rate"],
        "gpu_peak_mib": lambda row: row["gpu"]["peak_mib"],
        "worker_peak_rss_mib": lambda row: row["worker_peak_rss_mib"],
    }
    summary = {}
    for name, select in selectors.items():
        samples = []
        for row in rounds:
            if row["comparable"]:
                rust, python = (select(row["variants"][mode]) for mode in POD_VARIANTS)
                if rust is not None and python is not None:
                    samples.append((rust, python))
        summary[name] = {
            "pairs": len(samples),
            "rust_median": statistics.median(pair[0] for pair in samples) if samples else None,
            "python_median": statistics.median(pair[1] for pair in samples) if samples else None,
            "rust_minus_python_median": statistics.median(rust - python for rust, python in samples) if samples else None,
        }
    return summary


def benchmark_pod(settings, warming, measured, args):
    """在一个 Pod 内交替两种模式，保留已完成的轮次与部分结果。"""
    gpu = selected_gpu(args.gpu)
    interpreter = args.app_dir / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    command_output([str(interpreter), "-c", "from comfyui_rs._native import PromptScheduler"])
    probe = None if args.no_phase_probe else MODEL_PHASE_PROBE.read_bytes()
    metadata = {"gpu": gpu, "app_dir": str(args.app_dir), "data_dir": str(args.data_dir),
                "python": str(interpreter), "comfy_args": args.comfy_arg, "cache_none": not args.use_comfy_cache,
                "allow_shared_gpu": args.allow_shared_gpu, "planned_rounds": args.rounds,
                "model_phase_probe_sha256": hashlib.sha256(probe).hexdigest() if probe is not None else None}
    rounds = []
    reference_runtime = None
    for index in range(args.rounds):
        order = POD_VARIANTS if index % 2 == 0 else POD_VARIANTS[::-1]
        variants = {}
        for mode in order:
            try:
                print(f"第 {index + 1}/{args.rounds} 轮，开始测量 {mode}。", file=sys.stderr)
                result, runtime = run_pod_worker(warming, measured, args, gpu, interpreter, mode, probe)
                if reference_runtime is None:
                    reference_runtime = runtime
                    metadata["runtime"] = runtime
                if runtime != reference_runtime:
                    raise RuntimeError("ComfyUI、Python 或 PyTorch 的运行版本不一致")
                variants[mode] = result
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
                metadata["error"] = f"第 {index + 1} 轮 {mode}：{error}"
                print(metadata["error"], file=sys.stderr)
                break
        report = make_report(settings, variants, separate_runs=True, same_gpu_uuid=gpu["uuid"],
                             expected_variants=POD_VARIANTS)
        if probe is not None:
            phase_coverage(report)
        rounds.append({"index": index + 1, "order": list(order), **report})
        complete = len(rounds) == args.rounds and all(row["comparable"] for row in rounds)
        snapshot = {"settings": settings, "pod": metadata, "rounds": rounds, "paired": summarize_pod_pairs(rounds),
                    "complete": complete, "comparable": complete and not args.allow_shared_gpu,
                    "isolated_gpu_requested": not args.allow_shared_gpu}
        # 长时间压测每轮保存一次，后续中断时可保留已经采完的对照。
        if args.output is not None:
            write_report_json(snapshot, args.output)
        if "error" in metadata:
            break
    return snapshot


def print_summary(report):
    """用终端表格显示主指标，详细失败与采样信息保留在 JSON 中。"""
    if "rounds" in report:
        if not report["isolated_gpu_requested"]:
            print("共享 GPU 探索模式有外部负载干扰，报告标记为不可正式对照。", file=sys.stderr)
        for row in report["rounds"]:
            print(f"第 {row['index']} 轮  {' → '.join(row['order'])}", file=sys.stderr)
            print_summary({**row, "pod": report["pod"]})
        print("配对指标                       轮数   Rust 中位数   Python 中位数   Rust−Python 中位数", file=sys.stderr)
        for name, summary in report["paired"].items():
            values = [summary[key] for key in ("rust_median", "python_median", "rust_minus_python_median")]
            columns = "   ".join(format(value, ".4f") if value is not None else "—" for value in values)
            print(f"{name:<30} {summary['pairs']:<6} {columns}", file=sys.stderr)
        return
    show_rss = "docker" in report or "pod" in report
    header = "版本            成功/计划   吞吐(成功/s)   P99(s)   显存峰值(MiB)   OOM/已知终态"
    if show_rss:
        header += "   worker RAM 峰值(MiB)"
    print(header, file=sys.stderr)
    for name, variant in report["variants"].items():
        metrics = variant["measured"]
        counts = metrics["status_counts"]
        throughput, p99 = metrics["success_per_second"], metrics["success_p99_seconds"]
        peak, known = variant["gpu"]["peak_mib"], metrics["known_terminal_count"]
        rate = f"{counts.get('oom', 0)}/{known} ({metrics['oom_rate']:.1%})" if known else "—"
        row = (f"{name:<15} {counts.get('success', 0)}/{len(variant['jobs']):<7} "
               f"{format(throughput, '.3f') if throughput is not None else '—':<14} "
               f"{format(p99, '.3f') if p99 is not None else '—':<8} "
               f"{str(peak) if peak is not None else '—':<15} {rate}")
        if show_rss:
            rss = variant["worker_peak_rss_mib"]
            row += f"   {format(rss, '.1f') if rss is not None else '—'}"
        print(row, file=sys.stderr)
    if (report["settings"]["workflows_without_mutable_seed"] or report["settings"]["keep_seeds"]) and not (
        report.get("docker", report.get("pod", {})).get("cache_none")
    ):
        print("部分工作流使用固定种子或无可改种子。若要测每次完整执行，各版本使用相同的 --cache-none。", file=sys.stderr)
    if report["shared_gpu_in_one_run"]:
        print("两个实例记录了同一 GPU。请分别启动并采集，再合并报告，避免模型驻留影响对照。", file=sys.stderr)
    if not report["gpu_metrics_complete"]:
        print("显存采样缺失，报告保留已采集的其它指标。", file=sys.stderr)
    if "phase_metrics_complete" in report:
        print("模型阶段             版本           任务数   中位数(ms)   P99(ms)", file=sys.stderr)
        for variant_name, variant in report["variants"].items():
            phases = variant["model_phases"]["host_phases"]
            cuda = variant["model_phases"]["weight_move_cuda"]
            stage_rows = [(name, stats, "host") for name, stats in phases.items()]
            stage_rows.append(("weight_move_cuda", cuda, "stream"))
            for name, stats, metric in stage_rows:
                middle, p99 = stats[f"median_{metric}_ms"], stats[f"p99_{metric}_ms"]
                print(f"{name:<20} {variant_name:<14} {stats['jobs']:<8} "
                      f"{format(middle, '.3f') if middle is not None else '—':<13} "
                      f"{format(p99, '.3f') if p99 is not None else '—'}", file=sys.stderr)
            if cuda["partial"]:
                print(f"{variant_name} 有未完成、超限或失败的 CUDA 事件，流计时仅覆盖已读取的事件。", file=sys.stderr)
        if not report["phase_metrics_complete"]:
            print("模型阶段记录不完整，JSON 保留已采集的任务和缺失计数。", file=sys.stderr)


def make_report(settings, variants, *, separate_runs=False, same_gpu_uuid=None, expected_variants=None):
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
    expected = {"original", "hybrid"}
    if same_gpu_uuid is not None:
        expected = set(expected_variants or DOCKER_VARIANTS)
        incomplete = incomplete or set(variants) != expected
        gpu_complete = gpu_complete and bool(gpu_ids) and all(uuid == same_gpu_uuid for uuid in gpu_ids)
    return {
        "settings": settings,
        "variants": variants,
        "comparable": set(variants) == expected and not incomplete and not shared_gpu and gpu_complete,
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


def write_report_json(report, output):
    """将结构化报告写入文件或标准输出。"""
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        output.write_text(content, encoding="utf-8")


def save_report(report, output):
    """保存完整 JSON，并在标准错误流显示方便阅读的对照。"""
    write_report_json(report, output)
    print_summary(report)


def main(argv=None):
    """支持 API、Docker 对照和在同一 Pod 中切换 Rust 与 Python。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-url", type=endpoint, help="原版 ComfyUI API 根地址")
    parser.add_argument("--hybrid-url", type=endpoint, help="Rust 混合版 ComfyUI API 根地址")
    docker = parser.add_argument_group("单卡 Docker 对照")
    docker.add_argument("--docker-image", help="本地构建的混合版 CUDA 镜像")
    docker.add_argument("--before-image", help="采集模型阶段的改前 CUDA 镜像")
    docker.add_argument("--after-image", help="采集模型阶段的改后 CUDA 镜像")
    docker.add_argument("--original-source", type=Path, help="同版本的原版 ComfyUI 源码目录")
    docker.add_argument("--data-dir", type=Path, help="各组共享的数据目录，默认为当前目录")
    docker.add_argument("--gpu", help="各组复用的物理 GPU 索引或 UUID，默认为 0")
    docker.add_argument("--startup-timeout", type=float, help="容器启动超时秒数，默认为 120")
    docker.add_argument("--use-comfy-cache", action="store_true", help="沿用 ComfyUI 的工作流缓存设置")
    docker.add_argument("--comfy-arg", action="append", default=[], help="传给各组的 ComfyUI 参数，每次指定一个值")
    pod = parser.add_argument_group("单 Pod Rust / Python 对照")
    pod.add_argument("--pod-native-compare", action="store_true", help="顺序启动当前镜像的 Rust 和 Python worker")
    pod.add_argument("--app-dir", type=Path, help="Pod 内混合版项目根目录，默认为 /app")
    pod.add_argument("--rounds", type=int, help="交替执行顺序的配对轮数，默认为 2")
    pod.add_argument("--no-phase-probe", action="store_true", help="在 Pod 对照中关闭分阶段探针")
    pod.add_argument("--allow-shared-gpu", action="store_true", help="允许 GPU 已有进程，结果仅供探索")
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
    profiled = bool(args.before_image or args.after_image)
    docker_mode = bool(args.docker_image) or profiled
    pod_mode = args.pod_native_compare
    if pod_mode and docker_mode:
        parser.error("Pod 和 Docker 压测模式不能同时使用")
    if not pod_mode and any((args.app_dir, args.rounds is not None, args.no_phase_probe, args.allow_shared_gpu)):
        parser.error("Pod 参数需与 --pod-native-compare 一起使用")
    if profiled and (not args.before_image or not args.after_image or args.docker_image or args.original_source):
        parser.error("双镜像阶段压测请同时指定 --before-image 和 --after-image")
    inputs = list(args.compare_reports or args.workflow or ())
    if profiled or (pod_mode and not args.no_phase_probe):
        inputs.append(MODEL_PHASE_PROBE)
    if args.output is not None and any(path.resolve() == args.output.resolve() for path in inputs):
        parser.error("输出文件不能覆盖输入")
    if pod_mode:
        if args.original_url or args.hybrid_url or args.original_source or args.compare_reports or args.no_gpu_metrics:
            parser.error("Pod 对照自行启动 worker，并采集同一卡的显存")
        args.app_dir = (args.app_dir or Path("/app")).resolve()
        if not (args.app_dir / "comfyui" / "main.py").is_file():
            parser.error("--app-dir 必须指向包含 comfyui/main.py 的混合版项目根目录")
        args.data_dir = (args.data_dir or args.app_dir).resolve()
        if not args.data_dir.is_dir():
            parser.error("--data-dir 必须是现有目录")
        if not args.no_phase_probe and not MODEL_PHASE_PROBE.is_file():
            parser.error("缺少 scripts/model_phase_probe.py")
        args.rounds = 2 if args.rounds is None else args.rounds
        if args.rounds <= 0:
            parser.error("--rounds 必须是正整数")
        owned = ("--base-directory", "--port", "--listen", "--output-directory", "--temp-directory", "--user-directory",
                 "--previews-directory", "--cuda-device", "--default-device", "--auto-launch", "--quick-test-for-ci")
        for argument in args.comfy_arg:
            option = argument.split("=", 1)[0]
            if option.startswith("--") and any(flag.startswith(option) for flag in owned):
                parser.error(f"Pod 对照会自行配置 {option}")
    elif docker_mode:
        if args.original_url or args.hybrid_url or args.compare_reports or args.no_gpu_metrics:
            parser.error("单卡 Docker 模式需要本地显存采样，并自行管理 API 实例")
        if profiled:
            if not MODEL_PHASE_PROBE.is_file():
                parser.error("缺少 scripts/model_phase_probe.py")
        else:
            if args.original_source is None or not (args.original_source / "main.py").is_file():
                parser.error("请通过 --original-source 指定包含 main.py 的原版 ComfyUI 源码目录")
            args.original_source = args.original_source.resolve()
            if (args.original_source / "extra_model_paths.yaml").is_file():
                parser.error("原版会自动加载源码目录内的 extra_model_paths.yaml，请使用干净的原版源码目录，并把共享配置放在 --data-dir 下")
        args.data_dir = (args.data_dir or Path.cwd()).resolve()
        if not args.data_dir.is_dir():
            parser.error("--data-dir 必须是现有目录")
        if profiled and any((args.data_dir / "custom_nodes" / name).exists() for name in (
            phase_report.PROBE_FILENAME, phase_report.PROBE_FILENAME + ".disabled",
        )):
            parser.error("数据目录中有同名的模型计时探针，请移开自定义节点目录中的同名文件")
    elif any((args.original_source, args.data_dir, args.gpu, args.startup_timeout is not None,
              args.use_comfy_cache, args.comfy_arg)):
        parser.error("单卡运行参数需与 Docker 镜像或 --pod-native-compare 一起使用")
    if pod_mode or docker_mode:
        args.gpu = args.gpu or "0"
        args.startup_timeout = 120.0 if args.startup_timeout is None else args.startup_timeout
        if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
            parser.error("--startup-timeout 必须是有限的正数")
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

    if not (args.original_url or args.hybrid_url or docker_mode or pod_mode):
        parser.error("请指定 ComfyUI API 地址、Docker 镜像或 --pod-native-compare")
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

    if pod_mode:
        try:
            report = benchmark_pod(settings, warming, measured, args)
            save_report(report, args.output)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            parser.exit(1, f"无法运行 Pod 对照：{error}\n")
        return 0 if report["complete"] else 1

    if docker_mode:
        try:
            with ExitStack() as stack:
                gpu = selected_gpu(args.gpu)
                probe = None
                if profiled:
                    directory = stack.enter_context(TemporaryDirectory(prefix="comfyui-phase-"))
                    probe = Path(directory).resolve() / phase_report.PROBE_FILENAME
                    # 整轮固定探针内容，宿主工作区后续编辑不会改变第二组测量。
                    probe.write_bytes(MODEL_PHASE_PROBE.read_bytes())
                    probe.chmod(0o644)
                variants, metadata = benchmark_docker(warming, measured, args, gpu, probe_source=probe)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            parser.exit(1, f"无法准备单卡对照：{error}\n")
        report = make_report(settings, variants, separate_runs=True, same_gpu_uuid=gpu["uuid"],
                             expected_variants=IMAGE_VARIANTS if profiled else DOCKER_VARIANTS)
        report["docker"] = metadata
        if profiled:
            phase_coverage(report)
        try:
            save_report(report, args.output)
        except OSError:
            parser.error("无法写入基准报告")
        return 0 if report["comparable"] else 1

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
