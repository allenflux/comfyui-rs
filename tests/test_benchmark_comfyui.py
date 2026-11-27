import copy
import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import benchmark_comfyui as benchmark


def workflow(seed, source="KSampler", seed_key="seed"):
    """构造含采样及普通输入的 API 格式工作流。"""
    return {
        "1": {"class_type": source, "inputs": {seed_key: seed}},
        "2": {"class_type": "CustomSeedTag", "inputs": {"seed": 7}},
        "3": {"class_type": "KSampler", "inputs": {"seed": ["2", 0]}},
    }


def history(status, exception_type=None, message=None):
    """按 ComfyUI 历史接口格式提供终态和错误事件。"""
    events = [["execution_start", {"timestamp": 1000}]]
    if status == "success":
        events.append(["execution_success", {"timestamp": 1020}])
    elif status == "interrupted":
        events.append(["execution_interrupted", {"timestamp": 1020}])
    else:
        events.append(["execution_error", {
            "timestamp": 1020,
            "exception_type": exception_type,
            "exception_message": message,
        }])
    return {"status": {
        "status_str": "success" if status == "success" else "error",
        "completed": status == "success",
        "messages": events,
    }}


class FakeComfyUI:
    """模拟两个独立 worker，使 HTTP 对照测试不依赖 GPU 或监听端口。"""

    def __init__(self):
        self.payloads = {"http://original.test": [], "http://hybrid.test": []}
        self.results = {}

    def request(self, base_url, route, timeout, payload=None):
        """按各 worker 的提交顺序生成独立任务及历史记录。"""
        if route == "/system_stats":
            return {"system": {"comfyui_version": "test"}, "devices": []}
        if route == "/queue":
            return {"queue_running": [], "queue_pending": []}
        if route == "/prompt" and payload is None:
            return {"exec_info": {"queue_remaining": 0}}
        if route == "/prompt":
            submissions = self.payloads[base_url]
            submissions.append(copy.deepcopy(payload))
            number = len(submissions)
            prompt_id = payload.get("prompt_id") or f"job-{number}"
            result = history("success")
            if base_url == "http://original.test" and number == 3:
                result = history("error", "torch.OutOfMemoryError", "CUDA out of memory")
            elif base_url == "http://original.test" and number == 4:
                result = history("error", "RuntimeError", "Model failed to load")
            self.results[(base_url, prompt_id)] = result
            return {"prompt_id": prompt_id, "number": number, "node_errors": {}}
        if route.startswith("/history/"):
            prompt_id = route.removeprefix("/history/")
            return {prompt_id: self.results[(base_url, prompt_id)]}
        raise AssertionError(f"Unexpected mocked request: {base_url} {route}")


class FakeDockerComfyUI(FakeComfyUI):
    """模拟宿主命令和单卡上的一个活动实例。"""

    gpu_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    image_id = "sha256:" + "a" * 64
    variants = ("original", "hybrid_rust", "hybrid_python")
    hosts = ("http://original.test", "http://hybrid.test", "http://hybrid-python.test")

    def __init__(self, cpu_fallback=None):
        """记录启动和清理事件，允许指定一组模拟 CPU 回退。"""
        super().__init__()
        self.payloads = {host: [] for host in self.hosts}
        self.cpu_fallback = cpu_fallback
        self.container = None
        self.commands = []
        self.launches = []
        self.removed = []

    def command(self, command, timeout=30):
        """模拟 Docker 和 NVIDIA 查询，并检查容器从不重叠。"""
        self.commands.append(command)
        if command[0] == "nvidia-smi":
            if "--query-compute-apps=pid" in command:
                assert self.container is None
                assert f"--id={self.gpu_uuid}" in command
                return ""
            return f"{self.gpu_uuid}, NVIDIA Test GPU, 32768, 595.58"
        if command[:3] == ["docker", "image", "inspect"]:
            return self.image_id
        if command[:2] == ["docker", "run"]:
            assert self.container is None
            self.container = command[command.index("--name") + 1]
            self.launches.append(command)
            return self.container
        if command[:2] == ["docker", "port"] and command[2] == self.container:
            return f"127.0.0.1:{21000 + len(self.launches)}"
        if command[:2] == ["docker", "exec"] and command[2] == self.container:
            if command[3:] == ["cat", "/proc/1/status"]:
                return f"Name:\tpython\nVmHWM:\t {len(self.launches) * 1048576} kB\nVmRSS:\t 1024 kB"
            return ""
        if command[:3] == ["docker", "rm", "--force"] and command[3] == self.container:
            self.removed.append(self.container)
            self.container = None
            return ""
        raise AssertionError(f"Unexpected mocked command: {command}")

    def request(self, base_url, route, timeout, payload=None):
        """从当前实例的端口返回设备、任务提交与历史。"""
        assert self.container is not None
        assert base_url == f"http://127.0.0.1:{21000 + len(self.launches)}"
        index = len(self.launches) - 1
        if route == "/system_stats":
            device_type = "cpu" if self.variants[index] == self.cpu_fallback else "cuda"
            return {"system": {"comfyui_version": "test", "python_version": "test",
                               "pytorch_version": "test"}, "devices": [{"type": device_type}]}
        return super().request(self.hosts[index], route, timeout, payload)


class BenchmarkTests(unittest.TestCase):
    def test_workload_repeats_in_same_order_with_reproducible_seeds(self):
        """两侧复用同一计划，重复任务变更采样种子且不修改输入文件内容。"""
        original = [workflow(99), workflow(101, "KSamplerAdvanced", "noise_seed")]
        source = copy.deepcopy(original)
        warmups, measured = benchmark.make_workload(original, repeats=3, warmups=1, seed=1234)
        self.assertEqual(original, source)
        self.assertEqual([job.workflow_index for job in warmups], [1, 2])
        self.assertEqual([job.workflow_index for job in measured], [1, 2, 1, 2, 1, 2])
        self.assertEqual([job.sequence for job in [*warmups, *measured]], list(range(1, 9)))
        self.assertEqual((warmups, measured), benchmark.make_workload(original, 3, 1, 1234))
        for workflow_index, field in enumerate(("seed", "noise_seed"), start=1):
            jobs = [job for job in [*warmups, *measured] if job.workflow_index == workflow_index]
            seeds = [job.prompt["1"]["inputs"][field] for job in jobs]
            self.assertEqual(len(set(seeds)), len(seeds))
            self.assertTrue(all(job.prompt["2"]["inputs"]["seed"] == 7 for job in jobs))
            self.assertTrue(all(job.prompt["3"]["inputs"]["seed"] == ["2", 0] for job in jobs))
        _, fixed = benchmark.make_workload(original, 2, 0, 1234, keep_seeds=True)
        self.assertEqual([job.prompt for job in fixed], original * 2)

    def test_terminal_classifier_separates_cuda_oom_and_other_failure(self):
        """只根据明确的 CUDA 或 PyTorch 错误证据统计显存不足。"""
        cases = (
            (history("success"), "success"),
            (history("error", "torch.OutOfMemoryError", "Allocation failed"), "oom"),
            (history("error", "RuntimeError", "CUDA error: out of memory"), "oom"),
            (history("error", "RuntimeError", "CUDA_ERROR_OUT_OF_MEMORY"), "oom"),
            (history("error", "MemoryError", "Host out of memory"), "execution_error"),
            (history("interrupted"), "interrupted"),
        )
        for result, expected in cases:
            with self.subTest(expected=expected, result=result):
                self.assertEqual(benchmark.classify_history(result), expected)

    def test_success_percentile_excludes_failed_and_unobserved_jobs(self):
        """失败与超时进入状态计数，成功延迟用各自的样本计算。"""
        results = [
            benchmark.JobResult(1, 1, "success", 0.5),
            benchmark.JobResult(1, 2, "success", 0.5),
            benchmark.JobResult(1, 3, "oom", 12.0),
            benchmark.JobResult(1, 4, "execution_error", 24.0),
            benchmark.JobResult(1, 5, "timed_out", 30.0),
        ]
        measured = benchmark.summarize(results, elapsed_seconds=20.0)
        self.assertEqual(measured["success_p99_seconds"], 0.5)
        self.assertEqual(measured["success_per_second"], 0.1)
        self.assertEqual(measured["status_counts"]["oom"], 1)
        self.assertEqual(measured["status_counts"]["execution_error"], 1)
        self.assertEqual(measured["status_counts"]["timed_out"], 1)
        self.assertEqual(measured["known_terminal_count"], 4)
        self.assertEqual(measured["oom_rate"], 0.25)
        self.assertIsNone(benchmark.summarize(results[2:], 20.0)["success_p99_seconds"])
        distinct = [benchmark.JobResult(1, index, "success", index) for index in range(1, 101)]
        self.assertEqual(benchmark.summarize(distinct, 6000)["success_p99_seconds"], 99)

    def test_submission_timeout_stays_unknown_without_repeating_post(self):
        """提交超时时，服务可能已收到任务，需要保留未知状态并避免重复调用。"""
        job = benchmark.Job(workflow_index=1, sequence=1, prompt=workflow(11))
        cases = (
            (TimeoutError(), "submission_unknown"),
            (benchmark.HttpStatusError(408), "submission_unknown"),
            (benchmark.HttpStatusError(429), "rejected"),
        )
        for error, expected in cases:
            with self.subTest(error=error):
                with patch.object(benchmark, "request_json", side_effect=error) as transport:
                    result = benchmark.run_job("http://original.test", job, timeout=1, poll_interval=0.1)
                self.assertEqual(result.status, expected)
                transport.assert_called_once()

    def test_partial_validation_waits_for_enqueued_job_without_counting_success(self):
        """服务接收部分有效节点后仍需等待终态，并排除完整工作流成功指标。"""
        job = benchmark.Job(1, 1, workflow(11))
        acknowledgements = [
            {"prompt_id": "partial", "node_errors": {"2": {"errors": [{"type": "invalid_input"}]}}},
            {"partial": history("success")},
        ]
        with patch.object(benchmark, "request_json", side_effect=acknowledgements) as transport:
            result = benchmark.run_job("http://original.test", job, timeout=1, poll_interval=0.1)
        self.assertEqual(result.status, "partial_validation")
        self.assertEqual([call.args[1] for call in transport.call_args_list], ["/prompt", "/history/partial"])
        summary = benchmark.summarize([result], 1.0)
        self.assertEqual(summary["known_terminal_count"], 0)
        self.assertIsNone(summary["success_p99_seconds"])
        self.assertIsNone(summary["oom_rate"])

    def test_missing_nvidia_smi_leaves_peak_and_baseline_unknown(self):
        """未找到采样命令时保留错误计数，峰值和基线不写成零。"""
        with patch.object(benchmark.shutil, "which", return_value=None):
            sampler = benchmark.GpuSampler("0", 0.1)
            sampler.start()
        report = sampler.stop()
        self.assertTrue(report["enabled"])
        self.assertEqual(report["sample_count"], 0)
        self.assertEqual(report["sample_error_count"], 1)
        self.assertIsNone(report["device_uuid"])
        self.assertIsNone(report["baseline_mib"])
        self.assertIsNone(report["peak_mib"])

    def test_gpu_peak_uses_only_valid_samples_from_selected_uuid(self):
        """基线、峰值和终点以同一张 GPU 的采样为准。"""
        rows = ["GPU-primary, 1024\n", "GPU-primary, 4096\n", "GPU-other, 8192\n", "GPU-primary, 2048\n"]
        with (
            patch.object(benchmark.shutil, "which", return_value="/test/nvidia-smi"),
            patch.object(benchmark.threading, "Thread"),
            patch.object(benchmark.subprocess, "run", side_effect=[SimpleNamespace(stdout=row) for row in rows]),
        ):
            sampler = benchmark.GpuSampler("GPU-primary", 0.1)
            sampler.start()
            sampler._sample()
            sampler._sample()
            report = sampler.stop()
        self.assertEqual(report["device_uuid"], "GPU-primary")
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["sample_error_count"], 1)
        self.assertEqual(report["baseline_mib"], 1024)
        self.assertEqual(report["peak_mib"], 4096)

    def test_cli_compares_identical_workloads_and_preserves_missing_gpu_samples(self):
        """通过命令行生成对照报告，核对两侧任务顺序和未采样显存。"""
        fake = FakeComfyUI()
        with TemporaryDirectory() as directory:
            work = Path(directory)
            first = work / "first.json"
            second = work / "second.json"
            output = work / "report.json"
            first.write_text(json.dumps(workflow(11)), encoding="utf-8")
            second.write_text(json.dumps({"prompt": workflow(22, "RandomNoise", "noise_seed")}), encoding="utf-8")
            argv = [
                "--original-url", "http://original.test", "--hybrid-url", "http://hybrid.test",
                "--workflow", str(first), "--workflow", str(second), "--warmups", "1", "--repeats", "2",
                "--concurrency", "1", "--seed", "17", "--no-gpu-metrics", "--output", str(output),
            ]
            with (
                patch.object(benchmark, "request_json", side_effect=fake.request),
                redirect_stderr(io.StringIO()),
            ):
                result = benchmark.main(argv)
            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))

        original, hybrid = fake.payloads.values()
        self.assertEqual(len(original), 6)
        # 请求标识允许各自生成，工作流、输入和提交顺序保持一致。
        self.assertEqual([entry["prompt"] for entry in original], [entry["prompt"] for entry in hybrid])
        self.assertTrue(report["comparable"])
        variants = report["variants"]
        self.assertEqual(variants["original"]["measured"]["status_counts"]["oom"], 1)
        self.assertEqual(variants["original"]["measured"]["status_counts"]["execution_error"], 1)
        self.assertEqual(variants["hybrid"]["measured"]["status_counts"]["success"], 4)
        for mode in variants.values():
            self.assertEqual(mode["gpu"]["sample_count"], 0)
            self.assertIsNone(mode["gpu"]["baseline_mib"])
            self.assertIsNone(mode["gpu"]["peak_mib"])

    def test_offline_comparison_accepts_matching_separate_reports_only(self):
        """分别采集的同卡报告可离线合并，工作负载指纹必须相同。"""
        fake = FakeComfyUI()
        gpu = {
            "enabled": True, "device_uuid": "GPU-one-card", "sample_count": 3,
            "sample_error_count": 0, "baseline_mib": 1024, "peak_mib": 4096,
        }
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            work = Path(directory)
            workload = work / "workflow.json"
            original_path, hybrid_path, merged_path = [work / f"{name}.json" for name in ("original", "hybrid", "merged")]
            workload.write_text(json.dumps(workflow(11)), encoding="utf-8")

            with (
                patch.object(benchmark, "request_json", side_effect=fake.request),
                patch.object(benchmark.GpuSampler, "start"),
                patch.object(benchmark.GpuSampler, "stop", return_value=gpu),
            ):
                for mode, path in (("original", original_path), ("hybrid", hybrid_path)):
                    result = benchmark.main([
                        f"--{mode}-url", f"http://{mode}.test", "--workflow", str(workload),
                        "--warmups", "0", "--repeats", "1", "--seed", "17", "--output", str(path),
                    ])
                    self.assertEqual(result, 0)

            original = json.loads(original_path.read_text(encoding="utf-8"))
            hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
            fingerprint = original["settings"]["workload_fingerprint"]
            self.assertTrue(fingerprint)
            self.assertEqual(fingerprint, hybrid["settings"]["workload_fingerprint"])

            args = ["--compare-reports", str(original_path), str(hybrid_path), "--output", str(merged_path)]
            with (
                patch.object(benchmark, "request_json", side_effect=AssertionError("Unexpected HTTP request")) as transport,
                patch.object(benchmark.GpuSampler, "start", side_effect=AssertionError("Unexpected GPU sample")) as sampler,
            ):
                self.assertEqual(benchmark.main(args), 0)
                merged_text = merged_path.read_text(encoding="utf-8")
                merged = json.loads(merged_text)
                self.assertTrue(merged["comparable"])
                self.assertFalse(merged["shared_gpu_in_one_run"])
                self.assertTrue(merged["gpu_metrics_complete"])
                self.assertEqual(merged["settings"]["workload_fingerprint"], fingerprint)
                self.assertEqual(merged["variants"], {**original["variants"], **hybrid["variants"]})

                hybrid["settings"]["workload_fingerprint"] = "different"
                hybrid_path.write_text(json.dumps(hybrid), encoding="utf-8")
                with self.assertRaises(SystemExit) as rejected:
                    benchmark.main(args)
                self.assertEqual(rejected.exception.code, 2)
                self.assertEqual(merged_path.read_text(encoding="utf-8"), merged_text)
                transport.assert_not_called()
                sampler.assert_not_called()


class DockerBenchmarkTests(unittest.TestCase):
    def run_benchmark(self, fake):
        """用两个工作流完整执行 CLI，拦截 Docker、HTTP 和显存采样。"""
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            work = Path(directory).resolve()
            original, data = work / "original", work / "data"
            original.mkdir()
            (original / "main.py").touch()
            (data / "models").mkdir(parents=True)
            (data / "extra_model_paths.yaml").write_text("{}", encoding="utf-8")
            first, second, output = [work / f"{name}.json" for name in ("a", "b", "report")]
            first.write_text(json.dumps(workflow(11)), encoding="utf-8")
            second.write_text(json.dumps(workflow(22, "RandomNoise", "noise_seed")), encoding="utf-8")
            gpu = {"enabled": True, "device_uuid": fake.gpu_uuid, "sample_count": 3,
                   "sample_error_count": 0, "baseline_mib": 1024, "peak_mib": 4096}
            with (
                patch.object(benchmark, "command_output", side_effect=fake.command),
                patch.object(benchmark, "request_json", side_effect=fake.request),
                patch.object(benchmark, "GpuSampler") as sampler,
            ):
                sampler.return_value.stop.return_value = gpu
                code = benchmark.main([
                    "--docker-image", "test:cuda", "--original-source", str(original),
                    "--data-dir", str(data), "--gpu", "2", "--workflow", str(first),
                    "--workflow", str(second), "--output", str(output), "--repeats", "2",
                    "--comfy-arg=--reserve-vram", "--comfy-arg=4",
                ])
                selectors = [call.args[0] for call in sampler.call_args_list]
            return code, json.loads(output.read_text(encoding="utf-8")), selectors

    def test_cli_runs_three_disjoint_containers_on_same_card_with_identical_prompts(self):
        """原版和两种混合版复用工作负载，按顺序独占同一物理卡。"""
        fake = FakeDockerComfyUI()
        code, report, selectors = self.run_benchmark(fake)
        self.assertEqual(code, 0)
        self.assertTrue(report["comparable"])
        self.assertNotIn("phase_metrics_complete", report)
        for result in report["variants"].values():
            self.assertNotIn("model_phases", result)
            self.assertTrue(all("prompt_id" not in job for job in result["jobs"]))
        self.assertEqual(tuple(report["variants"]), fake.variants)
        self.assertEqual(selectors, [fake.gpu_uuid] * 3)
        self.assertEqual(len(fake.launches), 3)
        self.assertEqual(fake.removed, [command[command.index("--name") + 1] for command in fake.launches])
        self.assertIsNone(fake.container)
        self.assertEqual(report["variants"]["original"]["measured"]["oom_rate"], 0.25)
        self.assertEqual([result["gpu"]["peak_mib"] for result in report["variants"].values()], [4096] * 3)
        self.assertEqual([result["worker_peak_rss_mib"] for result in report["variants"].values()], [1024, 2048, 3072])
        prompts = [[payload["prompt"] for payload in batch] for batch in fake.payloads.values()]
        self.assertEqual([len(batch) for batch in prompts], [6] * 3)
        self.assertEqual(prompts, [prompts[0]] * 3)
        original, rust, python = fake.launches
        self.assertIn("/benchmark/original/main.py", original)
        self.assertIn("/app/extra_model_paths.yaml", original)
        self.assertIn("COMFYUI_RS_NATIVE_MODE=rust", rust)
        self.assertIn("COMFYUI_RS_NATIVE_MODE=python", python)
        checks = [command for command in fake.commands if command[:2] == ["docker", "exec"] and command[3] == benchmark.DOCKER_PYTHON]
        self.assertEqual([command[2] for command in checks], [rust[rust.index("--name") + 1]])
        for command in fake.launches:
            self.assertIn(fake.image_id, command)
            self.assertIn(f"device={fake.gpu_uuid}", command)
            self.assertIn(f"{report['docker']['data_dir']}/models:/app/models:ro", command)
            self.assertIn("--cache-none", command)
            self.assertEqual(command[-2:], ["--reserve-vram", "4"])

    def test_cli_cleans_up_failed_start_and_writes_partial_report(self):
        """设备配置失败后清理新容器，并保留已完成的原版测量。"""
        fake = FakeDockerComfyUI(cpu_fallback="hybrid_rust")
        code, report, selectors = self.run_benchmark(fake)
        self.assertEqual(code, 1)
        self.assertFalse(report["comparable"])
        self.assertTrue(report["incomplete"])
        self.assertEqual(tuple(report["variants"]), ("original",))
        self.assertEqual(selectors, [fake.gpu_uuid])
        self.assertEqual(len(fake.launches), 2)
        self.assertEqual(fake.removed, [command[command.index("--name") + 1] for command in fake.launches])
        self.assertIsNone(fake.container)
        self.assertIn("hybrid_rust", report["docker"]["error"])


class FakeModelPhaseDocker(FakeDockerComfyUI):
    """模拟两份镜像各自的探针就绪记录、预热和测量任务。"""

    variants = ("before", "after")
    hosts = ("http://before.test", "http://after.test")
    image_ids = {"bench:before": "sha256:" + "b" * 64, "bench:after": "sha256:" + "c" * 64}
    marker = json.dumps({"schema": 1, "probe_ready": True})

    def __init__(self, drop_record=None):
        super().__init__()
        self.drop_record = drop_record
        self.ready_containers = set()

    def command(self, command, timeout=30):
        """在任务前读取就绪记录，在任务后以倒序回传探针记录。"""
        if command[:3] == ["docker", "image", "inspect"]:
            self.commands.append(command)
            return self.image_ids[command[-1]]
        if command[:2] == ["docker", "exec"] and command[-1] == benchmark.phase_report.PROBE_PATH:
            self.commands.append(command)
            assert command[2] == self.container
            index = len(self.launches) - 1
            if command[3:] == ["head", "-n", "1", benchmark.phase_report.PROBE_PATH]:
                assert not self.payloads[self.hosts[index]]
                self.ready_containers.add(self.container)
                return self.marker
            assert command[3:] == ["cat", benchmark.phase_report.PROBE_PATH]
            lines = [self.marker]
            for number in range(len(self.payloads[self.hosts[index]]), 0, -1):
                if (self.variants[index], number) == self.drop_record:
                    continue
                host_ms = 999 if number <= 2 else index * 10 + number
                lines.append(json.dumps({
                    "schema": 1, "prompt_id": f"server-{self.variants[index]}-{number}",
                    "workflow_host_ms": host_ms + 1, "probe_errors": 0,
                    "phases": {"model_load": {"calls": 2, "host_ms": host_ms}},
                    "weight_move_cuda": {"stream_ms": host_ms,
                                         "events_complete": 1, "events_incomplete": 0,
                                         "events_dropped": 0, "events_failed": 0},
                }))
            return "\n".join(lines)
        return super().command(command, timeout)

    def request(self, base_url, route, timeout, payload=None):
        """模拟镜像返回独立的任务 ID，探针就绪后才接受提交。"""
        if route == "/prompt" and payload is not None:
            assert self.container in self.ready_containers
            index = len(self.launches) - 1
            number = len(self.payloads[self.hosts[index]]) + 1
            payload = {**payload, "prompt_id": f"server-{self.variants[index]}-{number}"}
        return super().request(base_url, route, timeout, payload)


class ModelPhaseDockerTests(unittest.TestCase):
    def run_benchmark(self, fake):
        """执行双镜像 CLI 并读取完整报告。"""
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            data = Path(directory).resolve()
            (data / "models").mkdir()
            first, second, output = [data / f"{name}.json" for name in ("a", "b", "report")]
            first.write_text(json.dumps(workflow(11)), encoding="utf-8")
            second.write_text(json.dumps(workflow(22, "RandomNoise", "noise_seed")), encoding="utf-8")
            probe = data / "probe.py"
            probe.write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
            gpu = {"enabled": True, "device_uuid": fake.gpu_uuid, "sample_count": 3,
                   "sample_error_count": 0, "baseline_mib": 1024, "peak_mib": 4096}
            with (
                patch.object(benchmark, "command_output", side_effect=fake.command),
                patch.object(benchmark, "request_json", side_effect=fake.request),
                patch.object(benchmark, "MODEL_PHASE_PROBE", probe),
                patch.object(benchmark, "GpuSampler") as sampler,
            ):
                sampler.return_value.stop.return_value = gpu
                code = benchmark.main([
                    "--before-image", "bench:before", "--after-image", "bench:after",
                    "--data-dir", str(data), "--gpu", "2", "--workflow", str(first),
                    "--workflow", str(second), "--output", str(output), "--repeats", "2",
                    "--comfy-arg=--disable-all-custom-nodes", "--comfy-arg=--whitelist-custom-nodes=fixture.py",
                ])
                selectors = [call.args[0] for call in sampler.call_args_list]
            return code, json.loads(output.read_text(encoding="utf-8")), selectors

    def test_dual_images_profile_identical_workloads_on_same_gpu(self):
        """同一张卡依次运行两份镜像，按任务匹配阶段，排除预热。"""
        fake = FakeModelPhaseDocker()
        code, report, selectors = self.run_benchmark(fake)
        self.assertEqual(code, 0)
        self.assertTrue(report["comparable"])
        self.assertTrue(report["phase_metrics_complete"])
        self.assertEqual(tuple(report["variants"]), fake.variants)
        self.assertEqual(selectors, [fake.gpu_uuid] * 2)
        self.assertIsNone(fake.container)
        self.assertEqual(fake.removed, [command[command.index("--name") + 1] for command in fake.launches])
        prompts = [[payload["prompt"] for payload in batch] for batch in fake.payloads.values()]
        self.assertEqual([len(batch) for batch in prompts], [6, 6])
        self.assertEqual(prompts[0], prompts[1])
        probe_target = f":/app/comfyui/custom_nodes/{benchmark.phase_report.PROBE_FILENAME}:ro"
        mounts = [next(value for value in command if value.endswith(probe_target)) for command in fake.launches]
        self.assertEqual(mounts[0], mounts[1])
        self.assertFalse(Path(mounts[0].removesuffix(probe_target)).exists())
        for index, (variant, command) in enumerate(zip(fake.variants, fake.launches)):
            self.assertIn(fake.image_ids[f"bench:{variant}"], command)
            self.assertIn(f"device={fake.gpu_uuid}", command)
            self.assertIn(f"COMFYUI_RS_MODEL_PHASE_PATH={benchmark.phase_report.PROBE_PATH}", command)
            whitelist = command.index("--whitelist-custom-nodes")
            self.assertEqual(command[whitelist + 1:], [benchmark.phase_report.PROBE_FILENAME, "fixture.py"])
            variant_data = report["variants"][variant]
            phases = variant_data["model_phases"]
            self.assertEqual(phases["measured_records"], 4)
            self.assertEqual([row["sequence"] for row in phases["records"]], [3, 4, 5, 6])
            self.assertEqual(phases["host_phases"]["model_load"], {
                "jobs": 4, "calls": 8, "median_host_ms": 4.5 + 10 * index,
                "p99_host_ms": 6 + 10 * index,
            })
            self.assertEqual(phases["weight_move_cuda"]["events_complete"], 4)
            self.assertNotIn("prompt_id", variant_data["jobs"][0])
            self.assertNotIn("prompt_id", phases["records"][0])

    def test_missing_measurement_marks_phase_comparison_incomplete_and_cleans_up(self):
        """测量任务缺失探针数据时，保留可用指标并返回失败。"""
        fake = FakeModelPhaseDocker(drop_record=("after", 4))
        code, report, _selectors = self.run_benchmark(fake)
        self.assertEqual(code, 1)
        self.assertFalse(report["comparable"])
        self.assertFalse(report["phase_metrics_complete"])
        self.assertIsNone(fake.container)
        self.assertEqual(len(fake.removed), 2)
        before, after = report["variants"]["before"], report["variants"]["after"]
        self.assertTrue(before["model_phases"]["complete"])
        self.assertEqual(after["model_phases"]["missing_measured_sequences"], [4])
        self.assertEqual(after["model_phases"]["measured_records"], 3)
        self.assertEqual(after["measured"]["status_counts"], {"success": 4})


class FakePodComfyUI(FakeComfyUI):
    """模拟 Pod 自己创建的进程和回环 API，并记录启动与清理顺序。"""

    gpu_uuid = FakeDockerComfyUI.gpu_uuid
    port = 21000

    def __init__(self, fail_start=None):
        super().__init__()
        self.fail_start = fail_start
        self.launches = []
        self.stopped = []
        self.process = None
        self.probe_bytes = []

    def command(self, command, timeout=30):
        """模拟物理卡信息、独占检查和原生模块预检。"""
        if command[0] == "nvidia-smi":
            if "--query-compute-apps=pid" in command:
                assert self.process is None
                assert f"--id={self.gpu_uuid}" in command
                return ""
            assert "--id=2" in command
            return f"{self.gpu_uuid}, NVIDIA Test GPU, 32768, 595.58"
        assert command[1:] == ["-c", "from comfyui_rs._native import PromptScheduler"]
        return ""

    def launch(self, command, *, cwd, env, start_new_session, **_options):
        """模拟新 worker 及其临时探针，前一个必须已完成清理。"""
        assert self.process is None and start_new_session
        index = len(self.launches) + 1
        self.process = SimpleNamespace(pid=1000 + index, poll=Mock(return_value=None), wait=Mock(return_value=0))
        self.launches.append((command, cwd, env))
        self.payloads[f"http://worker-{index}.test"] = []
        if path := env.get("COMFYUI_RS_MODEL_PHASE_PATH"):
            probe = Path(path).parent / "probe_nodes" / benchmark.phase_report.PROBE_FILENAME
            self.probe_bytes.append(probe.read_bytes())
            Path(path).write_text(FakeModelPhaseDocker.marker + "\n", encoding="utf-8")
        return self.process

    def stop(self, pid, signum):
        """核对只向当前测试子进程组发送结束信号。"""
        assert self.process.pid == pid
        assert signum == benchmark.signal.SIGTERM
        self.stopped.append(pid)
        self.process = None

    def request(self, base_url, route, timeout, payload=None):
        """返回当前进程的设备和工作流结果，并模拟本地阶段记录。"""
        assert self.process is not None and base_url == f"http://127.0.0.1:{self.port}"
        index = len(self.launches)
        command, _cwd, env = self.launches[-1]
        if route == "/system_stats":
            kind = "cpu" if self.fail_start == index else "cuda"
            return {"system": {"argv": command, "comfyui_version": "test", "python_version": "test",
                               "pytorch_version": "test"}, "devices": [{"type": kind}]}
        result = super().request(f"http://worker-{index}.test", route, timeout, payload)
        path = env.get("COMFYUI_RS_MODEL_PHASE_PATH")
        if path is not None and route == "/prompt" and payload is not None:
            with Path(path).open("a", encoding="utf-8") as file:
                file.write(json.dumps({
                    "schema": 1, "prompt_id": result["prompt_id"], "workflow_host_ms": 5, "probe_errors": 0,
                    "phases": {"model_load": {"calls": 1, "host_ms": 3}},
                    "weight_move_cuda": {"stream_ms": None, "events_complete": 0, "events_incomplete": 0,
                                         "events_dropped": 0, "events_failed": 0},
                }) + "\n")
        return result


class PodNativeBenchmarkTests(unittest.TestCase):
    def run_benchmark(self, fake, *extra_args):
        """从 CLI 跑完整 Pod 路径，替换 GPU、进程和 HTTP。"""
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            app = Path(directory).resolve()
            (app / "comfyui").mkdir()
            (app / "comfyui" / "main.py").touch()
            first, second, output = [app / f"{name}.json" for name in ("a", "b", "report")]
            first.write_text(json.dumps(workflow(11)), encoding="utf-8")
            second.write_text(json.dumps(workflow(22, "RandomNoise", "noise_seed")), encoding="utf-8")
            gpu = {"enabled": True, "device_uuid": fake.gpu_uuid, "sample_count": 3,
                   "sample_error_count": 0, "baseline_mib": 1024, "peak_mib": 4096}
            with (
                patch.object(benchmark, "command_output", side_effect=fake.command),
                patch.object(benchmark, "request_json", side_effect=fake.request),
                patch.object(benchmark.subprocess, "Popen", side_effect=fake.launch),
                patch.object(benchmark.os, "killpg", side_effect=fake.stop),
                patch.object(benchmark.socket, "socket") as socket,
                patch.object(benchmark, "GpuSampler") as sampler,
            ):
                socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", fake.port)
                sampler.return_value.stop.return_value = gpu
                code = benchmark.main([
                    "--pod-native-compare", "--app-dir", str(app), "--gpu", "2", "--workflow", str(first),
                    "--workflow", str(second), "--output", str(output), "--repeats", "2", *extra_args,
                ])
                selectors = [call.args[0] for call in sampler.call_args_list]
            return code, json.loads(output.read_text(encoding="utf-8")), selectors

    def test_default_rounds_reverse_order_and_reuse_workload_without_probe(self):
        """无探针模式复用唯一 GPU、种子和工作流，两轮交换顺序。"""
        fake = FakePodComfyUI()
        code, report, selectors = self.run_benchmark(fake, "--no-phase-probe")
        self.assertEqual(code, 0)
        self.assertTrue(report["comparable"])
        self.assertEqual([row["order"] for row in report["rounds"]], [["rust", "python"], ["python", "rust"]])
        self.assertEqual([env["COMFYUI_RS_NATIVE_MODE"] for _, _, env in fake.launches], ["rust", "python", "python", "rust"])
        self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for _, _, env in fake.launches], [fake.gpu_uuid] * 4)
        self.assertTrue(all("COMFYUI_RS_MODEL_PHASE_PATH" not in env for _, _, env in fake.launches))
        self.assertEqual(selectors, [fake.gpu_uuid] * 4)
        self.assertEqual(fake.stopped, [1001, 1002, 1003, 1004])
        prompts = [rows for host, rows in fake.payloads.items() if host.startswith("http://worker-")]
        self.assertEqual([len(rows) for rows in prompts], [6] * 4)
        self.assertEqual(prompts[1:], prompts[:1] * 3)
        self.assertNotIn("phase_metrics_complete", report["rounds"][0])
        self.assertEqual(report["paired"]["gpu_peak_mib"]["pairs"], 2)

    def test_default_probe_profiles_matching_measured_jobs(self):
        """两组使用同一份探针文件，阶段统计排除预热。"""
        fake = FakePodComfyUI()
        code, report, _selectors = self.run_benchmark(fake, "--rounds", "1")
        self.assertEqual(code, 0)
        self.assertEqual(fake.probe_bytes, [benchmark.MODEL_PHASE_PROBE.read_bytes()] * 2)
        row = report["rounds"][0]
        self.assertTrue(row["phase_metrics_complete"])
        for result in row["variants"].values():
            self.assertEqual(result["model_phases"]["measured_records"], 4)
            self.assertEqual(result["model_phases"]["host_phases"]["model_load"]["median_host_ms"], 3)

    def test_startup_failure_retains_partial_round_and_cleans_up(self):
        """第二组设备准备失败时终止两个已启动进程，保留第一组。"""
        fake = FakePodComfyUI(fail_start=2)
        code, report, selectors = self.run_benchmark(fake, "--no-phase-probe")
        self.assertEqual(code, 1)
        self.assertFalse(report["complete"])
        self.assertFalse(report["comparable"])
        self.assertEqual(fake.stopped, [1001, 1002])
        self.assertEqual(len(fake.launches), 2)
        self.assertEqual(selectors, [fake.gpu_uuid])
        self.assertIn("python", report["pod"]["error"])
        self.assertEqual(len(report["rounds"]), 1)
        self.assertEqual(list(report["rounds"][0]["variants"]), ["rust"])


if __name__ == "__main__":
    unittest.main()
