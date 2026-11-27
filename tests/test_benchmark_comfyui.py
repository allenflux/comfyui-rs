import copy
import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
