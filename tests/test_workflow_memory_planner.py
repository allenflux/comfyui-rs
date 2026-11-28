import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


COMFYUI = Path(__file__).resolve().parents[1] / "comfyui"
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "comfy_aimdo", "comfy_kitchen"))


def check_planner():
    sys.path.insert(0, str(COMFYUI))
    sys.argv[:] = [str(COMFYUI / "main.py"), "--cpu"]
    from comfy import options

    options.enable_args_parsing()
    import nodes
    import torch
    from comfy import model_management
    from comfy.controlnet import ControlBase
    from comfy.model_patcher import ModelPatcher
    from comfy_api.latest import io
    from comfy_execution.graph import DynamicPrompt
    from comfy_execution.workflow_memory import WorkflowMemoryPlanner

    def model(size, loaded):
        patcher = ModelPatcher(torch.nn.Linear(1, 1), torch.device("cuda:0"), torch.device("cpu"))
        patcher.model_size = lambda: size
        patcher.loaded_size = lambda: loaded
        return patcher

    def node(class_type="MemoryTestNode", **inputs):
        return {"class_type": class_type, "inputs": inputs}

    class ExpandingNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="MemoryTestExpanding", enable_expand=True)

        @classmethod
        def execute(cls):
            return io.NodeOutput()

    class Control(ControlBase):
        def get_models(self):
            return [shared.clone()]

    nodes.NODE_CLASS_MAPPINGS.update(MemoryTestNode=object, MemoryTestExpanding=ExpandingNode)
    shared, unused = model(400, 400), model(200, 200)
    # 采样分支通过惰性链接到达，并在缓存的条件里持有共享权重的克隆。
    prompt = DynamicPrompt({
        "loader": node(),
        "encode": node(clip=["loader", 0]),
        "control": node(),
        "sample": node(conditioning=["control", 0]),
        "output": node(lazy_input=["sample", 0]),
    })
    planner = WorkflowMemoryPlanner("aggressive", prompt, ["encode", "output"], {
        "loader": [[shared], [unused]],
        "control": [[[[torch.zeros(1), {"control": Control()}]]]],
    })
    with (
        patch.object(model_management, "offloadable_models", return_value=[shared, unused]),
        patch.object(model_management, "offload_models") as offload,
    ):
        planner.after_node("encode", ["output"], {shared}, allow_offload=True)
        offload.assert_not_called()
        planner.observe_outputs("sample", [[torch.zeros(1)]])
        planner.after_node("sample", ["output"], {shared}, allow_offload=True)
        offload.assert_called_once_with([shared])

    # 展开节点在运行时才引入引用已有模型的消费者。
    prompt = DynamicPrompt({"first": node(), "expand": node("MemoryTestExpanding")})
    planner = WorkflowMemoryPlanner("aggressive", prompt, prompt.all_node_ids(), {})
    with (
        patch.object(model_management, "offloadable_models", return_value=[shared]),
        patch.object(model_management, "offload_models") as offload,
    ):
        planner.observe_outputs("first", [[shared]])
        planner.after_node("first", ["expand"], {shared}, allow_offload=True)
        offload.assert_not_called()
        prompt.add_ephemeral_node("extra", node(model=["first", 0]), "expand", "expand")
        planner.after_node("expand", ["extra"], set(), allow_offload=True)
        offload.assert_not_called()
        planner.after_node("extra", [], {shared}, allow_offload=True)
        offload.assert_called_once_with([shared])

    # 自适应策略只在下游权重和推理预留量超过可用显存时回收，并在填平缺口后停止。
    requested = model(400, 0)
    prompt = DynamicPrompt({"first": node(), "loader": node(), "next": node(model=["loader", 0])})
    for free_bytes, expected in ((300, [shared]), (900, [])):
        planner = WorkflowMemoryPlanner("adaptive", prompt, ["first", "next"], {"loader": [[requested]]})
        with (
            patch.object(model_management, "offloadable_models", return_value=[shared, unused]),
            patch.object(model_management, "minimum_inference_memory", return_value=200),
            patch.object(model_management, "get_total_memory", return_value=1000),
            patch.object(model_management, "get_free_memory", return_value=free_bytes),
            patch.object(model_management, "offload_models") as offload,
        ):
            planner.after_node("first", ["next"], {shared, unused}, allow_offload=True)
            if expected:
                offload.assert_called_once_with(expected)
            else:
                offload.assert_not_called()


@unittest.skipUnless(HAS_RUNTIME, "请在项目构建环境中运行工作流显存规划测试")
class WorkflowMemoryPlannerTests(unittest.TestCase):
    def test_releases_only_eligible_models_for_the_remaining_graph_and_pressure(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--check-planner"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-planner"]:
        check_planner()
    else:
        unittest.main()
