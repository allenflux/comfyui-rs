"""根据工作流依赖和节点实际运行结果，确定模型权重的显存驻留时机。"""

import logging
from collections import defaultdict
from weakref import WeakValueDictionary

import torch

import nodes
from comfy import model_management
from comfy.clip_vision import ClipVisionModel, Output as ClipVisionOutput
from comfy.controlnet import ControlBase
from comfy.hooks import EnumHookType, HookGroup
from comfy.model_patcher import ModelPatcher
from comfy.samplers import CFGGuider
from comfy_api.internal import _ComfyNodeInternal
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker, is_link
from comfy_execution.node_monitor import custom_node_source


def _model_key(model):
    # LoRA 等节点会返回不同的 patcher，但这些 patcher 可以共享同一份底层权重。
    # 加上设备可以分别计算多卡工作流中每张卡的驻留量。
    return id(model.model), model.load_device


def _output_models(output):
    """遍历输出插槽里的模型引用，包括嵌在条件、引导器和挂钩中的模型。"""
    models = set()
    seen = set()
    pending = [output]
    opaque = False
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))

        if isinstance(value, ModelPatcher):
            models.add(value)
            pending.extend(value.model_patches_models())
            pending.extend(value.get_additional_models())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, ControlBase):
            pending.extend(value.get_models())
        elif isinstance(value, CFGGuider):
            pending.append(value.model_patcher)
            pending.append(value.original_conds)
        elif isinstance(value, HookGroup):
            for hook in value.get_type(EnumHookType.AdditionalModels):
                pending.extend(hook.models)
        elif isinstance(value, ClipVisionModel):
            pending.extend((value.patcher, value.naf))
        elif isinstance(value, ClipVisionOutput):
            pending.extend(vars(value).values())
        elif isinstance(value, (torch.Tensor, str, int, float, bytes, ExecutionBlocker)) or value is None:
            continue
        else:
            patcher = getattr(value, "patcher", None)
            if isinstance(patcher, ModelPatcher):
                pending.append(patcher)
            else:
                opaque = True
    return models, opaque


def _can_change_graph(class_type):
    node_class = nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        return True
    if issubclass(node_class, _ComfyNodeInternal):
        try:
            schema = node_class.SCHEMA or node_class.GET_SCHEMA()
        except Exception:
            # 惰性分支可能永远不会执行。读不到节点声明时保留模型，交给执行器处理该分支。
            return True
        return schema.enable_expand or any(
            hidden in (schema.hidden or ()) for hidden in (io.Hidden.dynprompt, io.Hidden.execution_list)
        )
    # 旧版自定义节点可以直接返回子图，没有在节点声明中标记这一能力。
    return custom_node_source(node_class) is not None


class WorkflowMemoryPlanner:
    def __init__(self, strategy, prompt, pending_nodes, cached_outputs):
        self.strategy = strategy
        self.prompt = prompt
        self._nodes = {}
        self._slots = {}
        # 计划只记录身份和依赖。用弱引用查询模型大小和权重身份，以免延长缓存模型的生命周期。
        self._models = WeakValueDictionary()
        self._weights = WeakValueDictionary()
        self._touched = set()
        self._enabled = True

        for node_id, outputs in cached_outputs.items():
            self.observe_outputs(node_id, outputs)
        # 在首个节点执行前整理目标输出可达的依赖。后续按实际待执行的节点重新查找使用者。
        self._remaining_uses(pending_nodes)

    def _remember(self, models):
        keys = set()
        for model in models:
            key = _model_key(model)
            # 采样时可能短暂创建共享权重的 clone，优先保留图中已有的可用引用。
            if self._models.get(key) is None:
                self._models[key] = model
            self._weights[key] = model.model
            keys.add(key)
        return keys

    def _node_plan(self, node_id):
        node = self.prompt.get_node(node_id)
        plan = self._nodes.get(node_id)
        # 循环和子图可以替换节点定义。定义变化时重新提取链接。
        if plan is None or plan[0] is not node:
            links = tuple(
                (value[0], value[1]) for value in node["inputs"].values()
                if is_link(value) and isinstance(value[1], int)
            )
            plan = node, links, _can_change_graph(node["class_type"])
            self._nodes[node_id] = plan
        return plan[1:]

    def _slot(self, producer, index):
        outputs = self._slots.get(producer)
        if outputs is None or not -len(outputs) <= index < len(outputs):
            return (), False
        return outputs[index]

    def _remaining_uses(self, pending_nodes):
        # 把可能激活的惰性输入也算作未来依赖。遇到已缓存或已经完成的插槽，
        # 直接使用该插槽里的模型引用，避免把只参与过上游计算的模型算作下游依赖。
        visit = list(pending_nodes)
        seen = set()
        uses = []
        uncertain = False
        while visit:
            node = visit.pop()
            if node in seen or not self.prompt.has_node(node):
                continue
            seen.add(node)
            links, can_change_graph = self._node_plan(node)
            uncertain |= can_change_graph
            needed = set()
            for producer, index in links:
                if producer not in self._slots:
                    visit.append(producer)
                    continue
                keys, opaque = self._slot(producer, index)
                needed.update(keys)
                uncertain |= opaque
            if needed:
                uses.append(needed)
        return uses, uncertain

    def observe_outputs(self, node_id, outputs):
        if not self._enabled:
            return
        slots = []
        for output in outputs or ():
            try:
                models, opaque = _output_models(output)
                slots.append((self._remember(models), opaque))
            except Exception:
                logging.debug("Could not inspect a model output at node %s", node_id, exc_info=True)
                slots.append((set(), True))
        self._slots[node_id] = slots

    def after_node(self, node_id, pending_nodes, loaded_models, *, allow_offload, cached=False):
        if not self._enabled:
            return

        try:
            if not cached:
                links, _ = self._node_plan(node_id)
                used = self._remember(loaded_models)
                unknown_input = False
                for producer, index in links:
                    keys, opaque = self._slot(producer, index)
                    used.update(keys)
                    unknown_input |= opaque
                self._touched.update(used)

                # 无法直接识别的输出有可能封装了输入模型或运行时加载的模型。
                # 传递这些候选引用，直到该输出的使用者执行完。缓存命中缺少当时的输入信息。
                if used and node_id in self._slots:
                    self._slots[node_id] = [
                        (keys | used, unknown_input) if opaque else (keys, False)
                        for keys, opaque in self._slots[node_id]
                    ]

            if not allow_offload or not self._touched:
                return
            resident = model_management.offloadable_models()
            candidates = [
                model for model in resident
                if (key := _model_key(model)) in self._touched and self._weights.get(key) is model.model
            ]
            if not candidates:
                return

            uses, uncertain = self._remaining_uses(pending_nodes)
            if uncertain:
                return
            live = set().union(*uses)
            # 多卡副本的存储身份不同。只要共享模型家族仍有使用者，就暂缓卸载其副本。
            live_families = {model.clone_base_uuid for key in live if (model := self._models.get(key)) is not None}
            candidates = [
                model for model in candidates
                if _model_key(model) not in live and model.clone_base_uuid not in live_families
            ]
            if candidates and self.strategy == "adaptive":
                candidates = self._under_pressure(candidates, resident, uses)
            if candidates:
                count = model_management.offload_models(candidates)
                logging.debug("Workflow memory (%s) offloaded %d models after node %s", self.strategy, count, node_id)
        except Exception:
            self._enabled = False
            logging.warning("Workflow model offloading failed. Resuming standard memory management.", exc_info=True)

    def _under_pressure(self, candidates, resident, uses):
        # 每个节点只补齐其尚未驻留的权重。以单个后续节点的最大需求估算每张卡的压力，
        # 因为不同节点的模型通常不会在同一阶段加载。
        loaded_bytes = {_model_key(model): model.loaded_size() for model in resident}
        upcoming_bytes = defaultdict(int)
        for keys in uses:
            node_bytes = defaultdict(int)
            for key in keys:
                if (model := self._models.get(key)) is not None:
                    node_bytes[model.load_device] += max(0, model.model_size() - loaded_bytes.get(key, 0))
            for device, size in node_bytes.items():
                upcoming_bytes[device] = max(upcoming_bytes[device], size)

        headroom = model_management.minimum_inference_memory()
        shortfall = {}
        targets = []
        # 复用原生加载路径的推理预留量和 10% 的权重余量。只回收足以补齐缺口的候选模型。
        for model in sorted(candidates, key=lambda model: loaded_bytes[_model_key(model)], reverse=True):
            device = model.load_device
            if device not in shortfall:
                need = min(model_management.get_total_memory(device), headroom + upcoming_bytes[device] * 1.1)
                shortfall[device] = need - model_management.get_free_memory(device)
            if shortfall[device] > 0:
                targets.append(model)
                shortfall[device] -= loaded_bytes[_model_key(model)]
        return targets
