# ComfyUI RS

V0.4 在同一个仓库构建和运行 ComfyUI 的 Python + Rust 版本。`comfyui/` 包含来自本地 ComfyUI `7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b` 的 Git 跟踪源码、测试、配置及许可证，并保留了该 checkout 中已有的 CLI 和服务端修改。前端继续通过 ComfyUI 依赖清单里的固定版本安装。

## 构建和启动

安装 Python 3.10 或更新版本和 Rust 1.85 或更新版本。在项目根目录运行：

```sh
python3 build.py
python main.py
```

`build.py` 创建项目自己的 `.venv`，安装 `comfyui/requirements.txt` 和编译后的 Rust 扩展。`python main.py` 用同一个环境启动内置 ComfyUI，原样传递 ComfyUI 启动参数。默认 Web UI 地址为 `http://127.0.0.1:8188`。已有的 `python3 run.py` 脚本仍可使用。

```sh
# 用 CPU 启动，适合验证构建和工作流入口。
python main.py --cpu
# 使用现有模型目录，并保留内置节点和自带配置。
python main.py --models-directory /path/to/models
# 复用已经配置了 ComfyUI 依赖和 PyTorch 的虚拟环境。
python3 build.py --venv /path/to/venv --no-deps
python main.py --venv /path/to/venv
```

`build.py` 使用所选 Python 环境的 pip 配置。已有匹配部署环境的 CUDA 版 PyTorch 时，可以用 `--venv` 直接复用。

默认数据路径与原版 ComfyUI 同名，位于项目根目录。模型放在 `models/`，自定义节点放在 `custom_nodes/`，输入和输出分别使用 `input/`、`output/`。原来的 `user/` 可以整个复制过来，默认用户的工作流位于 `user/default/workflows/`。将 `extra_model_paths.yaml` 一起复制到根目录后，其中的相对路径仍以该 YAML 所在目录为基准。`--base-directory` 和单独的目录参数仍可显式指定其他位置。

## Docker 构建与编译测试

```sh
docker build --progress=plain -t comfyui-rs:cpu .
docker run --rm -p 127.0.0.1:8188:8188 \
  -v "$PWD/models:/app/models" -v "$PWD/output:/app/output" \
  comfyui-rs:cpu
```

Docker 构建编译并安装 Rust 扩展，运行 Rust、Python 回归测试和 ComfyUI CPU 启动检查。容器数据目录与项目根目录布局一致，可把现有的 `input/`、`user/` 和 `custom_nodes/` 挂载到 `/app/` 下同名目录。旧的 `extra_model_paths.yaml` 可挂载到 `/app/extra_model_paths.yaml`，其引用的目录也需要在容器内可访问。启动入口使用 `main.py`，默认监听容器的 8188 端口。

### Linux amd64 CUDA 镜像

CUDA 镜像使用 PyTorch `2.15.0+cu132` 和 torchvision `0.29.2+cu132`，构建会检查 wheel 包含 RTX 5090 所需的 `sm_120` 和 H200 所需的 `sm_90`。宿主机需要 NVIDIA Container Toolkit，建议使用与 CUDA 13.2 对应的 R595 或更新驱动。[PyTorch CUDA wheel](https://download.pytorch.org/whl/cu132/torch/) · [NVIDIA GPU 架构表](https://developer.nvidia.com/cuda/gpus) · [CUDA 驱动对应表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver) · [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```sh
docker build --platform linux/amd64 -f Dockerfile.cuda -t comfyui-rs:cuda-amd64 .
docker run --rm --gpus all -p 127.0.0.1:8188:8188 \
  -v "$PWD/models:/app/models" -v "$PWD/output:/app/output" \
  comfyui-rs:cuda-amd64
```

容器沿用 `python main.py`、8188 端口和 `/app/` 数据目录。可在镜像名后追加原有 ComfyUI 参数。下面的命令可在有 GPU 的宿主机验证容器实际执行了 CUDA 运算，并显示 GPU 架构。

```sh
docker run --rm --gpus all --entrypoint /app/.venv/bin/python comfyui-rs:cuda-amd64 -c \
  'import torch; assert torch.cuda.is_available(); x=torch.ones((16,16), device="cuda"); assert (x@x).sum().item()==4096; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

## 原版与混合版压测

用两个版本启动同一 ComfyUI 提交、同一模型和节点依赖，统一 PyTorch、启动参数及缓存设置。在压测机上准备 `nvidia-smi`，让两个实例分别使用同型号的独占 GPU。脚本按顺序执行两边，输入使用 ComfyUI 导出的 API 格式 JSON，可以重复传入 `--workflow` 形成一套混合负载。

```sh
python3 scripts/benchmark_comfyui.py \
  --original-url http://127.0.0.1:8188 --hybrid-url http://127.0.0.1:8189 \
  --workflow workloads/image-api.json --workflow workloads/video-api.json \
  --original-gpu "$ORIGINAL_GPU" --hybrid-gpu "$HYBRID_GPU" \
  --warmups 2 --repeats 50 --concurrency 2 --output benchmark.json
```

压测复用工作流顺序、运行次数和种子变化规则，预热结果单独统计。吞吐按成功任务数除以测量阶段用时计算，P99 统计成功任务从提交到查询到结果的耗时。显存峰值是 `nvidia-smi` 定时采样的整卡占用最大值。OOM 率只统计可判定的执行结果，提交拒绝、超时和结果未知单独列出。`--no-gpu-metrics` 可用于 CPU 验证，报告中的显存值为空。

工作流执行顺序会影响模型切换时，可用 `--concurrency 1` 固定提交顺序。脚本默认修改内置采样节点的整数字段种子，`--keep-seeds` 可保留原值。需要让固定输入的工作流每次完整执行时，两端可使用相同的 `--cache-none`。

只有一张卡时，每次只运行一个实例，先采集原版，再停掉原版并启动混合版。分别传 `--original-url` 或 `--hybrid-url`，其余压测参数保持相同，然后合并两个报告。可以用这个方法分别测 5090 和 H200。

```sh
python3 scripts/benchmark_comfyui.py --original-url http://127.0.0.1:8188 \
  --workflow workloads/image-api.json --original-gpu "$GPU_UUID" --repeats 50 --output original.json
# 停止原版实例，再用同一张 GPU 启动混合版。
python3 scripts/benchmark_comfyui.py --hybrid-url http://127.0.0.1:8188 \
  --workflow workloads/image-api.json --hybrid-gpu "$GPU_UUID" --repeats 50 --output hybrid.json
python3 scripts/benchmark_comfyui.py --compare-reports original.json hybrid.json --output comparison.json
```

## Python 和 Rust 的接入点

- `comfyui/` 提供原有 Web 服务、工作流执行、模型加载和节点。张量与推理仍由 PyTorch 处理。
- `src/` 提供 Rust 调度、显存和权重预算、缓存及执行图规划。`python/comfyui_rs/` 是 PyO3 模块的 Python 接入层。

源码中已接入 Rust，启动时默认启用。设置 `COMFYUI_RS_NATIVE_MODE=python` 并重启，可以同时切回上游任务堆、显存和权重计算、缓存、依赖图和分块规划。显存回收排序也支持单独切换：

| `COMFYUI_RS_EVICTION_MODE` | 运行方式 |
| --- | --- |
| `rust`（默认） | 使用 Rust 排序，原生扩展无法加载或输入无法转换时使用 Python |
| `shadow` | 同时运行 Python 和 Rust，使用 Python 结果，发现差异时记录警告 |
| `python` | 直接使用 Python 内置排序 |

```sh
COMFYUI_RS_NATIVE_MODE=python python main.py
COMFYUI_RS_EVICTION_MODE=python python main.py
```

排序保留相同键的输入顺序和原元组对象。本机微基准使用 macOS arm64、CPython 3.13 和 Release wheel，20 个合成候选的中位耗时为 Python 0.53 µs、Rust 调用路径 1.50 µs。真实 GPU 的显存峰值、OOM 率和 P99 仍需测量。

## 实现范围

Web UI 跟随 ComfyUI，API 和自定义节点继续使用 Python，模型推理继续使用 PyTorch。

| 优先级 | ComfyUI 模块 | V0.4 接入点 |
| --- | --- | --- |
| P0 | `comfy/model_management.py` | Rust 计算低显存权重加载预算和回收候选顺序，PyTorch 管理实际分配。 |
| P0 | `comfy/memory_management.py` | 小批次直接计算。较大的普通张量批次在一次 Rust 调用中合并几何尺寸、字节对齐和聚合缓冲区偏移，PyTorch 创建张量视图。 |
| P0 | `execution.py` | Rust 管理单实例优先级队列，Python 执行工作流、记录历史和处理中断。 |
| P1 | `comfy/model_patcher.py` | Rust 计算经典加载的预取窗口和动态显存的权重加载顺序，沿用 ComfyUI 的张量及权重处理。 |
| P1 | `comfy_execution/caching.py` | Rust 选择 LRU 淘汰项及 RAM 压力回收顺序，Python 释放对象并复测内存。 |
| P2 | `comfy_execution/graph.py` | Rust 选择就绪节点、分析循环依赖，并为控制节点批量规划待移除的边和依赖释放。 |
| P2 | `server.py` | Rust 为 `/prompt` 管理并发提交名额。HTTP 响应和工作流校验由 Python 负责，队列容量检查在原有入队锁内完成。 |
| 按需 | `comfy/utils.py` | Rust 规划图像和视频分块的边界，PyTorch 执行切片与融合。 |

### 配置任务接入上限

```sh
COMFYUI_RS_MAX_PENDING_PROMPTS=32 COMFYUI_RS_MAX_CONCURRENT_SUBMISSIONS=4 python main.py
```

`COMFYUI_RS_MAX_PENDING_PROMPTS` 限制单实例中等待执行的任务数。`COMFYUI_RS_MAX_CONCURRENT_SUBMISSIONS` 限制同时处理的 `/prompt` 请求数。达到任一上限时，`/prompt` 返回 HTTP 429。省略变量或设为 `0` 时，使用 ComfyUI 的无限制行为。

## 在测试 worker 采集基线

内置观测节点按需启用。它在每次工作流执行后记录耗时和状态，在 NVIDIA CUDA 上记录驱动空闲显存和 PyTorch 分配器的峰值：

```sh
COMFYUI_RS_OBSERVE=1 python main.py
```

在 INFO 日志中搜索 `comfyui_rs_observation `，其后的 JSON 可以通过 `prompt_id` 对照 ComfyUI 历史记录。`duration_ms` 是执行器耗时，`status` 区分 `success`、`error`、`oom`、`interrupted`。`cuda` 字段包含驱动空闲字节、PyTorch 执行前保留字节、保留及分配峰值。观测会重置当前进程的每设备峰值计数器，适合在单 GPU、单 worker 的测试实例运行。设备级显存采样可补充观测 CUDA 扩展、其他进程和 DynamicVRAM 占用。

如果要在日志中附加预算判断，设置 `COMFYUI_RS_EXPECTED_EXTRA_BYTES` 为预计新增显存字节数，并通过 `COMFYUI_RS_HEADROOM_BYTES` 设置安全余量，默认为 `0`。预算判断只用于观测。

## 本地验证

```sh
cargo test --locked --all-features
.venv/bin/python -m unittest discover -s tests
```

Python 测试直接加载本仓库中的集成版本。设置 `COMFYUI_RS_COMFYUI_PATH` 可以额外与原版 checkout 对照。原版许可证见 `comfyui/LICENSE`。
