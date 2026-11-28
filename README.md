# ComfyUI RS

V0.4 在同一个仓库构建和运行 ComfyUI 的 Python + Rust 版本。`comfyui/` 包含来自本地 ComfyUI `7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b` 的 Git 跟踪源码、测试、配置及许可证，并保留了该 checkout 中已有的 CLI 和服务端修改。前端继续通过 ComfyUI 依赖清单里的固定版本安装。

## 构建和启动

安装 CPython 3.13 或更新版本和 Rust 1.85 或更新版本。在 PyPI 提供当前平台 wheel 的机器上，从项目根目录运行：

```sh
python3 build.py
python3 main.py
```

`build.py` 创建 `.venv-candle`，从 pip 配置的索引安装自带 Candle 的 `candle-torch` wheel，再安装 ComfyUI 依赖与 Rust 扩展。PyPI 的 `candle-torch 0.1.0` 提供 Apple Silicon macOS 的 CPU wheel。需要 Metal、CUDA 或当前索引缺少平台 wheel 时，可用 `python3 build.py --candle-wheels /path/to/wheels` 安装对应的自带 Candle 的 wheel。[PyPI 发布文件](https://pypi.org/project/candle-torch/0.1.0/#files)

`python main.py` 与 `python run.py` 都使用 `.venv-candle` 并转发 ComfyUI 参数。默认 Web UI 地址为 `http://127.0.0.1:8188`。

```sh
# 用 CPU 启动，适合验证构建和工作流入口。
python main.py --cpu
# 使用现有模型目录，并保留内置节点和自带配置。
python main.py --models-directory /path/to/models
# 已有安装了 Candle wheel 和兼容依赖的环境时，只重建 Rust 扩展。
python3 build.py --venv /path/to/candle-env --no-deps
python3 main.py --venv /path/to/candle-env
```

启动时会检查 `torch` 和 `candle` 是否由同一个 `candle-torch` wheel 提供，并拒绝安装了原版 `torch`、`torchvision`、`torchaudio` 或独立 `candle-nn` 的环境。工作流支持范围取决于 Candle 及兼容层的算子和节点覆盖。

`build.py` 单独安装 `torchsde`、`kornia` 和 `spandrel`，避免它们的上游包元数据拉入原版 PyTorch。它们在运行时使用 `candle-torch` 提供的接口。

用 Candle 环境执行 `scripts/smoke_candle_workflow.py` 可以验证内置图像工作流的 PNG 输出和三步 Euler 采样。测试数据保存在临时目录。

默认数据路径与原版 ComfyUI 同名，位于项目根目录。模型放在 `models/`，自定义节点放在 `custom_nodes/`，输入和输出分别使用 `input/`、`output/`。原来的 `user/` 可以整个复制过来，默认用户的工作流位于 `user/default/workflows/`。将 `extra_model_paths.yaml` 一起复制到根目录后，其中的相对路径仍以该 YAML 所在目录为基准。`--base-directory` 和单独的目录参数仍可显式指定其他位置。

### 工作流显存调度

直接运行 `python main.py` 会使用 `adaptive`。可在启动时显式选择其他策略。

```sh
python main.py --memory-strategy aggressive
python main.py --memory-strategy default
```

| 策略 | 行为 |
| --- | --- |
| `adaptive` | 根据当前可用显存、推理预留量和已知后续节点的权重需求决定是否提前回收。`workflow-aware` 是该模式的别名。 |
| `aggressive` | 模型最后一次使用后，把本次工作流用到的可卸载权重移回 CPU。 |
| `default` | 沿用 ComfyUI 的显存管理。 |

调度从请求的输出节点反向分析依赖，并用缓存和节点的实际输出识别模型引用。例如，一次工作流有两个提示词编码节点共享同一个 CLIP，`aggressive` 会等两个节点都结束才卸载 CLIP。LoRA 克隆和条件里引用的 ControlNet 也会计入后续使用。异步任务、动态子图或无法判断的依赖尚未解析时，调度会推迟回收。

`adaptive` 按设备估算下游单个节点需要补入的权重，以及推理所需的预留空间。可用空间足够时保留已驻留的权重，便于队列中的下一个任务复用。模型按需加载，实际加载预算和动态分页继续由 ComfyUI 模型管理器处理。

调度只管理 ComfyUI 登记且能卸载到 CPU 的权重。CPU、MPS 共享内存、`--highvram`、`--gpu-only` 和 `--disable-smart-memory` 沿用各自的内存策略。

## Docker 构建与编译测试

```sh
docker build --build-context candle_wheels=/path/to/linux-cpu-wheels \
  --progress=plain -t comfyui-rs:cpu .
docker run --rm -p 127.0.0.1:8188:8188 \
  -v "$PWD/models:/app/models" -v "$PWD/output:/app/output" \
  comfyui-rs:cpu
```

`candle_wheels` 指向包含自带 Candle 的 Linux `candle-torch` wheel 的目录。CPU 镜像构建会编译 Rust 扩展并运行回归测试和 ComfyUI 启动检查。容器数据目录与项目根目录布局一致，可把现有的 `input/`、`user/` 和 `custom_nodes/` 挂载到 `/app/` 下同名目录。旧的 `extra_model_paths.yaml` 可挂载到 `/app/extra_model_paths.yaml`，其引用的目录也需要在容器内可访问。启动入口使用 `main.py`，默认监听容器的 8188 端口。

### Linux amd64 CUDA 镜像

CUDA 镜像需要为 Linux amd64 和 CUDA 13.2 构建的 `candle-torch` wheel。构建 Candle CUDA 内核时需选择目标 GPU 的计算能力，宿主机需安装兼容驱动和 NVIDIA Container Toolkit。[NVIDIA GPU 架构表](https://developer.nvidia.com/cuda/gpus) · [CUDA 驱动对应表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver) · [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```sh
docker build --build-context candle_wheels=/path/to/linux-cuda-wheels \
  --platform linux/amd64 -f Dockerfile.cuda -t comfyui-rs:cuda-amd64 .
docker run --rm --gpus all -p 127.0.0.1:8188:8188 \
  -v "$PWD/models:/app/models" -v "$PWD/output:/app/output" \
  comfyui-rs:cuda-amd64
```

容器沿用 `python main.py`、8188 端口和 `/app/` 数据目录。可在镜像名后追加原有 ComfyUI 参数。下面的命令可在有 GPU 的宿主机验证容器实际执行了 CUDA 运算，并输出设备名。

```sh
docker run --rm --gpus all --entrypoint /app/.venv-candle/bin/python comfyui-rs:cuda-amd64 -c \
  'import torch; assert torch.__candle_torch__ and torch.cuda.is_available(); x=torch.ones((16,16), device="cuda"); assert (x@x).sum().item()==4096; print(torch.cuda.get_device_name())'
```

## 原版与混合版压测

### 在同一个 Pod 对比开启 Rust 和 Python 回退

在有 `nvidia-smi` 的专用压测 Pod 内执行。脚本复用当前镜像的 `/app`、同一张 GPU 和同一套工作流，通过 `COMFYUI_RS_NATIVE_MODE` 切换模式。示例跑两轮，第一轮先 Rust 后 Python，第二轮反过来。每组启动一个新的测试进程，结束后由脚本清理。

```sh
cd /app
python3 scripts/benchmark_comfyui.py \
  --pod-native-compare --app-dir /app --data-dir /app --gpu 0 \
  --workflow workloads/model-a-api.json --workflow workloads/model-b-api.json \
  --rounds 2 --warmups 2 --repeats 30 --concurrency 1 --timeout 1800 \
  --output /tmp/comfyui-pod-comparison.json
```

正式对照需要独占所选 GPU。如果已有 ComfyUI 占用这张卡，脚本默认等待它释放再开始。脚本只管理本次启动的测试进程。只能在共享卡上试跑时可显式加 `--allow-shared-gpu`，报告会标记为探索结果。当前镜像若已支持 `COMFYUI_RS_NATIVE_MODE`，可把新版 `benchmark_comfyui.py`、`model_phase_report.py` 和 `model_phase_probe.py` 放在同一个 Pod 可读目录，使用该目录里的脚本和 `--app-dir /app`。

`--data-dir` 指向现有的模型、输入和自定义节点数据，`--gpu` 接收 Pod 内可见的 GPU 索引或 UUID。两组默认使用 `--cache-none`，可用 `--use-comfy-cache` 恢复 ComfyUI 的工作流缓存。用 `--comfy-arg=参数` 逐项传入两组共用的启动参数。

- 测常驻表现时只传一个单模型工作流并加 `--use-comfy-cache`，让加载节点可命中缓存。用内置采样节点的不同种子确保每次仍会执行推理。
- 测切换时传两个使用不同模型的工作流，保持 `--concurrency 1`，并在报告里核对模型切换次数。测显存压力时用生产中的高分辨率、大批次或视频输入，结合报告中的显存峰值、回收调用及 OOM 率判断是否触发压力。
- 初次试跑可用 `--rounds 1 --repeats 2`。加 `--warmups 0` 会把新进程的首次工作流计入测量。

指定 `--output` 时，每轮结束都会保存 JSON。报告按轮保留吞吐、端到端 P99、显存峰值、OOM 率、worker RAM 及分阶段耗时，并汇总配对差值。要在关闭阶段计时探针时复测速度，可加 `--no-phase-probe`。如需跨 Pod 保留报告，把 `--output` 指向已有持久卷上的可写路径。

### 改前、改后镜像的模型阶段对照

在 GPU 宿主机上，用相同版本的 ComfyUI、Python 和张量后端准备改前和改后镜像。为已有的改前镜像保留独立标签。以下命令构建使用 Candle 的改后镜像。

```sh
docker build --build-context candle_wheels=/path/to/linux-cuda-wheels \
  --platform linux/amd64 -f Dockerfile.cuda -t comfyui-rs:cuda-after .
```

选用不同模型的两份 API 格式工作流，让同一张 GPU 依次执行 A、B。模型的合计驻留显存接近或超过显卡容量时，更容易观察到回收和重新加载。`--gpu` 支持宿主机 GPU 索引或 UUID。测 H200 时选对应的卡，并把输出文件改为 `model-phases-h200.json`。

```sh
python3 scripts/benchmark_comfyui.py \
  --before-image comfyui-rs:cuda-before --after-image comfyui-rs:cuda-after \
  --data-dir "$PWD" --gpu 0 \
  --workflow workloads/model-a-api.json --workflow workloads/model-b-api.json \
  --warmups 2 --repeats 50 --concurrency 1 --timeout 1800 \
  --output model-phases-5090.json
```

脚本固定本地镜像 ID，在同一张卡上先跑改前，再清理容器并跑改后。两组复用同一份只读探针和数据挂载、工作流顺序、种子及启动参数。默认用 `--cache-none`，生产环境使用缓存时可加 `--use-comfy-cache`。需要部署专用参数时用 `--comfy-arg` 逐项传入。

报告对成功的测量任务统计模型读取与加载、模型切换、权重搬运和显存回收的调用次数、中位数及 P99。主机侧计时包含子调用，阶段间可能重叠。切换次数按相邻模型管理请求的模型身份集合变化计数，也包含单个工作流内部的切换。CUDA 事件只覆盖探针能跟踪的搬运，按各流事件区间累计，多个流的事件区间可以重叠。报告同时保留采集覆盖情况、吞吐、端到端 P99、显存峰值和 OOM 率。RTX 5090 和 H200 的 CUDA 实测待在对应主机完成。

### 一张 GPU 跑三组对照

在安装了 Docker、NVIDIA Container Toolkit 和 `nvidia-smi` 的 Linux 宿主机上构建 CUDA 镜像。用下面的命令从现有 ComfyUI 仓库准备一份固定提交的原版源码。独立工作树可以避开原有目录中的本地代码改动和 `extra_model_paths.yaml`。

```sh
git -C /path/to/ComfyUI worktree add --detach /path/to/ComfyUI-benchmark \
  7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b
```

将真实工作流从 ComfyUI 导出为 API 格式 JSON。下例依次提交 A、B 两个工作流。测试换模时，选用不同的模型，并让两个模型的常驻显存需求合计接近或超过 GPU 的可用显存。测试显存压力时，使用生产中的高分辨率、长视频或大批次输入。

```sh
python3 scripts/benchmark_comfyui.py \
  --docker-image comfyui-rs:cuda-amd64 \
  --original-source /path/to/ComfyUI-benchmark --data-dir "$PWD" --gpu 0 \
  --workflow workloads/model-a-api.json --workflow workloads/model-b-api.json \
  --warmups 2 --repeats 50 --concurrency 1 --timeout 1800 \
  --output benchmark-5090.json
```

脚本在选定的 GPU 上依次运行上游源码、混合版开启 Rust、混合版关闭 Rust。每组结束后移除容器，再启动下一组。三组共用同一镜像里的 Python 和张量后端，并要求上游节点在该环境里可运行。历史镜像可以继续使用 `/app/.venv`。`--gpu` 可填 `nvidia-smi` 显示的索引或 `GPU-...` UUID。压测期间让脚本独占该 GPU。

三组以只读方式共享 `--data-dir` 下已有的 `models/`、`input/`、`custom_nodes/` 和 `extra_model_paths.yaml`，输出、临时文件和用户状态写入各自容器。模型路径配置引用的路径需要在容器内可访问。自定义节点的 Python 依赖需提前装进同一个测试镜像。

`--concurrency 1` 保持交替提交顺序。Docker 模式默认用 `--cache-none` 让每次提交完整执行，并关闭资产系统。用 `--use-comfy-cache` 可以沿用 ComfyUI 的默认缓存。生产启动参数可用 `--comfy-arg` 逐个传给三组，例如 `--comfy-arg=--reserve-vram --comfy-arg=4`。慢启动可提高 `--startup-timeout`，单位为秒，默认 120。

吞吐按成功任务数除以测量阶段用时计算。P99 统计成功任务从提交到查询到结果的耗时。显存峰值取 `nvidia-smi` 对同一张整卡在测量阶段的采样最大值。OOM 率以已知执行终态为分母，拒绝、超时和结果未知分别计数。报告也记录 `worker_peak_rss_mib`，取 ComfyUI 主进程从启动到测量结束的常驻 RAM 峰值，包含预热。测 RAM 缓存回收时加 `--use-comfy-cache`，三组沿用相同的缓存配置。

三组复用相同的任务顺序和种子规则，预热单独统计。脚本默认变更内置采样节点的整数种子，`--keep-seeds` 可保留原值。

### 自行启动实例做双组对照

使用相同 ComfyUI 提交、模型、节点依赖、张量后端、启动参数和缓存设置。在压测机上准备 `nvidia-smi`，让两个实例分别使用同型号的独占 GPU。脚本按顺序执行两组，`--no-gpu-metrics` 可用于 CPU 验证，报告中的显存值为空。

```sh
python3 scripts/benchmark_comfyui.py \
  --original-url http://127.0.0.1:8188 --hybrid-url http://127.0.0.1:8189 \
  --workflow workloads/image-api.json --workflow workloads/video-api.json \
  --original-gpu "$ORIGINAL_GPU" --hybrid-gpu "$HYBRID_GPU" \
  --warmups 2 --repeats 50 --concurrency 2 --output benchmark.json
```

手动复用同一张卡时，每次启动一个实例。分别传 `--original-url` 或 `--hybrid-url`，其余压测参数保持相同，再合并两个报告。需要每次完整执行时，两端使用相同的 `--cache-none`。

```sh
python3 scripts/benchmark_comfyui.py --original-url http://127.0.0.1:8188 \
  --workflow workloads/image-api.json --original-gpu "$GPU_UUID" --repeats 50 --output original.json
# 停止原版实例，再用同一张 GPU 启动混合版。
python3 scripts/benchmark_comfyui.py --hybrid-url http://127.0.0.1:8188 \
  --workflow workloads/image-api.json --hybrid-gpu "$GPU_UUID" --repeats 50 --output hybrid.json
python3 scripts/benchmark_comfyui.py --compare-reports original.json hybrid.json --output comparison.json
```

## Python 和 Rust 的接入点

- `comfyui/` 提供原有 Web 服务、工作流执行、模型加载和节点。`candle-torch` 提供节点使用的 Python 张量接口，Candle 执行张量运算与推理。
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

Web UI 跟随 ComfyUI，API 和自定义节点继续使用 Python，兼容的模型与节点通过 Candle 执行推理。

| 优先级 | ComfyUI 模块 | V0.4 接入点 |
| --- | --- | --- |
| P0 | `comfy/model_management.py` | Rust 计算低显存权重加载预算和回收候选顺序，张量后端管理实际分配。 |
| P0 | `comfy/memory_management.py` | 小批次直接计算。较大的普通张量批次在一次 Rust 调用中合并几何尺寸、字节对齐和聚合缓冲区偏移，再由张量后端创建视图。 |
| P0 | `execution.py` | Rust 管理单实例优先级队列，Python 执行工作流、记录历史和处理中断。 |
| P1 | `comfy/model_patcher.py` | Rust 计算经典加载的预取窗口和动态显存的权重加载顺序，沿用 ComfyUI 的张量及权重处理。 |
| P1 | `comfy_execution/caching.py` | Rust 选择 LRU 淘汰项及 RAM 压力回收顺序，Python 释放对象并复测内存。 |
| P2 | `comfy_execution/graph.py` | Rust 选择就绪节点、分析循环依赖，并为控制节点批量规划待移除的边和依赖释放。 |
| P2 | `server.py` | Rust 为 `/prompt` 管理并发提交名额。HTTP 响应和工作流校验由 Python 负责，队列容量检查在原有入队锁内完成。 |
| 按需 | `comfy/utils.py` | Rust 规划图像和视频分块的边界，张量后端执行切片与融合。 |

### 配置任务接入上限

```sh
COMFYUI_RS_MAX_PENDING_PROMPTS=32 COMFYUI_RS_MAX_CONCURRENT_SUBMISSIONS=4 python main.py
```

`COMFYUI_RS_MAX_PENDING_PROMPTS` 限制单实例中等待执行的任务数。`COMFYUI_RS_MAX_CONCURRENT_SUBMISSIONS` 限制同时处理的 `/prompt` 请求数。达到任一上限时，`/prompt` 返回 HTTP 429。省略变量或设为 `0` 时，使用 ComfyUI 的无限制行为。

## 在测试 worker 采集基线

内置观测节点按需启用。它在每次工作流执行后记录耗时和状态，在支持相关统计的 NVIDIA CUDA 设备上记录驱动空闲显存和分配器峰值。

```sh
COMFYUI_RS_OBSERVE=1 python main.py
```

在 INFO 日志中搜索 `comfyui_rs_observation `，其后的 JSON 可以通过 `prompt_id` 对照 ComfyUI 历史记录。`duration_ms` 是执行器耗时，`status` 区分 `success`、`error`、`oom`、`interrupted`。`cuda` 字段包含驱动空闲字节、执行前保留字节、保留及分配峰值。观测会重置当前进程的每设备峰值计数器，适合在单 GPU、单 worker 的测试实例运行。设备级显存采样可补充观测 CUDA 扩展、其他进程和 DynamicVRAM 占用。

如果要在日志中附加预算判断，设置 `COMFYUI_RS_EXPECTED_EXTRA_BYTES` 为预计新增显存字节数，并通过 `COMFYUI_RS_HEADROOM_BYTES` 设置安全余量，默认为 `0`。预算判断只用于观测。

## 定位自定义节点 OOM

使用正常的 `python main.py` 启动即可。自定义节点监控默认开启，工作流和第三方节点沿用原有入口。Pod 中在应用日志里搜索 `comfyui_rs_node `，前缀后面是 JSON。用 `prompt_id` 找到工作流，再用 `node_id`、`node_class` 和 `pack` 定位节点。单次调用用 `call_id` 关联，`function` 和 `list_index` 可以区分节点方法及列表中的各次调用。

监控也会把每条记录追加到 `user/node-monitor/` 下按 worker 命名的 JSONL 文件。用 `--user-directory` 启动时，该目录随用户目录移动。可在启动时设置 `COMFYUI_RS_NODE_MONITOR_LOG_DIR` 指向已有持久卷上的可写目录，以便 Pod 重启后仍能取回日志。每个 worker 的文件默认按 8 MiB 轮转，保留 4 份历史文件。实际能回看的时长由节点调用量决定。

链路跑完后，可以在 Pod shell 打包当前日志目录。

```sh
node_log_dir="${COMFYUI_RS_NODE_MONITOR_LOG_DIR:-/app/user/node-monitor}"
tar -czf "/tmp/comfyui-node-monitor-$(date +%Y%m%d-%H%M%S).tgz" -C "$node_log_dir" .
```

| 事件 | 如何阅读 |
| --- | --- |
| `start` | 节点开始时的内存快照。 |
| `sample` | 节点执行期间的进程内存采样。进程约每 250 ms 采样，长时间执行的节点约每 2 秒记录一次。 |
| `finish` | 调用耗时、结果、采样次数、观测峰值及结束快照。捕获到的内存不足会带上 OOM 类型。 |

内存快照记录 worker 的 RSS、估算有效可用 RAM，以及可读取时的 cgroup 限额。设备支持时，还会记录张量分配器按设备统计的分配量和保留量。采样观察的是进程和设备计数，异步节点并行执行时可能同时推高计数，采样也可能错过短于间隔的峰值。日志只包含节点标识、状态和测量数据，输入、输出和异常正文无需写入节点监控记录。

如果 Pod 的终止状态显示 `OOMKilled`，从前一次 worker 的日志里用 `worker_run_id` 和 `call_id` 找到已有 `start` 或 `sample` 的未完成调用，并可按 `timestamp_ms` 恢复顺序。这些调用是排查线索。监控仅采集数据，任务失败继续由 ComfyUI 原有逻辑处理。

## 本地验证

```sh
cargo test --locked --all-features
.venv-candle/bin/python -m unittest discover -s tests
```

Python 测试直接加载本仓库中的集成版本。设置 `COMFYUI_RS_COMFYUI_PATH` 可以额外与原版 checkout 对照。原版许可证见 `comfyui/LICENSE`。
