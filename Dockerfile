ARG PYTHON_IMAGE=python:3.13-slim-bookworm
ARG RUST_IMAGE=rust:1.85-bookworm

FROM ${RUST_IMAGE} AS rust-toolchain

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS build
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_TARGET_DIR=/tmp/comfyui-rs-target \
    PATH="/usr/local/cargo/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=candle_wheels /candle_torch-*.whl /tmp/candle-wheels/
COPY comfyui/requirements.txt comfyui/requirements-candle-compat.txt comfyui/requirements-candle-constraints.txt comfyui/
COPY Cargo.toml Cargo.lock pyproject.toml build.py torch_backend.py ./
COPY src/ src/
COPY python/ python/
RUN python build.py --candle-wheels /tmp/candle-wheels

# 使用刚构建的扩展运行 Rust 和 Python 回归测试。
COPY . .
RUN cargo test --locked --all-features \
    && .venv-candle/bin/python -m unittest discover -s tests \
    && .venv-candle/bin/python -E -s scripts/smoke_candle_workflow.py

FROM base AS runtime
RUN groupadd --gid 1000 comfyui \
    && useradd --uid 1000 --gid 1000 --create-home comfyui
# 将测试过的虚拟环境单独复制，便于复用依赖层。
COPY --from=build --chown=comfyui:comfyui /app/.venv-candle/ /app/.venv-candle/
COPY --chown=comfyui:comfyui . .
RUN install -d -o comfyui -g comfyui \
        /app/models /app/custom_nodes /app/input /app/output /app/user /app/temp /app/previews \
    && install -o comfyui -g comfyui -m 0644 /app/comfyui/input/example.png /app/input/example.png
USER comfyui

# 通过项目入口初始化 CPU worker 和路由，使用临时目录存放验证数据。
RUN python main.py --cpu --disable-assets --disable-all-custom-nodes \
    --base-directory /tmp/comfyui-smoke --quick-test-for-ci \
    && rm -rf /tmp/comfyui-smoke

EXPOSE 8188
ENTRYPOINT ["python", "main.py", "--cpu", "--listen", "0.0.0.0"]
