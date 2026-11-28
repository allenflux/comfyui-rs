"""构建 ComfyUI 和 Rust 扩展共用的 Python 环境。"""

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

from torch_backend import verify_candle_environment


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "comfyui" / "requirements.txt"


def main():
    """在同一个虚拟环境中安装 ComfyUI 及其使用的 Rust 扩展。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        type=Path,
        metavar="PATH",
        help="Python 虚拟环境路径，默认为项目目录下的 .venv 或 .venv-candle",
    )
    parser.add_argument(
        "--torch-backend",
        choices=("pytorch", "candle"),
        default="pytorch",
        help="candle 使用已配置好兼容依赖的环境",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="复用已配置的环境，跳过 ComfyUI 依赖安装，仍构建并安装 Rust 扩展",
    )
    args = parser.parse_args()
    if not REQUIREMENTS.is_file():
        parser.error(f"找不到随项目提供的依赖文件 {REQUIREMENTS}")

    use_candle = args.torch_backend == "candle"
    default_env = ROOT / (".venv-candle" if use_candle else ".venv")
    env_path = (args.venv or default_env).expanduser().resolve()
    python = env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        if use_candle:
            parser.error(f"Candle 后端需要已有的虚拟环境 {env_path}")
        if args.no_deps:
            parser.error("--no-deps 需要已有的 Python 虚拟环境")
        if sys.version_info < (3, 10):
            parser.error("创建环境需要 Python 3.10 或更新版本")
        print(f"创建 Python 虚拟环境 {env_path}", flush=True)
        venv.create(env_path, with_pip=True)
    if not (env_path / "pyvenv.cfg").is_file():
        parser.error(f"{env_path} 缺少 pyvenv.cfg，请提供 Python 虚拟环境路径")
    version_check = subprocess.run(
        [str(python), "-c", "import sys; sys.exit(sys.version_info < (3, 10))"]
    )
    if version_check.returncode != 0:
        parser.error(f"{python} 必须使用 Python 3.10 或更新版本")
    if use_candle and not verify_candle_environment(python, REQUIREMENTS.parent):
        parser.exit(1)

    pip = [str(python), "-m", "pip"]
    if subprocess.run([*pip, "--version"], capture_output=True).returncode != 0:
        # 一些已有的虚拟环境没有 pip，用该解释器自带的 ensurepip 初始化。
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
    if not use_candle and not args.no_deps:
        print("安装 ComfyUI 依赖", flush=True)
        subprocess.run(
            [*pip, "install", "-r", str(REQUIREMENTS)],
            cwd=REQUIREMENTS.parent,
            check=True,
        )

    print("构建并安装 comfyui-rs 原生扩展", flush=True)
    # 强制重新安装，源码有改动但版本号相同时也会重新构建。
    subprocess.run(
        [*pip, "install", "--no-deps", "--force-reinstall", str(ROOT)], check=True
    )
    subprocess.run([str(python), "-c", "import comfyui_rs._native"], check=True)
    print(f"构建完成，使用 Python {python}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"构建命令执行失败（退出码 {error.returncode}）。", file=sys.stderr)
        raise SystemExit(error.returncode if error.returncode > 0 else 1) from None
