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
COMPAT_REQUIREMENTS = REQUIREMENTS.with_name("requirements-candle-compat.txt")


def main():
    """在同一个虚拟环境中安装 ComfyUI 及其使用的 Rust 扩展。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        type=Path,
        metavar="PATH",
        help="Python 虚拟环境路径，默认为项目目录下的 .venv-candle",
    )
    parser.add_argument(
        "--torch-backend",
        choices=("candle",),
        default="candle",
    )
    parser.add_argument(
        "--candle-wheels",
        type=Path,
        metavar="DIR",
        help="用本地 candle-torch wheel 替代软件包索引中的版本",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="复用已配置的环境，跳过 ComfyUI 依赖安装，仍构建并安装 Rust 扩展",
    )
    args = parser.parse_args()
    wheels = args.candle_wheels.expanduser().resolve() if args.candle_wheels else None
    if wheels is not None and not wheels.is_dir():
        parser.error(f"找不到 Candle wheel 目录 {wheels}")
    if wheels is not None and args.no_deps:
        parser.error("--candle-wheels 和 --no-deps 不能同时使用")
    if not args.no_deps:
        for requirements in (REQUIREMENTS, COMPAT_REQUIREMENTS):
            if not requirements.is_file():
                parser.error(f"找不到随项目提供的依赖文件 {requirements}")

    env_path = (args.venv or ROOT / ".venv-candle").expanduser().resolve()
    python = env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        if args.no_deps:
            parser.error(f"--no-deps 需要已有的 Candle 环境 {env_path}")
        if sys.version_info < (3, 13):
            parser.error("创建 Candle 环境需要 Python 3.13 或更新版本")
        print(f"创建 Python 虚拟环境 {env_path}", flush=True)
        venv.create(env_path, with_pip=True)
    if not (env_path / "pyvenv.cfg").is_file():
        parser.error(f"{env_path} 缺少 pyvenv.cfg，请提供 Python 虚拟环境路径")
    version_check = subprocess.run(
        [str(python), "-c", "import sys; sys.exit(sys.version_info < (3, 13))"]
    )
    if version_check.returncode != 0:
        parser.error(f"{python} 必须使用 Python 3.13 或更新版本")
    if not verify_candle_environment(python, REQUIREMENTS.parent, require_wheel=args.no_deps):
        parser.exit(1)

    pip = [str(python), "-m", "pip"]
    if subprocess.run([*pip, "--version"], capture_output=True).returncode != 0:
        # 一些已有的虚拟环境没有 pip，用该解释器自带的 ensurepip 初始化。
        subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
    if wheels is not None:
        subprocess.run(
            [*pip, "install", "--force-reinstall", "--no-deps", "--no-index", "--only-binary", ":all:",
             "--find-links", str(wheels), "candle-torch"],
            check=True,
        )
    if not args.no_deps:
        print("安装 ComfyUI 依赖", flush=True)
        subprocess.run(
            [*pip, "install", "--only-binary", "candle-torch", "-r", str(REQUIREMENTS)],
            cwd=REQUIREMENTS.parent,
            check=True,
        )
        subprocess.run(
            [*pip, "install", "--no-deps", "-r", str(COMPAT_REQUIREMENTS)],
            cwd=COMPAT_REQUIREMENTS.parent,
            check=True,
        )
        if not verify_candle_environment(python, REQUIREMENTS.parent):
            parser.exit(1)

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
