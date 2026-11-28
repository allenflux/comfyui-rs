"""使用项目构建出的 Python 环境启动内置 ComfyUI。"""

import argparse
import os
from pathlib import Path

from torch_backend import CANDLE_PYTHON_FLAGS, verify_candle_environment


ROOT = Path(__file__).resolve().parent


def main():
    """使用根目录数据布局，并将其余参数按原顺序交给 ComfyUI。"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--torch-backend", choices=("pytorch", "candle"), default="pytorch")
    args, comfyui_args = parser.parse_known_args()

    use_candle = args.torch_backend == "candle"
    default_env = ROOT / (".venv-candle" if use_candle else ".venv")
    venv = (args.venv or default_env).expanduser().resolve()
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        parser.exit(1, f"找不到运行环境 {python}，请先执行 python3 build.py。\n")

    entry = ROOT / "comfyui" / "main.py"
    flags = CANDLE_PYTHON_FLAGS if use_candle else ()
    if use_candle and not verify_candle_environment(python, entry.parent):
        parser.exit(1)

    # 保留调用者的工作目录供相对参数使用，由 ComfyUI 进程直接接收信号。
    os.environ["COMFYUI_RS_INSTALL_ROOT"] = str(ROOT)
    os.execv(python, [str(python), *flags, str(entry), *comfyui_args])


if __name__ == "__main__":
    main()
