"""使用项目构建出的 Python 环境启动内置 ComfyUI。"""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    """使用根目录数据布局，并将其余参数按原顺序交给 ComfyUI。"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    args, comfyui_args = parser.parse_known_args()

    venv = args.venv.expanduser().resolve()
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        parser.exit(1, f"找不到运行环境 {python}，请先执行 python3 build.py。\n")

    # 保留调用者的工作目录供相对参数使用，由 ComfyUI 进程直接接收信号。
    os.environ["COMFYUI_RS_INSTALL_ROOT"] = str(ROOT)
    os.execv(python, [str(python), str(ROOT / "comfyui" / "main.py"), *comfyui_args])


if __name__ == "__main__":
    main()
