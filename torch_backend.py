"""验证 ComfyUI 选中的 Candle 环境。"""

import subprocess


CANDLE_PYTHON_FLAGS = ("-E", "-s")

_CHECK_CANDLE_TORCH = """
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
import sys

for stock in ("torch", "torchvision", "torchaudio"):
    try:
        distribution(stock)
    except PackageNotFoundError:
        continue
    raise SystemExit(f"Candle 环境中安装了原版 {stock}。请使用单独的虚拟环境。")

try:
    distribution("candle-nn")
except PackageNotFoundError:
    pass
else:
    raise SystemExit("Candle 环境中安装了独立的 candle-nn。请使用单独的虚拟环境安装自带 Candle 的 candle-torch wheel。")

try:
    backend = distribution("candle-torch")
except PackageNotFoundError:
    if sys.argv[1] == "allow-missing" and all(find_spec(name) is None for name in ("torch", "candle")):
        raise SystemExit(0)
    raise SystemExit("找不到 candle-torch。请先用 build.py 安装到所选环境。")

files = {file.as_posix(): file for file in backend.files or ()}
for name in ("torch", "candle"):
    entry = files.get(f"{name}/__init__.py")
    module = find_spec(name)
    if entry is None or module is None or module.origin is None:
        raise SystemExit(f"找不到 candle-torch wheel 提供的 {name} 包。")
    if Path(module.origin).resolve() != Path(backend.locate_file(entry)).resolve():
        raise SystemExit(f"{name} 导入被其他路径覆盖：{module.origin}")
"""


def verify_candle_environment(python, entry_dir, *, require_wheel=True):
    """使用启动时的模块搜索规则检查包来源，保持导入留给 ComfyUI。"""
    wheel_check = "required" if require_wheel else "allow-missing"
    result = subprocess.run(
        [str(python), *CANDLE_PYTHON_FLAGS, "-c", _CHECK_CANDLE_TORCH, wheel_check],
        cwd=entry_dir,
        check=False,
    )
    return result.returncode == 0
