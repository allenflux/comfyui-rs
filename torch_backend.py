"""验证 ComfyUI 选中的 Candle 环境。"""

import subprocess


CANDLE_PYTHON_FLAGS = ("-E", "-s")

_CHECK_CANDLE_TORCH = """
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path

try:
    distribution("torch")
except PackageNotFoundError:
    pass
else:
    raise SystemExit("Candle 环境中安装了原版 torch。请使用单独的虚拟环境。")

try:
    backend = distribution("candle-torch")
except PackageNotFoundError:
    raise SystemExit("找不到 candle-torch。请先在所选环境中安装 candle-torch wheel。")

entry = next((file for file in backend.files or () if file.as_posix() == "torch/__init__.py"), None)
module = find_spec("torch")
if entry is None or module is None or module.origin is None:
    raise SystemExit("找不到 candle-torch wheel 提供的 torch 包。")
if Path(module.origin).resolve() != Path(backend.locate_file(entry)).resolve():
    raise SystemExit(f"torch 导入被其他路径覆盖：{module.origin}")
"""


def verify_candle_environment(python, entry_dir):
    """使用启动时的模块搜索规则检查包来源，保持导入留给 ComfyUI。"""
    result = subprocess.run(
        [str(python), *CANDLE_PYTHON_FLAGS, "-c", _CHECK_CANDLE_TORCH],
        cwd=entry_dir,
        check=False,
    )
    return result.returncode == 0
