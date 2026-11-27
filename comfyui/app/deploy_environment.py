import functools
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_DEPLOY_ENV = "local-git"
_ENV_FILENAME = ".comfy_environment"

# 安装标记随启动入口存放，不受 --base-directory 切换用户数据目录的影响。
_COMFY_INSTALL_DIR = os.path.realpath(os.getenv(
    "COMFYUI_RS_INSTALL_ROOT", os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))


@functools.cache
def get_deploy_environment() -> str:
    """从安装目录读取可选的部署标记，供运行环境识别使用。"""
    env_file = os.path.join(_COMFY_INSTALL_DIR, _ENV_FILENAME)
    try:
        with open(env_file, encoding="utf-8") as f:
            # Cap the read so a malformed or maliciously crafted file (e.g.
            # a single huge line with no newline) can't blow up memory.
            first_line = f.readline(128).strip()
            value = "".join(c for c in first_line if 32 <= ord(c) < 127)
            if value:
                return value
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.error("Failed to read %s: %s", env_file, e)

    return _DEFAULT_DEPLOY_ENV
