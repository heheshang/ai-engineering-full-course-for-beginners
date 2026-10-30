"""统一日志工具：给各个演示脚本提供「开箱即用」的中文日志。

设计目标
--------
1. **零配置即可用**：脚本里只要 ``logger = get_logger(__name__)``，
   就能同时输出到控制台和日志文件，不需要每个脚本重复写 handler 配置。
2. **级别可配**：通过环境变量控制，避免为了调试而改代码。
3. **不重复输出**：重复调用 ``get_logger`` 不会叠加 handler，
   否则日志会出现多份相同内容。
4. **异常也能落盘**：脚本抛异常时，通过 ``log_exception`` 打印完整堆栈，
   方便复现问题时定位。

环境变量
--------
``LOG_LEVEL``
    日志级别，默认 ``INFO``。可选：``DEBUG`` / ``INFO`` / ``WARNING`` /
    ``ERROR`` / ``CRITICAL``。
``LOG_DIR``
    日志文件目录，默认项目根目录下的 ``logs``。
``LOG_FILE_ENABLED``
    设为 ``0`` / ``false`` 时只输出到控制台，不写文件（CI 环境常用）。
``LOG_TO_STDOUT``
    设为 ``0`` / ``false`` 时不输出到控制台（后台服务常用）。

日志格式
--------
    2026-10-30 12:00:00 | INFO     | src.01-rag-demo.RAGDemo | 文档切分完成，共 8 个片段

各字段依次为：时间 | 级别 | logger 名称 | 日志正文。

使用示例
--------
    from utils.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("开始加载文档：%s", path)
    try:
        ...
    except Exception:
        logger.exception("加载文档失败：%s", path)
        raise
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 日志格式：时间 | 级别 | logger 名 | 正文
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# 时间格式精确到秒，够用且不冗长
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 允许通过环境变量关闭的布尔开关
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

# 已配置过的 logger 名称，避免重复添加 handler 导致日志重复
_CONFIGURED: set[str] = set()


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔型环境变量。

    Args:
        name: 环境变量名。
        default: 未设置或取值无法识别时的默认值。

    Returns:
        布尔值。取值为 "1/true/yes/on" 视为 True，"0/false/no/off" 视为 False，
        其余（未设置、大小写之外的写法）返回 ``default``。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return default


def _resolve_level() -> int:
    """根据 ``LOG_LEVEL`` 环境变量解析日志级别。

    Returns:
        logging 模块的级别常量。解析失败时回退到 ``logging.INFO``。
    """
    raw = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = logging.getLevelName(raw)
    if isinstance(level, int):
        return level
    return logging.INFO


def _resolve_log_dir() -> Path:
    """确定日志文件输出目录。

    默认取「项目根目录 / logs」，即当前文件的上两级目录
    （``src/utils/logging_utils.py`` -> ``src/utils`` -> ``src`` -> 项目根）。
    这样无论从哪个工作目录运行脚本，日志都落在同一个地方。
    """
    default_dir = Path(__file__).resolve().parents[2] / "logs"
    raw = os.getenv("LOG_DIR")
    if not raw:
        return default_dir
    return Path(raw).expanduser()


def _build_file_handler(level: int) -> logging.Handler | None:
    """创建写入文件的 handler。

    Args:
        level: 该 handler 记录的最低级别。

    Returns:
        ``RotatingFileHandler``；若日志目录不可写则返回 ``None``（降级为仅控制台）。
    """
    log_dir = _resolve_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # 按天切分，单文件上限 5MB，最多保留 5 个历史文件，避免磁盘被撑满
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            log_dir / "course.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        return handler
    except OSError as exc:  # 磁盘只读 / 权限不足等情况：降级为仅控制台
        print(f"[logging] 无法写入日志文件（{exc}），仅输出到控制台", file=sys.stderr)
        return None


def get_logger(name: str | None = None) -> logging.Logger:
    """获取一个已配置好的 logger。

    Args:
        name: logger 名称，通常传 ``__name__``，便于在日志里定位到具体脚本。

    Returns:
        配置完成的 ``logging.Logger`` 实例。
        重复传入同一个 name 不会重复添加 handler。
    """
    logger = logging.getLogger(name or "course")

    # 已经配置过就直接复用，避免日志重复输出
    if name in _CONFIGURED:
        return logger

    level = _resolve_level()
    logger.setLevel(level)
    # 避免日志向上冒泡到 root logger 被重复打印
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # 控制台输出：用 stderr 通道之外的 stdout，保证与演示脚本的 print 输出分离
    if _env_flag("LOG_TO_STDOUT", True):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

    # 文件输出：可选
    if _env_flag("LOG_FILE_ENABLED", True):
        file_handler = _build_file_handler(level)
        if file_handler is not None:
            logger.addHandler(file_handler)

    _CONFIGURED.add(name or "course")
    logger.debug("日志初始化完成：name=%s, level=%s", logger.name, logging.getLevelName(level))
    return logger


def log_exception(logger: logging.Logger, message: str, *args: object) -> None:
    """记录异常并打印完整堆栈。

    相比 ``logger.exception``，这里允许调用方传入自定义前缀信息，
    便于在日志里快速定位是哪个环节失败。

    Args:
        logger: 目标 logger。
        message: 日志前缀消息，可使用 ``%s`` 占位符。
        *args: 与 ``message`` 占位符对应的参数。
    """
    logger.error(message, *args)
    logger.exception("异常详情（完整堆栈见下）")


__all__ = ["LOG_DATE_FORMAT", "LOG_FORMAT", "get_logger", "log_exception"]
