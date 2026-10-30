#!/usr/bin/env python3
"""为指定的演示脚本批量接入统一日志（一次性辅助工具，非项目运行代码）。

用途：
    给尚未接入 `utils.logging_utils` 的演示脚本补上：
      1. sys.path 注入（保证任意工作目录都能 import 到 src/utils）；
      2. logger 初始化。

实现要点：
    - 插入位置：模块 docstring / ``from __future__`` 之后，
      **整个 import 区之后**（避免 import 出现在文件中间）；
    - 注入后自动做一次语法校验（ast.parse），失败会明确报出。

用法：
    python tools/add_logging.py <file1.py> <file2.py> ...
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# 要注入的 import 片段
IMPORT_BLOCK = '''import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
'''

LOGGER_BLOCK = '''from utils.logging_utils import get_logger{extra}

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)
'''


def _module_docstring_end(lines: list[str]) -> int:
    """返回模块 docstring 结束后的行索引（0-based，指向下一行）。

    Args:
        lines: 文件所有行。

    Returns:
        docstring 结束后的行索引；若文件没有 docstring 则返回 0。
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(('"""', "'''", 'r"""', "r'''")):
            quote = stripped[:3]
            if stripped.count(quote) >= 2:
                return i + 1
            for j in range(i + 1, len(lines)):
                if quote in lines[j]:
                    return j + 1
        break
    return 0


def _import_section_end(lines: list[str], start: int) -> int:
    """返回整个 import 区结束后的行索引。

    扫描过程中允许出现空行、注释行，但一旦遇到非 import 的语句就停止。

    Args:
        lines: 文件所有行。
        start: 开始扫描的位置。

    Returns:
        import 区之后的行索引。
    """
    i = start
    last_import_end = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if stripped.startswith(("import ", "from ")):
            # 多行括号导入：一直吃到右括号为止
            if "(" in stripped and ")" not in stripped:
                while i < len(lines) and ")" not in lines[i]:
                    i += 1
            last_import_end = i + 1
            i += 1
            continue
        # 遇到第一个非 import 语句就结束
        break
    return last_import_end


def inject_imports(text: str) -> tuple[str, bool]:
    """在 import 区之后插入 sys.path 注入与 logger 初始化。

    Args:
        text: 源文件内容。

    Returns:
        (处理后的内容, 是否发生了修改)
    """
    if "utils.logging_utils" in text:
        return text, False

    lines = text.split("\n")

    insert_at = _module_docstring_end(lines)
    insert_at = _import_section_end(lines, insert_at)

    extra = ", log_exception" if "log_exception(" in text else ""
    block = [
        "",
        IMPORT_BLOCK.rstrip("\n"),
        "",
        LOGGER_BLOCK.format(extra=extra).rstrip("\n"),
        "",
    ]

    lines = lines[:insert_at] + block + lines[insert_at:]
    return "\n".join(lines), True


def check_syntax(path: Path) -> str | None:
    """校验文件语法是否正确。

    Args:
        path: 待校验的文件路径。

    Returns:
        出错信息；语法正确时返回 None。
    """
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return f"语法错误（第 {exc.lineno} 行）：{exc.msg}"
    return None


def main() -> None:
    """处理命令行传入的所有文件。"""
    if len(sys.argv) < 2:
        print("用法：python tools/add_logging.py <file1.py> [file2.py ...]")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        original = path.read_text(encoding="utf-8")
        updated, changed = inject_imports(original)
        if not changed:
            print(f"[SKIP] {path}（已接入或无 import 语句）")
            continue

        path.write_text(updated, encoding="utf-8")
        error = check_syntax(path)
        if error:
            print(f"[FAIL] {path} -> {error}")
        else:
            print(f"[OK]   {path}")


if __name__ == "__main__":
    main()
