#!/usr/bin/env python3
"""静态检查：确认所有演示脚本都正确接入了统一日志模块。

检查项
------
1. **未定义的名字**：脚本里用了 ``logger.xxx`` 但没有 ``logger = get_logger(...)``；
2. **缺失的导入**：用了 ``log_exception`` 但没有从 ``utils.logging_utils`` 导入它；
3. **未使用的导入**：导入了 ``log_exception`` 但一次都没用（会造成 lint 告警）；
4. **sys.path 深度**：``parents[N]`` 是否真的能定位到 ``src/utils/logging_utils.py``；
5. **语法正确性**：用 ``ast.parse`` 校验。

用法：
    python tools/check_logging.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOGGING_MODULE = "utils.logging_utils"


def check_file(path: Path) -> list[str]:
    """检查单个文件的日志接入情况。

    Args:
        path: 待检查的 .py 文件路径。

    Returns:
        问题列表；无问题时为空列表。
    """
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")

    # --- 5. 语法检查 ---
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return [f"语法错误（第 {exc.lineno} 行）：{exc.msg}"]

    # 不涉及日志的文件直接跳过；
    # logging_utils.py 自身是日志模块的实现，不适用下面这些规则
    if LOGGING_MODULE not in text or path.name == "logging_utils.py":
        return problems

    # --- 1. logger 是否已定义 ---
    if "logger = get_logger(" not in text:
        problems.append("使用了 logger 但没有 `logger = get_logger(__name__)`")

    # --- 2/3. log_exception 的导入与使用 ---
    uses_log_exception = "log_exception(" in text
    imports_log_exception = re.search(
        r"from\s+utils\.logging_utils\s+import\s+[^\n]*\blog_exception\b", text
    )
    if uses_log_exception and not imports_log_exception:
        problems.append("使用了 log_exception 但没有导入它")
    if imports_log_exception and not uses_log_exception:
        problems.append("导入了 log_exception 但从未使用（会造成 lint 告警）")

    # --- 4. sys.path 深度 ---
    match = re.search(r"parents\[(\d+)\]", text)
    if not match:
        problems.append("没有找到 sys.path 注入语句（parents[N]）")
    else:
        depth = int(match.group(1))
        resolved = path.resolve().parent
        for _ in range(depth):
            resolved = resolved.parent
        if not (resolved / "utils" / "logging_utils.py").exists():
            problems.append(
                f"sys.path 深度错误：parents[{depth}] 指向 {resolved}，"
                "定位不到 src/utils/logging_utils.py"
            )

    return problems


def main() -> None:
    """遍历所有演示脚本并汇总检查结果。"""
    total = 0
    failed = 0

    for path in sorted(SRC.rglob("*.py")):
        total += 1
        problems = check_file(path)
        if problems:
            failed += 1
            print(f"[FAIL] {path.relative_to(ROOT)}")
            for problem in problems:
                print(f"       - {problem}")

    print(f"\n共检查 {total} 个文件，{total - failed} 个通过，{failed} 个存在问题。")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
