"""独立评测脚本：跑完整评测套件并生成可视化报告。

与 pytest 的关系
---------------
本脚本**复用 pytest 里的同一批用例**，但输出的是 HTML 报告而不是断言结果。

用途：

- 想看每个指标的分数分布、逐条理由，而不是只看 pass/fail；
- 想把报告存档或分享给同事；
- 想在 CI 里产出评测报告工件。

用法
----
```bash
# 跑全部评测并生成报告
python evals/run_eval.py

# 只跑 RAG 相关评测
python evals/run_eval.py --suite rag

# 指定裁判模型与阈值
python evals/run_eval.py --threshold 0.7 --judge-provider deepeval

# 跳过 LLM 评判（只跑离线确定性用例）
python evals/run_eval.py --skip-llm
```

报告输出到 ``evals/reports/`` 目录。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 把 evals 与 src 加入模块搜索路径
EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(EVALS_DIR.parent / "src"))

from dotenv import load_dotenv  # noqa: E402
from eval_config import judge_provider, threshold  # noqa: E402

from utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

# 报告输出目录
REPORT_DIR = EVALS_DIR / "reports"

# 各套件对应的测试文件
SUITES = {
    "rag": "test_rag_quality.py",
    "agent": "test_agent_tools.py",
    "prompt": "test_prompt_engineering.py",
    "chains": "test_chains.py",
    "memory": "test_memory.py",
    "offline": "test_offline_ingestion.py",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行 DeepEval 评测套件并生成 HTML 报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evals/run_eval.py                    # 跑全部
  python evals/run_eval.py --suite rag        # 只跑 RAG
  python evals/run_eval.py --skip-llm         # 只跑离线确定性用例
  python evals/run_eval.py --threshold 0.7    # 提高通过阈值
""",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        action="append",
        help="只跑指定套件（可重复指定）；默认跑全部",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="指标通过阈值（0~1），会覆盖 EVAL_THRESHOLD",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["deepeval", "openai", "proxy", "ollama"],
        help="裁判模型提供方，会覆盖 EVAL_JUDGE_PROVIDER",
    )
    parser.add_argument(
        "--judge-model",
        help="裁判模型名，会覆盖 EVAL_JUDGE_MODEL",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="跳过所有需要 LLM 评判的用例，只跑离线确定性用例",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_DIR),
        help=f"报告输出目录（默认 {REPORT_DIR}）",
    )
    return parser.parse_args()


def main() -> int:
    """主入口。

    Returns:
        进程退出码：0 表示全部通过，1 表示有失败，2 表示运行出错。
    """
    args = parse_args()

    # 加载 .env（若存在）
    load_dotenv()

    # 命令行参数转成环境变量，供 eval_config 读取
    if args.threshold is not None:
        os.environ["EVAL_THRESHOLD"] = str(args.threshold)
    if args.judge_provider:
        os.environ["EVAL_JUDGE_PROVIDER"] = args.judge_provider
    if args.judge_model:
        os.environ["EVAL_JUDGE_MODEL"] = args.judge_model
    if args.skip_llm:
        os.environ["EVAL_SKIP_LLM"] = "1"

    # 选定要跑的测试文件
    if args.suite:
        selected = [SUITES[name] for name in args.suite]
    else:
        selected = list(SUITES.values())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("评测套件启动")
    logger.info("  测试文件   ：%s", ", ".join(selected))
    logger.info("  裁判模型   ：%s", judge_provider())
    logger.info("  通过阈值   ：%.2f", threshold())
    logger.info("  跳过 LLM   ：%s", args.skip_llm)
    logger.info("  报告目录   ：%s", output_dir)
    logger.info("=" * 60)

    # 延迟导入：确保环境变量设置完成后再加载 pytest / deepeval
    import pytest

    test_paths = [str(EVALS_DIR / name) for name in selected]

    exit_code = pytest.main(
        [
            *test_paths,
            "-v",
            "--tb=short",
            "-p",
            "no:randomly",
        ]
    )

    logger.info("=" * 60)
    if exit_code == 0:
        logger.info("评测全部通过")
    else:
        logger.warning("评测存在失败用例，退出码 %d", exit_code)
    logger.info("=" * 60)

    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
