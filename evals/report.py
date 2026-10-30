"""评测报告生成：把 pytest 结果与 DeepEval 指标汇总成可归档的报告。

产物（全部写入 ``evals/reports/``）
--------------------------------
============================  ==========================================
文件                          内容
============================  ==========================================
``test_run_<ts>.json``        DeepEval 原生 test run（LLM 评判用例的指标明细）
``evaluation_<ts>.html``      自包含 HTML 报告（用例总览 + 指标明细）
``evaluation_<ts>.md``        同内容的 Markdown 版
``latest.html``               最近一次 HTML 报告的副本，方便直接打开
============================  ==========================================

为什么不用 DeepEval 自带的 HTML 导出
------------------------------------
DeepEval 的 ``export_to_html`` 只覆盖 **LLM 评判** 的用例；
而本项目评测套件里还有一整组**离线确定性用例**
（``test_offline_ingestion.py``），它们用普通 ``assert`` 断言，
根本不进入 DeepEval 的 test run。

如果直接用 DeepEval 的导出，离线用例会在报告里彻底消失，
让人误以为「只跑了 LLM 评判那几条」。所以这里自己渲染报告，
把两类用例放在同一张总览表里。

数据从哪来
----------
- **用例结果**：由 ``conftest.py`` 的 ``pytest_runtest_makereport`` 钩子
  逐条回调 :func:`record_report` 采集（与 pytest 内部状态机无关，
  因此 xdist 并行执行时也能正确汇总）。
- **指标明细**：从 DeepEval 的 ``global_test_run_manager`` 里取，
  需要先让 DeepEval 处于「运行中」状态（即 ``DEEPEVAL=1``），
  见 :func:`configure_reporting`。
"""

from __future__ import annotations

import html
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# evals/ 目录
EVALS_DIR = Path(__file__).resolve().parent

# 报告输出目录。可以用环境变量 ``EVAL_REPORT_DIR`` 覆盖
# （``run_eval.py --output-dir`` 就是通过它实现的）。
REPORT_DIR = Path(os.getenv("EVAL_REPORT_DIR") or (EVALS_DIR / "reports"))

# 状态常量
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# 失败详情最多保留的字符数（避免超长 traceback 撑爆报告）
MAX_MESSAGE_CHARS = 1500


# --------------------------------------------------------------------------- #
# 一、用例结果采集
# --------------------------------------------------------------------------- #
@dataclass
class TestOutcome:
    """单个 pytest 用例的执行结果。"""

    nodeid: str
    suite: str
    name: str
    doc: str
    status: str = STATUS_PASSED
    duration: float = 0.0
    message: str = ""

    @property
    def status_label(self) -> str:
        """中文状态标签。"""
        return {
            STATUS_PASSED: "通过",
            STATUS_FAILED: "失败",
            STATUS_SKIPPED: "跳过",
        }.get(self.status, self.status)


@dataclass
class OutcomeRegistry:
    """用例结果登记表。

    以 ``nodeid`` 为键，保证同一用例只记录一条；
    setup / call / teardown 三个阶段的结果合并到同一条记录里。
    """

    outcomes: dict[str, TestOutcome] = field(default_factory=dict)

    def record(self, item, report) -> None:
        """记录一次 pytest 阶段报告。

        Args:
            item: pytest 的 ``Item`` 对象。
            report: pytest 的阶段报告（``TestReport``）。
        """
        nodeid = report.nodeid
        entry = self.outcomes.get(nodeid)
        if entry is None:
            entry = TestOutcome(
                nodeid=nodeid,
                suite=_suite_of(nodeid),
                name=_short_name(nodeid),
                doc=_doc_of(item),
            )
            self.outcomes[nodeid] = entry

        entry.duration += report.duration

        if report.when == "setup":
            # setup 阶段失败（fixture 报错）→ 用例失败
            if report.outcome == "failed":
                entry.status = STATUS_FAILED
                entry.message = _shorten(report.longreprtext)
            elif report.outcome == "skipped":
                # setup 阶段跳过：通常是 fixture 里主动 skip
                entry.status = STATUS_SKIPPED
                entry.message = _skip_reason(report)
        elif report.when == "call":
            # call 阶段的三种结果：passed / failed / skipped
            # （pytest.skip() 写在用例函数体里时，skip 发生在 **call** 阶段，
            #  不是 setup 阶段——两种都要处理，否则跳过的用例会被误判为失败）
            if report.outcome == "passed":
                entry.status = STATUS_PASSED
            elif report.outcome == "skipped":
                entry.status = STATUS_SKIPPED
                entry.message = _skip_reason(report)
            else:
                entry.status = STATUS_FAILED
                entry.message = _shorten(report.longreprtext)
        elif (
            report.when == "teardown"
            and report.outcome == "failed"
            # teardown 报错也算用例失败（资源清理没成功），
            # 但不要覆盖掉「本来就跳过」的用例。
            and entry.status != STATUS_SKIPPED
        ):
            entry.status = STATUS_FAILED
            entry.message = _shorten(report.longreprtext)

    def as_list(self) -> list[TestOutcome]:
        """按 nodeid 排序返回所有结果。"""
        return sorted(self.outcomes.values(), key=lambda o: o.nodeid)

    def summary(self) -> dict[str, int]:
        """统计各状态的数量。"""
        counts = {STATUS_PASSED: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
        for outcome in self.outcomes.values():
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        return counts


# 全局登记表（一个 pytest 会话一份）
REGISTRY = OutcomeRegistry()


def record_report(item, report) -> None:
    """采集单条 pytest 阶段报告（供 ``conftest.py`` 的钩子调用）。

    Args:
        item: pytest 的 ``Item`` 对象。
        report: pytest 的阶段报告。
    """
    REGISTRY.record(item, report)


def _suite_of(nodeid: str) -> str:
    """从 nodeid 推断所属测试文件（作为套件名）。"""
    path = nodeid.split("::", 1)[0]
    stem = Path(path).stem
    return stem if stem else path


def _short_name(nodeid: str) -> str:
    """取 nodeid 的最后一段（用例函数名）。"""
    parts = nodeid.split("::")
    return parts[-1] if parts else nodeid


def _doc_of(item) -> str:
    """取用例的 docstring 首行，作为简短说明。"""
    function = getattr(item, "function", None)
    doc = getattr(function, "__doc__", None)
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _skip_reason(report) -> str:
    """提取跳过原因。"""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return _shorten(str(longrepr)) if longrepr else ""


def _shorten(text: str | None) -> str:
    """截断过长的文本。"""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[:MAX_MESSAGE_CHARS] + "\n... （已截断）"


# --------------------------------------------------------------------------- #
# 二、DeepEval 侧配置与指标提取
# --------------------------------------------------------------------------- #
def configure_reporting(output_dir: Path, identifier: str | None = None) -> None:
    """让 DeepEval 进入「运行中」状态并把原生 test run 写到报告目录。

    **必须在 pytest 启动之前调用**，因为 DeepEval 的 pytest 插件在
    ``pytest_sessionstart`` 阶段就会根据 ``DEEPEVAL=1`` 决定是否创建
    test run；测试运行结束后再由 :func:`finalize_reports` 落盘。

    Args:
        output_dir: 报告输出目录。
        identifier: 本次运行的标识，会写进 DeepEval 的 test run。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # DeepEval 的 pytest 插件靠这个环境变量判断「正在评测中」
    os.environ["DEEPEVAL"] = "1"
    # 原生 test run（test_run_<ts>.json）的落盘目录
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = str(output_dir)

    if identifier:
        try:
            from deepeval.utils import set_identifier

            set_identifier(identifier)
        except Exception:  # pragma: no cover - deepeval 内部 API 变化时兜底
            pass


def collect_metric_rows(test_run) -> list[dict]:
    """从 DeepEval 的 test run 中提取指标明细。

    Args:
        test_run: ``deepeval.test_run.test_run.TestRun`` 实例（可能为空）。

    Returns:
        每个指标一行的字典列表，字段为
        ``case`` / ``metric`` / ``score`` / ``threshold`` / ``status`` /
        ``reason`` / ``model``。
    """
    rows: list[dict] = []
    if test_run is None:
        return rows

    cases = list(test_run.test_cases or []) + list(test_run.conversational_test_cases or [])
    for case in cases:
        for metric in case.metrics_data or []:
            if metric.error:
                status = "error"
            elif metric.success is None:
                status = "skipped"
            else:
                status = STATUS_PASSED if metric.success else STATUS_FAILED
            rows.append(
                {
                    "case": case.name,
                    "metric": metric.name,
                    "score": metric.score,
                    "threshold": metric.threshold,
                    "status": status,
                    "reason": metric.reason or metric.error or "",
                    "model": metric.evaluation_model or "",
                }
            )
    return rows


def summarize_metrics(metric_rows: list[dict]) -> list[dict]:
    """按指标名聚合评分，得到「各个指标的评分」汇总。

    DeepEval 的 test run 里同时有**逐条明细**（每个用例每个指标一行）和
    **聚合结果**（``test_run.metrics_scores``）。这里自己从明细聚合，
    是因为明细里还带着阈值与状态，能算出更有意义的「通过率」。

    Args:
        metric_rows: :func:`collect_metric_rows` 的输出。

    Returns:
        每个指标一行的字典列表，按平均分从低到高排序（低分排前面，
        便于第一时间发现问题）。字段为 ``metric`` / ``count`` / ``avg`` /
        ``min`` / ``max`` / ``passes`` / ``fails`` / ``errors`` /
        ``pass_rate`` / ``threshold``。
    """
    buckets: dict[str, list[dict]] = {}
    for row in metric_rows:
        buckets.setdefault(row["metric"], []).append(row)

    summaries: list[dict] = []
    for metric_name, rows in buckets.items():
        scores = [r["score"] for r in rows if isinstance(r["score"], (int, float))]
        passes = sum(1 for r in rows if r["status"] == STATUS_PASSED)
        fails = sum(1 for r in rows if r["status"] == STATUS_FAILED)
        errors = sum(1 for r in rows if r["status"] == "error")
        verdicts = passes + fails

        # 阈值取该指标出现过的第一个非空阈值
        threshold = next((r["threshold"] for r in rows if r["threshold"] is not None), None)

        summaries.append(
            {
                "metric": metric_name,
                "count": len(rows),
                "avg": (sum(scores) / len(scores)) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "passes": passes,
                "fails": fails,
                "errors": errors,
                "pass_rate": (passes / verdicts * 100) if verdicts else None,
                "threshold": threshold,
                # 保留样本明细，供 HTML 画分数分布条
                "_samples": [
                    {"score": r["score"], "status": r["status"], "metric": r["metric"]}
                    for r in rows
                ],
            }
        )

    # 平均分低的排前面；没有分数的排最后
    summaries.sort(key=lambda s: (s["avg"] is None, s["avg"] if s["avg"] is not None else 0))
    return summaries


def metric_score_map(metric_rows: list[dict]) -> dict[str, list[dict]]:
    """按用例名归拢该用例产出的所有指标评分。

    用例名取 DeepEval 的 test case 名，与 pytest 的用例函数名一致
    （DeepEval 通过 ``PYTEST_RUN_TEST_NAME`` 拿到 nodeid 的最后一段）。

    Args:
        metric_rows: :func:`collect_metric_rows` 的输出。

    Returns:
        ``用例名 -> 该用例的指标行列表``。
    """
    mapping: dict[str, list[dict]] = {}
    for row in metric_rows:
        mapping.setdefault(row["case"], []).append(row)
    return mapping


def finalize_reports(
    output_dir: Path,
    identifier: str | None = None,
    duration: float = 0.0,
    extra: dict | None = None,
) -> dict[str, Path | None]:
    """收尾：落盘 DeepEval 原生 test run，并生成 HTML / Markdown 报告。

    Args:
        output_dir: 报告输出目录。
        identifier: 本次运行标识。
        duration: 本次运行耗时（秒）。
        extra: 附加到报告头部的元信息（如退出码、裁判模型等）。

    Returns:
        ``{"json": ..., "html": ..., "markdown": ...}`` 各产物路径；
        未生成时对应值为 ``None``。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    test_run = _current_test_run()
    json_path = _save_native_test_run(test_run, duration, identifier=identifier)

    outcomes = REGISTRY.as_list()
    metric_rows = collect_metric_rows(test_run)
    metric_summaries = summarize_metrics(metric_rows)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    header = {
        "identifier": identifier or "course",
        "timestamp": timestamp,
        "duration": duration,
        **(extra or {}),
    }

    html_path = output_dir / f"evaluation_{timestamp}.html"
    html_path.write_text(
        render_html(outcomes, metric_rows, metric_summaries, header),
        encoding="utf-8",
    )

    md_path = output_dir / f"evaluation_{timestamp}.md"
    md_path.write_text(
        render_markdown(outcomes, metric_rows, metric_summaries, header),
        encoding="utf-8",
    )

    # 便于直接打开「最近一次报告」
    latest = output_dir / "latest.html"
    shutil.copyfile(html_path, latest)

    return {"json": json_path, "html": html_path, "markdown": md_path}


def _current_test_run():
    """取当前 DeepEval test run（取不到就返回 None）。"""
    try:
        from deepeval.test_run import global_test_run_manager

        return global_test_run_manager.get_test_run()
    except Exception:  # pragma: no cover - deepeval 未安装或内部异常
        return None


def _save_native_test_run(test_run, duration: float, identifier: str | None = None) -> Path | None:
    """让 DeepEval 把原生 test run 落盘到报告目录。

    DeepEval 的 ``wrap_up_test_run`` 会同时完成：统计通过/失败数、
    写 ``test_run_<ts>.json``、清理临时文件。

    Args:
        test_run: 当前 test run。
        duration: 本次运行耗时（秒）。
        identifier: 本次运行标识；DeepEval 的 ``set_identifier`` 走的是
            ``Settings``（非持久化），而 test run 在创建时就把 identifier
            存下来了，所以这里额外补一次。

    Returns:
        落盘的 JSON 路径；没有可落盘内容时返回 ``None``。
    """
    if test_run is None:
        return None
    if not (test_run.test_cases or test_run.conversational_test_cases):
        # 没有任何 LLM 评判用例（例如 --skip-llm），没有可落盘的 test run
        return None

    if identifier and not test_run.identifier:
        test_run.identifier = identifier

    try:
        from deepeval.test_run import global_test_run_manager

        global_test_run_manager.wrap_up_test_run(duration, display_table=False)
    except Exception as exc:  # pragma: no cover - 落盘失败不应中断评测
        print(f"Warning: failed to finalize deepeval test run: {exc}", flush=True)

    folder = os.environ.get("DEEPEVAL_RESULTS_FOLDER")
    if not folder:
        return None

    candidates = sorted(Path(folder).glob("test_run_*.json"))
    return candidates[-1] if candidates else None


# --------------------------------------------------------------------------- #
# 三、渲染
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: #f6f7f9; color: #1f2328; line-height: 1.6;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 1px solid #e1e4e8; }
.sub { color: #656d76; font-size: 13px; margin-bottom: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; }
.card { flex: 1 1 150px; background: #fff; border: 1px solid #e1e4e8;
        border-radius: 8px; padding: 14px 16px; }
.card .num { font-size: 26px; font-weight: 600; }
.card .lbl { font-size: 12px; color: #656d76; }
.ok   { color: #1a7f37; }
.bad  { color: #cf222e; }
.skip { color: #9a6700; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden;
        font-size: 13px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #eaeef2;
         vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
code { background: #f0f2f4; padding: 1px 5px; border-radius: 4px;
       font-size: 12px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 12px;
         font-size: 12px; font-weight: 600; }
.badge.passed  { background: #dafbe1; color: #1a7f37; }
.badge.failed  { background: #ffebe9; color: #cf222e; }
.badge.skipped { background: #fff8c5; color: #9a6700; }
.badge.error   { background: #ffebe9; color: #cf222e; }
.reason { color: #656d76; font-size: 12px; margin-top: 4px;
          white-space: pre-wrap; word-break: break-word; }
.num-cell { white-space: nowrap; font-variant-numeric: tabular-nums; }
.score-cell { font-weight: 600; }
.muted { color: #8b949e; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { display: inline-flex; align-items: center; gap: 5px;
        padding: 1px 7px; border-radius: 10px; font-size: 11px;
        border: 1px solid transparent; white-space: nowrap; }
.chip-metric { color: #656d76; }
.chip-score { font-weight: 700; font-variant-numeric: tabular-nums; }
.chip.passed { background: #dafbe1; border-color: #aceebb; }
.chip.passed .chip-score { color: #1a7f37; }
.chip.failed { background: #ffebe9; border-color: #ffcecb; }
.chip.failed .chip-score { color: #cf222e; }
.chip.error  { background: #ffebe9; border-color: #ffcecb; }
.chip.error .chip-score { color: #cf222e; }
.chip.skipped { background: #fff8c5; border-color: #f3e5a3; }
.chip.skipped .chip-score { color: #9a6700; }
.bars { display: flex; gap: 2px; align-items: flex-end; height: 20px; }
.bars .bar { display: inline-block; height: 100%; border-radius: 2px;
             min-width: 6px; opacity: 0.85; }
footer { margin-top: 36px; color: #8b949e; font-size: 12px;
         border-top: 1px solid #e1e4e8; padding-top: 12px; }
""".strip()


def _esc(value) -> str:
    """HTML 转义。"""
    return html.escape("" if value is None else str(value))


def _fmt_score(value) -> str:
    """格式化分数。"""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"


def _fmt_duration(seconds: float) -> str:
    """格式化耗时。"""
    if not seconds:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def _status_badge(status: str) -> str:
    """生成状态徽章。"""
    return f'<span class="badge {_esc(status)}">{_esc(status)}</span>'


def _case_scores_html(rows: list[dict]) -> str:
    """渲染某个用例产出的指标评分徽章。

    Args:
        rows: 该用例对应的指标行（:func:`metric_score_map` 的结果）。

    Returns:
        HTML 片段；无指标时返回占位符。
    """
    if not rows:
        return '<span class="muted">—</span>'
    chips = []
    for row in rows:
        score = row["score"]
        score_text = _fmt_score(score) if score is not None else "N/A"
        cls = {
            STATUS_PASSED: "passed",
            STATUS_FAILED: "failed",
            "error": "error",
        }.get(row["status"], "skipped")
        chips.append(
            f'<span class="chip {cls}" title="{_esc(row["metric"])}">'
            f'<span class="chip-metric">{_esc(row["metric"])}</span>'
            f'<span class="chip-score">{_esc(score_text)}</span></span>'
        )
    return f'<div class="chips">{"".join(chips)}</div>'


def render_html(
    outcomes: list[TestOutcome],
    metric_rows: list[dict],
    metric_summaries: list[dict],
    header: dict,
) -> str:
    """渲染 HTML 报告。

    Args:
        outcomes: 用例结果列表。
        metric_rows: DeepEval 指标明细。
        metric_summaries: 按指标聚合的评分汇总。
        header: 报告头部元信息。

    Returns:
        完整的 HTML 文档字符串。
    """
    counts = {STATUS_PASSED: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    total = len(outcomes)
    pass_rate = (counts[STATUS_PASSED] / total * 100) if total else 0.0

    # 全部指标的平均分（所有有分数的指标取均值）
    scored = [s["avg"] for s in metric_summaries if s["avg"] is not None]
    overall_avg = (sum(scored) / len(scored)) if scored else None

    parts: list[str] = []
    parts.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
    parts.append(f"<title>评测报告 · {_esc(header.get('identifier', 'course'))}</title>")
    parts.append(f'<style>{_CSS}</style></head><body><div class="wrap">')

    # --- 头部 ---
    parts.append(f"<h1>评测报告 · {_esc(header.get('identifier', 'course'))}</h1>")
    parts.append(
        '<div class="sub">'
        f"生成时间 {_esc(header.get('timestamp', ''))} · "
        f"总耗时 {_esc(_fmt_duration(header.get('duration', 0)))} · "
        f"裁判模型 {_esc(header.get('judge_model', '—'))} · "
        f"通过阈值 {_esc(header.get('threshold', '—'))}"
        "</div>"
    )

    # --- 概览卡片 ---
    parts.append('<div class="cards">')
    for num, lbl, cls in (
        (total, "用例总数", ""),
        (counts[STATUS_PASSED], "通过", "ok"),
        (counts[STATUS_FAILED], "失败", "bad"),
        (counts[STATUS_SKIPPED], "跳过", "skip"),
        (f"{pass_rate:.1f}%" if total else "—", "用例通过率", "ok"),
        (len(metric_summaries), "指标种类", ""),
        (len(metric_rows), "指标评分数", ""),
        (_fmt_score(overall_avg) if overall_avg is not None else "—", "指标平均分", ""),
    ):
        parts.append(
            f'<div class="card"><div class="num {cls}">{_esc(num)}</div>'
            f'<div class="lbl">{_esc(lbl)}</div></div>'
        )
    parts.append("</div>")

    # --- 各指标评分汇总 ---
    parts.append("<h2>各指标评分汇总</h2>")
    if not metric_summaries:
        parts.append(
            '<p class="sub">本次运行没有 LLM 评判指标'
            "（离线确定性用例不产生指标，可看下方用例总览）。</p>"
        )
    else:
        parts.append(
            "<table><thead><tr><th>指标</th><th>平均分</th><th>最低</th>"
            "<th>最高</th><th>阈值</th><th>样本数</th><th>通过</th>"
            "<th>失败</th><th>错误</th><th>通过率</th><th>分布</th>"
            "</tr></thead><tbody>"
        )
        for summary in metric_summaries:
            parts.append(
                f"<tr><td><strong>{_esc(summary['metric'])}</strong></td>"
                f'<td class="num-cell score-cell">{_esc(_fmt_score(summary["avg"]))}</td>'
                f'<td class="num-cell">{_esc(_fmt_score(summary["min"]))}</td>'
                f'<td class="num-cell">{_esc(_fmt_score(summary["max"]))}</td>'
                f'<td class="num-cell">{_esc(_fmt_score(summary["threshold"]))}</td>'
                f'<td class="num-cell">{_esc(summary["count"])}</td>'
                f'<td class="num-cell ok">{_esc(summary["passes"])}</td>'
                f'<td class="num-cell bad">{_esc(summary["fails"])}</td>'
                f'<td class="num-cell skip">{_esc(summary["errors"])}</td>'
                f'<td class="num-cell">{_esc(_fmt_percent(summary["pass_rate"]))}</td>'
                f"<td>{_score_bar(summary)}</td></tr>"
            )
        parts.append("</tbody></table>")

    # --- 用例总览（附带该用例的指标评分） ---
    parts.append("<h2>用例总览</h2>")
    if not outcomes:
        parts.append('<p class="sub">没有采集到任何用例结果。</p>')
    else:
        score_map = metric_score_map(metric_rows)
        parts.append(
            "<table><thead><tr><th>状态</th><th>套件</th><th>用例</th>"
            "<th>指标评分</th><th>说明</th><th>耗时</th></tr></thead><tbody>"
        )
        for outcome in outcomes:
            doc = outcome.doc or outcome.message
            parts.append(
                f"<tr><td>{_status_badge(outcome.status)}</td>"
                f"<td><code>{_esc(outcome.suite)}</code></td>"
                f"<td><code>{_esc(outcome.name)}</code></td>"
                f"<td>{_case_scores_html(score_map.get(outcome.name, []))}</td>"
                f"<td>{_esc(doc)}</td>"
                f'<td class="num-cell">{_esc(_fmt_duration(outcome.duration))}</td></tr>'
            )
        parts.append("</tbody></table>")

    # --- 指标明细 ---
    parts.append("<h2>LLM 评判指标明细</h2>")
    if not metric_rows:
        parts.append(
            '<p class="sub">本次运行没有 LLM 评判指标'
            "（离线确定性用例不产生指标，可看上方用例总览）。</p>"
        )
    else:
        parts.append(
            "<table><thead><tr><th>状态</th><th>用例</th><th>指标</th>"
            "<th>分数</th><th>阈值</th><th>模型</th><th>理由</th>"
            "</tr></thead><tbody>"
        )
        for row in metric_rows:
            reason = row["reason"]
            reason_html = f"<div>{_esc(reason)}</div>" if reason else ""
            if len(reason) > 160:
                reason_html = (
                    f"<details><summary>{_esc(reason[:160])}…</summary>"
                    f'<div class="reason">{_esc(reason)}</div></details>'
                )
            parts.append(
                f"<tr><td>{_status_badge(row['status'])}</td>"
                f"<td><code>{_esc(row['case'])}</code></td>"
                f"<td>{_esc(row['metric'])}</td>"
                f'<td class="num-cell score-cell">{_esc(_fmt_score(row["score"]))}</td>'
                f'<td class="num-cell">{_esc(_fmt_score(row["threshold"]))}</td>'
                f"<td>{_esc(row['model'])}</td>"
                f"<td>{reason_html}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append(
        "<footer>由 <code>evals/report.py</code> 生成 · "
        "离线用例来自普通 pytest 断言，LLM 指标来自 DeepEval test run</footer>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def _fmt_percent(value) -> str:
    """格式化百分比。"""
    return f"{value:.0f}%" if isinstance(value, (int, float)) else "—"


def _score_bar(summary: dict) -> str:
    """渲染指标分数分布条。

    用一条水平条展示该指标下所有样本的分数分布：
    绿色=通过、红色=失败、灰色=错误，按分数从低到高排列。
    """
    rows = summary.get("_samples") or []
    if not rows:
        return "—"
    colors = {STATUS_PASSED: "#1a7f37", STATUS_FAILED: "#cf222e", "error": "#8b949e"}
    blocks = []
    for row in rows:
        color = colors.get(row["status"], "#8b949e")
        width = max(6, int((row["score"] if row["score"] is not None else 0) * 12))
        blocks.append(
            f'<span class="bar" style="background:{color};width:{width}px" '
            f'title="{_esc(row["metric"])} · {_esc(_fmt_score(row["score"]))}"></span>'
        )
    return f'<div class="bars">{"".join(blocks)}</div>'


def render_markdown(
    outcomes: list[TestOutcome],
    metric_rows: list[dict],
    metric_summaries: list[dict],
    header: dict,
) -> str:
    """渲染 Markdown 报告。

    Args:
        outcomes: 用例结果列表。
        metric_rows: DeepEval 指标明细。
        metric_summaries: 按指标聚合的评分汇总。
        header: 报告头部元信息。

    Returns:
        Markdown 文档字符串。
    """
    counts = {STATUS_PASSED: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    total = len(outcomes)
    pass_rate = (counts[STATUS_PASSED] / total * 100) if total else 0.0

    scored = [s["avg"] for s in metric_summaries if s["avg"] is not None]
    overall_avg = (sum(scored) / len(scored)) if scored else None

    lines: list[str] = []
    lines.append(f"# 评测报告 · {header.get('identifier', 'course')}")
    lines.append("")
    lines.append(f"- 生成时间：{header.get('timestamp', '')}")
    lines.append(f"- 总耗时：{_fmt_duration(header.get('duration', 0))}")
    lines.append(f"- 裁判模型：{header.get('judge_model', '—')}")
    lines.append(f"- 通过阈值：{header.get('threshold', '—')}")
    lines.append(
        f"- 用例总数：{total}（通过 {counts[STATUS_PASSED]} / "
        f"失败 {counts[STATUS_FAILED]} / 跳过 {counts[STATUS_SKIPPED]}）"
    )
    lines.append(f"- 用例通过率：{pass_rate:.1f}%")
    lines.append(f"- 指标种类：{len(metric_summaries)}")
    lines.append(f"- 指标评分数：{len(metric_rows)}")
    lines.append(f"- 指标平均分：{_fmt_score(overall_avg) if overall_avg is not None else '—'}")
    lines.append("")

    lines.append("## 各指标评分汇总")
    lines.append("")
    if not metric_summaries:
        lines.append("_本次运行没有 LLM 评判指标（离线确定性用例不产生指标）。_")
    else:
        lines.append(
            "| 指标 | 平均分 | 最低 | 最高 | 阈值 | 样本数 | 通过 | 失败 | 错误 | 通过率 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for summary in metric_summaries:
            lines.append(
                f"| {summary['metric']} | {_fmt_score(summary['avg'])} | "
                f"{_fmt_score(summary['min'])} | {_fmt_score(summary['max'])} | "
                f"{_fmt_score(summary['threshold'])} | {summary['count']} | "
                f"{summary['passes']} | {summary['fails']} | {summary['errors']} | "
                f"{_fmt_percent(summary['pass_rate'])} |"
            )
    lines.append("")

    lines.append("## 用例总览")
    lines.append("")
    if not outcomes:
        lines.append("_没有采集到任何用例结果。_")
    else:
        score_map = metric_score_map(metric_rows)
        lines.append("| 状态 | 套件 | 用例 | 指标评分 | 说明 | 耗时 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for outcome in outcomes:
            doc = (outcome.doc or outcome.message).replace("|", "\\|").replace("\n", " ")
            case_scores = _case_scores_text(score_map.get(outcome.name, []))
            lines.append(
                f"| {_status_text(outcome.status)} | `{outcome.suite}` | "
                f"`{outcome.name}` | {case_scores} | {doc} | {_fmt_duration(outcome.duration)} |"
            )
    lines.append("")

    lines.append("## LLM 评判指标明细")
    lines.append("")
    if not metric_rows:
        lines.append("_本次运行没有 LLM 评判指标（离线确定性用例不产生指标）。_")
    else:
        lines.append("| 状态 | 用例 | 指标 | 分数 | 阈值 | 模型 | 理由 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in metric_rows:
            reason = row["reason"].replace("|", "\\|").replace("\n", " ")
            if len(reason) > 200:
                reason = reason[:200] + "…"
            lines.append(
                f"| {_status_text(row['status'])} | `{row['case']}` | {row['metric']} | "
                f"{_fmt_score(row['score'])} | {_fmt_score(row['threshold'])} | "
                f"{row['model']} | {reason} |"
            )
    lines.append("")

    return "\n".join(lines)


def _case_scores_text(rows: list[dict]) -> str:
    """Markdown 用的用例指标评分文本。

    Args:
        rows: 该用例对应的指标行。

    Returns:
        形如 ``Answer Relevancy 1.00`` 的文本；无指标时返回 ``—``。
    """
    if not rows:
        return "—"
    parts = []
    for row in rows:
        score_text = _fmt_score(row["score"]) if row["score"] is not None else "N/A"
        parts.append(f"{row['metric']} {score_text}")
    return "<br>".join(parts)


def _status_text(status: str) -> str:
    """Markdown 用的中文状态文本。"""
    return {
        STATUS_PASSED: "通过",
        STATUS_FAILED: "失败",
        STATUS_SKIPPED: "跳过",
        "error": "错误",
    }.get(status, status)


__all__ = [
    "EVALS_DIR",
    "REGISTRY",
    "REPORT_DIR",
    "STATUS_FAILED",
    "STATUS_PASSED",
    "STATUS_SKIPPED",
    "TestOutcome",
    "collect_metric_rows",
    "configure_reporting",
    "finalize_reports",
    "metric_score_map",
    "record_report",
    "render_html",
    "render_markdown",
    "summarize_metrics",
]
