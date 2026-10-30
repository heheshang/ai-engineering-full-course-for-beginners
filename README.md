# AI Engineering Full Course for Beginners

AI 工程入门课程示例代码，覆盖从 RAG、文本切分、提示工程、文档加载、记忆、
链式编排，到工具与多智能体（CrewAI）的完整实践路径。

所有示例均基于 **LangChain 1.x** 现行 API 编写，注释为中文，
并统一接入日志（同时输出到控制台与 `logs/course.log`）。

## 环境准备

```bash
# 需要 Python 3.14+
uv sync
```

在 `.env` 中配置 API Key：

```dotenv
DEEPSEEK_API_KEY=sk-...
# 可选：Serper 搜索（不配置则回退到 DuckDuckGo）
SERPER_API_KEY=...
```

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `src/01-rag-demo/` | RAG 基础：文档切分 + 向量检索 + Agent 问答 |
| `src/02-chunking-methods-demo/` | 各种文本切分器对比（字符 / 递归 / Markdown 标题 / Token） |
| `src/03-prompt-engineering-demo/` | 提示工程技巧（CoT / Few-shot / 角色约束 / 约束式输出） |
| `src/04-document-loaders-demo/` | PDF / HTML / CSV 等文档加载器 |
| `src/05-memory-demo/` | 对话记忆：Buffer / BufferWindow / Summary |
| `src/06-chains-demo/` | 链式编排：单链 / 顺序链 / 路由链 / SQL 链 |
| `src/07-tools-and-agents-demo/` | 工具调用与多智能体（CrewAI 三个实战） |
| `src/08-huggingface-demo/` | HuggingFace 本地模型：摘要 / 翻译 / 向量 / 图像理解 |
| `src/utils/` | 公共工具（统一日志 + DeepSeek 兼容代理） |
| `tools/` | 开发辅助脚本（批量接入日志、静态检查） |

## 演示清单

### 01 · RAG（`src/01-rag-demo/`）

| 文件 | 内容 |
| --- | --- |
| `RAGDemo.py` | 完整 RAG 流程：切分 → 向量化 → 入 Chroma → 相似度检索验证 → Agent 问答 |

### 02 · 文本切分（`src/02-chunking-methods-demo/`）

| 文件 | 内容 |
| --- | --- |
| `CharacterTextSplitterDemo.py` | 最基础的按字符分隔符切分 |
| `RecursiveCharacterTextSplitterDemo.py` | 多级分隔符递归切分（RAG 最常用） |
| `MarkdownHeaderTextSplitterDemo.py` | 按 Markdown 标题层级切分，保留标题元数据 |
| `TokenTextSplitterDemo.py` | 按 token 数量切分（最贴合大模型上下文窗口） |

### 03 · 提示工程（`src/03-prompt-engineering-demo/`）

| 文件 | 内容 |
| --- | --- |
| `BasicPromptDemo.py` | 基础提示词与 `PromptTemplate` |
| `ChainOfThoughtDemo.py` | 思维链提示 |
| `ConstrainedOutputPromptingDemo.py` | 约束式输出（含自动校验） |
| `FewShotDemo.py` | 少样本提示（含风格指纹自检） |
| `RoleTaskContextPromptDemo.py` | Role / Task / Context 结构化提示词 |

### 04 · 文档加载器（`src/04-document-loaders-demo/`）

| 文件 | 内容 |
| --- | --- |
| `CSVLoaderDemo.py` | CSV：多分隔符 / `source_column` / `metadata_columns` / `lazy_load` |
| `HTMLLoaderDemo.py` | HTML：官方 Loader + BeautifulSoup 自定义 + 表格解析 |
| `PDFLoaderDemo.py` | PDF：一页一 Document / 自定义切分 / metadata 精简 |
| `data/make_sample_pdf.py` | 生成带中文字体的示例 PDF |

### 05 · 对话记忆（`src/05-memory-demo/`）

| 文件 | 内容 |
| --- | --- |
| `ConversationBufferMemoryDemo.py` | 全量缓冲记忆（`checkpointer`） |
| `ConversationBufferWindowMemoryDemo.py` | 滑动窗口记忆（`before_model` 中间件裁剪） |
| `ConversationSummaryMemoryDemo.py` | 摘要记忆（`SummarizationMiddleware`） |

### 06 · 链式编排（`src/06-chains-demo/`）

| 文件 | 内容 |
| --- | --- |
| `SingleChainDemo.py` | 单链：单次 `prompt \| llm` |
| `SimpleSequentialChainDemo.py` | 简单顺序链：两段串联 |
| `SequentialChainDemo.py` | 顺序链：多段串联 + 中间结果观测 |
| `SequentialChainDemo2.py` | 带并行分支的顺序链（`RunnableParallel` 字典简写） |
| `RouterChainDemo.py` | 路由链：按问题类型分发到不同链 |
| `LLMRouterChainDemo.py` | 多提示路由链（`MultiPromptChain` 的现代写法） |
| `MathChainDemo.py` | 数学链：Agent + `numexpr` 计算器工具 |
| `MathChainDemo2.py` | 数学链：从旧版 `LLMMathChain` 迁移到 1.x |
| `RAGChainDemo.py` | RAG 链：LCEL 显式流水线的完整拆解 |
| `SQLChainDemo.py` | SQL 链（Text-to-SQL）：LCEL 固定链 vs Agent + 工具 |

### 07 · 工具与多智能体（`src/07-tools-and-agents-demo/`）

| 文件 | 内容 |
| --- | --- |
| `MathTools_Agents_Usecase.py` | 自定义 `@tool` + `ModelRetryMiddleware` 重试 |
| `SQLDatabaseTool_SqlAgent.py` | `SQLDatabaseToolkit` + Agent 执行查询 |
| `WebScraping_Using_Apify.py` | Apify 爬虫 + 本地 Ollama Embedding 的 RAG |
| `CrewAI_Usecase1_WebScraping.py` | CrewAI：网页抓取 + TXT 语义检索 + 问答 |
| `CrewAI_Usecase2_Personalized_Email_Drafts.py` | CrewAI：线索画像 + 个性化外联邮件 |
| `CrewAI_Usecase3_Trading_Platform.py` | CrewAI：交易平台层级式多智能体协作 |

### 08 · HuggingFace 本地模型（`src/08-huggingface-demo/`）

| 文件 | 内容 |
| --- | --- |
| `SentenceSummarization.py` | 句子摘要（bfloat16 精度 + 短文本场景） |
| `SentenceTranslation.py` | 句子翻译 |
| `SentenceEmbeddings.py` | 句子向量 |
| `ImageUnderstanding.py` | 图像理解（BLIP） |

## 统一日志

所有演示脚本都接入了 `src/utils/logging_utils.py`，日志**同时输出到控制台和
`logs/course.log`**，格式为 `时间 | 级别 | 模块名 | 消息`。

```python
from utils.logging_utils import get_logger, log_exception

logger = get_logger(__name__)

logger.info("检索完成，命中 %d 个片段", len(results))
logger.warning("文档过长，建议切分：%d 字符", len(text))

try:
    result = chain.invoke(payload)
except Exception:
    # log_exception 会额外打印完整堆栈，便于定位问题
    log_exception(logger, "调用链失败：%s", payload)
    raise
```

常用约定：

* **每个关键节点都记一条 INFO**：模型初始化、切分结果、检索命中、token 用量、
  文件落盘等，方便在日志里还原完整执行链路。
* **异常一律用 `log_exception`**：它会在错误信息后追加完整堆栈。
* **不要把整段上下文写进日志**：长文本只在控制台打印，日志里记长度/条数即可。

日志目录默认是项目根目录下的 `logs/`，可用环境变量覆盖：

```bash
LOG_DIR=/var/log/ai-course LOG_LEVEL=DEBUG python src/01-rag-demo/RAGDemo.py
```

支持的环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `LOG_DIR` | `<项目根>/logs` | 日志文件目录 |
| `LOG_FILE_ENABLED` | `1` | 设为 `0` 时只输出到控制台（CI 场景） |
| `LOG_TO_STDOUT` | `1` | 设为 `0` 时不输出到控制台（后台服务场景） |

开发辅助脚本：

```bash
# 给新脚本批量接入日志（插入 import 与 logger 初始化，并做语法校验）
python tools/add_logging.py src/xx-demo/NewDemo.py

# 静态检查：logger 是否定义、log_exception 是否导入、sys.path 深度是否正确
python tools/check_logging.py
```

## DeepSeek 兼容代理

DeepSeek 官方 API 支持 `response_format={"type": "json_object"}`，但**不支持**
OpenAI 的 `{"type": "json_schema", ...}`。而 LangChain / CrewAI 的结构化输出
走的正是 `json_schema`，直连 DeepSeek 会报：

```
400 This response_format type is unavailable now
```

`src/utils/deepseek_proxy.py` 是一个本地 OpenAI 兼容代理，负责做双向转换：

* 请求：`json_schema` → 改写成 DeepSeek 支持的 tool-calling
* 响应：tool-call 的 `arguments` → 还原成 OpenAI 期望的 `message.content`
* 兜底：响应被 `max_tokens` 截断时自动提高上限重试

启动：

```bash
python src/utils/deepseek_proxy.py
# 监听 http://127.0.0.1:8765/v1
```

然后在代码里用标准 `openai` provider 即可：

```python
llm = LLM(
    model="openai/deepseek-chat",
    base_url="http://127.0.0.1:8765/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
```

设置 `PROXY_DEBUG=1` 可以打印 tool-call 参数的解析状态，便于排查截断问题。

## 运行示例

```bash
# 先启动代理（另一个终端）
python src/utils/deepseek_proxy.py

python src/01-rag-demo/RAGDemo.py
python src/07-tools-and-agents-demo/CrewAI_Usecase1_WebScraping.py
python src/07-tools-and-agents-demo/CrewAI_Usecase2_Personalized_Email_Drafts.py
python src/07-tools-and-agents-demo/CrewAI_Usecase3_Trading_Platform.py
```

## 说明

* 嵌入模型默认使用本地 ONNX MiniLM（`all-MiniLM-L6-v2`），首次运行会下载
  ~79MB 到 `~/.cache/chroma`，之后离线可用，也不需要 `OPENAI_API_KEY`。
* `src/07-tools-and-agents-demo/CrewAI_Usecase2_Personalized_Email_Drafts.py`
  会自动创建 `instructions/` 目录并写入示例内部资料，可替换为自己的实际内容。
* `src/04-document-loaders-demo/data/make_sample_pdf.py` 会在系统字体路径中
  查找可用的 CJK 字体；找不到时 PDF 中的中文将无法被正确提取。
* 各演示脚本的向量库持久化目录为 `chroma_db/`（已加入 `.gitignore`）。
