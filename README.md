# AI Engineering Full Course for Beginners

AI 工程入门课程示例代码，覆盖从 RAG、文本切分、提示工程、文档加载、记忆、
链式编排，到工具与多智能体（CrewAI）的完整实践路径。

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
| `src/01-rag-demo/` | RAG 基础：文档切分 + 向量检索 + 生成 |
| `src/02-chunking-methods-demo/` | 各种文本切分器对比 |
| `src/03-prompt-engineering-demo/` | 提示工程技巧（CoT / Few-shot / 角色约束等） |
| `src/04-document-loaders-demo/` | PDF / HTML / CSV 等文档加载器 |
| `src/05-memory-demo/` | 对话记忆：Buffer / BufferWindow / Summary |
| `src/06-chains-demo/` | 链式编排：单链 / 顺序链 / 路由链 / SQL 链 |
| `src/07-tools-and-agents-demo/` | 工具调用与多智能体（CrewAI 三个实战） |
| `src/08-huggingface-demo/` | HuggingFace 本地模型：摘要 / 翻译 / 向量 / 图像理解 |
| `src/utils/` | 公共工具（含 DeepSeek 兼容代理） |

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

python src/07-tools-and-agents-demo/CrewAI_Usecase1_WebScraping.py
python src/07-tools-and-agents-demo/CrewAI_Usecase2_Personalized_Email_Drafts.py
python src/07-tools-and-agents-demo/CrewAI_Usecase3_Trading_Platform.py
```

## 说明

* 嵌入模型默认使用本地 ONNX MiniLM（`all-MiniLM-L6-v2`），首次运行会下载
  ~79MB 到 `~/.cache/chroma`，之后离线可用，也不需要 `OPENAI_API_KEY`。
* `src/07-tools-and-agents-demo/CrewAI_Usecase2_Personalized_Email_Drafts.py`
  会自动创建 `instructions/` 目录并写入示例内部资料，可替换为自己的实际内容。
