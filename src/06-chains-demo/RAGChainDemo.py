"""链（Chain）演示 7：RAG 链（RAGChain）。

背景
----
RAG（Retrieval-Augmented Generation，检索增强生成）是让模型基于
「检索到的私有资料」回答问题的标准范式，流程分三步：

    ① 切分（Chunking）：把长文档切成小块（chunk）
    ② 嵌入（Embedding）+ 向量库：把每块转成向量，存入 Chroma 等向量数据库
    ③ 检索（Retrieval）+ 生成（Generation）：
       - 检索：把「问题」也转成向量，在向量库里找最相似的 top-k 个 chunk
       - 生成：把找到的 chunk 作为「上下文」拼进 prompt，交给大模型作答

为什么要这么做？
    大模型的训练数据有截止日期，且不知道你的私有文档。
    相比「把整本 PDF 塞进 prompt」，RAG 只检索最相关的几段，
    既省 token、又提高答案准确率，还方便更新知识（改文档即可，不改模型）。

本文件用 LangChain 1.x 的 **LCEL（表达式语言）** 把上述步骤串成一条链。

整体流程图：

    question（问题）
        │
        ├──────────────────────────────┐
        ▼                              ▼
  RunnablePassthrough            retriever（检索器）
  （原样透传问题）                 ├─ 把问题转成 embedding
        │                         ├─ 在 Chroma 里做相似度搜索
        │                         └─ 返回 top-k 个 Document 作为 context
        │                              │
        └──────────────┬───────────────┘
                       ▼
              prompt（ChatPromptTemplate）
              把 context 和 question 填进模板
                       │
                       ▼
                    llm（对话模型）
                       │
                       ▼
              StrOutputParser（转成纯字符串）
                       │
                       ▼
                    answer

依赖：pip install langchain langchain-deepseek langchain-openai
      langchain-chroma langchain-text-splitters chromadb python-dotenv
"""

import json
import os
from logging import debug

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import messages_to_dict  # noqa: F401  （备用，见文末「其他常用取数方式」）
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
from pathlib import Path

# 把 src 加入模块搜索路径，保证能 import 到 src/utils 下的公共工具
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import get_logger

# 初始化 logger：日志同时输出到控制台和 logs/course.log
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 0. 加载环境变量
# --------------------------------------------------------------------------- #
# load_dotenv() 自动读取当前目录（或逐级向上）目录中的 .env 文件，
# 把键值对写入 os.environ，无需手动 export。
load_dotenv()

# 读取 Key：优先 DEEPSEEK_API_KEY，其次兼容通用命名 API_KEY；
# 都没有就立刻报错，避免等到请求接口时才失败、难以排查。
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY（或 API_KEY）"
    )
# 再显式写回环境变量，供 init_chat_model 按 provider 自动读取。
os.environ["DEEPSEEK_API_KEY"] = api_key

# --------------------------------------------------------------------------- #
# 1. 初始化 LLM（负责「生成」这一半）
# --------------------------------------------------------------------------- #
# init_chat_model 是统一的模型工厂函数：
#   - provider 决定用哪家的客户端（deepseek / openai / anthropic ...）
#   - model    是具体模型名
#
# temperature=0 表示尽量输出确定性的结果，适合「基于给定资料问答」这种任务。
llm = init_chat_model(model="deepseek-chat", api_key=api_key, temperature=0)

# --------------------------------------------------------------------------- #
# 2. 准备原始文档（相当于「我们的私有知识库」）
# --------------------------------------------------------------------------- #
# 这里用几条销售记录做演示，真实场景替换成 PDF / Markdown / 数据库即可。
documents = [
    "HP printer sales are up by 30% in 2011 Q4",
    "Lenovo Laptop sales are down by 10% in 2012",
    "A3 printer sales is 5% more in middle east as compared to A4 total",
    "Apple printer sales are up by 30% in 2011 Q4",
    # 下面两条与上面重复，是原示例里的「脏数据」，保留以便观察去重效果
    "Lenovo Laptop sales are down by 10% in 2012",
    "A3 printer sales is 5% more in middle east as compared to A4 total",
]

# --------------------------------------------------------------------------- #
# 3. 切分文档（Chunking）
# --------------------------------------------------------------------------- #
# RecursiveCharacterTextSplitter「递归字符切分器」：
#   优先按分隔符依次尝试（段落 \n\n → 行 \n → 句子 → 单词 → 字符），
#   每一层都尽量保留语义完整的片段；
#   如果某个片段仍然超过 chunk_size，就交给下一层更细的分隔符继续切。
#   这样比「固定长度硬切」能保留更多语义，对检索效果更友好。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每个 chunk 的最大字符数（按 token 估算时用 chunk_size/token_ratio）
    chunk_overlap=20,  # 相邻 chunk 之间的重叠字符数
    # 重叠的作用：避免一句完整的话被切断后，两半各自语义不完整
)

# create_documents 把一组字符串转成 Document 对象列表
# （Document = page_content + metadata，是 LangChain 的通用数据结构）。
splits = text_splitter.create_documents(documents)

print(f"Document has been split into {len(splits)} chunks.")
print("\n\033[1mExample chunks:")
print("-" * 50)
print("\033[0m")
for i, chunk in enumerate(splits):
    print(f"\nChunk {i + 1}:")
    print(chunk)
    print("-" * 30)


# --------------------------------------------------------------------------- #
# 4. 配置 Embedding 模型（负责把文本变成向量）
# --------------------------------------------------------------------------- #
def ollama_base_url() -> str:
    """返回本地 Ollama 的 OpenAI 兼容接口地址。

    Ollama 从 0.1.x 起提供了 /v1 兼容层，因此可以直接复用
    langchain_openai 的客户端，而不必安装专用的 ollama 集成包。
    """
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") or (
        "http://localhost:11434/v1"
    )


def ollama_embedding_model() -> str:
    """返回本地 Ollama 中使用的 embedding 模型名。

    默认 qwen3-embedding:4b，可通过 .env 中的 OLLAMA_EMBED_MODEL 覆盖。
    """
    return os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b") or (
        "qwen3-embedding:4b"
    )


# OpenAIEmbeddings 通过 base_url 指向本地 Ollama，实现「离线、不花钱」的 embedding。
#
# 参数说明：
#   - api_key="ollama"：Ollama 不校验密钥，但 OpenAI SDK 要求这个参数非空，随便填即可
#   - check_embedding_ctx_length=False：关闭「上下文长度检查」。
#     该检查只对 OpenAI 官方模型有意义；本地模型的分词方式不同，
#     开着容易误报「输入超长」，因此关掉。
embeddings = OpenAIEmbeddings(
    model=ollama_embedding_model(),
    base_url=ollama_base_url(),
    api_key="ollama",
    check_embedding_ctx_length=False,
)

# --------------------------------------------------------------------------- #
# 5. 建立向量库（Vector Store）
# --------------------------------------------------------------------------- #
# from_documents 做了两件事：
#   1. 用 embedding 模型把每个 chunk 转成向量；
#   2. 把「向量 + 原文 + metadata」写入 Chroma（本地持久化到 ./chroma_db）。
#
# 这里是「离线建库」的一次性操作。真实项目里通常还会：
#   - 用 directory= 指定持久化目录（默认 ./chroma_db）
#   - 用 collection_name 指定集合名
#   - 加 ids / metadata 做过滤、去重
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
logger.info("向量库建库完成：共写入 %d 个 chunk", len(splits))

# --------------------------------------------------------------------------- #
# 6. 创建检索器（Retriever）
# --------------------------------------------------------------------------- #
# as_retriever() 把向量库包装成一个「检索器」，它可以被放进 LCEL 链里当一步用。
#
# 默认参数：
#   search_type="similarity"  —— 余弦相似度检索
#   search_kwargs={"k": 4}    —— 返回最相似的 4 个 chunk
#
# 想改返回条数：vectorstore.as_retriever(search_kwargs={"k": 3})
# 想用「 MMR（最大边际相关）」去重多样性：
#   search_type="mmr"
retriever = vectorstore.as_retriever()
logger.info("检索器创建完成：search_type=%s, k=%s",
             retriever.search_type, retriever.search_kwargs.get("k"))

# --------------------------------------------------------------------------- #
# 7. 编写 RAG 提示词模板
# --------------------------------------------------------------------------- #
# 这是 RAG 的核心：把「检索到的资料」作为 Context 塞给模型，
# 让模型**基于资料**回答，而不是靠自己的记忆/臆测。
#
# 注意 prompt 里必须出现 {context} 和 {question} 两个占位符，
# 它们分别会被下一步的 retriever 和 RunnablePassthrough 填入。
#
# 实际项目中建议再补一句「若资料中没有答案，请回答不知道」，
# 可以有效减少模型幻觉。
template = """Answer the following question based on the provided context:
            Context: {context}
            Question: {question}


            Answer:"""

# ChatPromptTemplate.from_template 把上面这段 f-string 风格的模板
# 转成一个可复用的 Prompt 对象（会做变量校验）。
prompt = ChatPromptTemplate.from_template(template)

# --------------------------------------------------------------------------- #
# 8. 组装 RAG 链（LCEL）
# --------------------------------------------------------------------------- #
# 这行代码是整个 RAG 的灵魂，用 | 把四个组件串成一条流水线：
#
#   {"context": retriever, "question": RunnablePassthrough()}
#        │
#        │ 字典这一步是「并行分支」：
#        │   - retriever 收到的是【输入的问题】（RunnablePassthrough 的输出）
#        │     → 输出 top-k 个 Document 列表，拼成 context 字符串
#        │   - RunnablePassthrough 原样透传问题 → question
#        │ 两个 key 正好对应 prompt 模板里的 {context} 和 {question}
#        ▼
#      prompt        ← 模板渲染，产出 ChatPromptValue
#        ▼
#       llm          ← 调用模型，产出 AIMessage
#        ▼
#  StrOutputParser() ← 把 AIMessage 转成纯字符串，调用时直接拿到 answer
#
# 用 | 串联的好处：每个环节都是 Runnable，可以单独测试、单独替换，
# 也可以用 .stream() 做流式输出，或用 batch() 批量处理。


def debug_print(name):
    """构造一个调试用的 RunnableLambda，把每一步的中间结果打印并记录到日志。

    Args:
        name: 该步骤的名称（如 "检索结果"）。

    Returns:
        一个 RunnableLambda，接收上一步的输出，原样返回，同时输出调试信息。
    """

    def _f(x):
        print(f"\n=== {name} ===")
        print(x)
        # 日志只记录摘要（长度/类型），避免把整段上下文重复写进日志文件
        logger.debug("链节点 %s：输出类型=%s, 长度=%d", name, type(x).__name__, len(str(x)))
        return x

    return RunnableLambda(_f)


rag_chain = (
    {"context": retriever | debug_print("检索结果"), "question": RunnablePassthrough()}
    | debug_print("进入prompt的输入")
    | prompt
    | debug_print("prompt输出")
    | llm
    | debug_print("LLM输出")
)
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
#     # | StrOutputParser()
# )

# --------------------------------------------------------------------------- #
# 9. 调用 RAG 链
# --------------------------------------------------------------------------- #
# invoke 的入参就是上面字典分支的输入，也就是「问题字符串」。
question = "What is Apple printer sales in 2011 Q4?"
logger.info("开始调用 RAG 链，问题：%s", question)
result = rag_chain.invoke(question)
logger.info("RAG 链调用完成，答案长度=%d 字符", len(result.content))

# 注意：当前链尾**没有**接 StrOutputParser()（被注释掉了），
# 所以 invoke() 返回的是一个 AIMessage 对象，而不是字符串，
# 更不是 {"messages": [...]} 这种「Agent 状态字典」。
# 这正是下面「打印完整信息」那段代码报错的原因：
#   AttributeError: 'AIMessage' object has no attribute 'items'
#
# 两种方案，任选其一：
#   A. 保持现状（保留 AIMessage）→ 用 result.content / result.response_metadata 取数
#   B. 放开 StrOutputParser()    → invoke() 直接返回纯字符串，但就拿不到元数据了
#
# 这里选 A：既能看到最终答案，又能打印 token 用量、模型名等完整信息。
print(f"\033[1m Question: {question}")
print(f"\033[0m Answer: {result.content}")

# --------------------------------------------------------------------------- #
# 10. 常见坑 / 进阶方向
# --------------------------------------------------------------------------- #
# 10.1 Chroma.from_documents 每次运行都会重新建库并写入，重复运行会导致
#      集合里堆积重复数据。生产环境应把建库与查询拆成两个脚本，
#      建库脚本只在文档更新时执行一次。
# 10.2 检索质量决定回答质量。检索不准时可以在链后加一道「重排序」
#      （rerank）环节，或调整 chunk_size / k 的取值。
# 10.3 想看模型到底依据了哪几段资料，可以在链后接上 retriever 单独查一遍，
#      或改用 stream_mode="messages" 观察中间过程。
# 10.4 生产环境建议加入「引用来源」：把检索到的 chunk 的 metadata 一起返回，
#      在答案里标注出处，既能减少幻觉也便于人工核查。
# 10.5 本例用 Ollama 本地 embedding；若改用云端 embedding，
#      只需替换 embeddings 的构造方式，其余代码完全不用动。


# --------------------------------------------------------------------------- #
# 11. 打印完整信息
# --------------------------------------------------------------------------- #
# 上面 result 是一个 AIMessage 对象，它身上主要有 5 个可读属性：
#
#   .content              → 正文（最终答案）
#   .response_metadata    → 厂商返回的原始元数据（model_name / finish_reason / token_usage ...）
#   .usage_metadata       → 标准化后的 token 用量统计
#   .additional_kwargs    → 额外字段（DeepSeek 的 reasoning_content、refusal 等）
#   .id / .type           → 消息 ID 与类型标记，便于链路追踪
#
# 下面把它们逐个打印出来。

print("=" * 70)
print("1) 完整 result（AIMessage 对象）")
print("=" * 70)
# AIMessage 是 pydantic 模型，不能直接 json 序列化。
# .model_dump() 会把它转成普通字典（LangChain 的消息 JSON 规范），
# 再交给 json.dumps 格式化输出，这样可读、可直接存日志或发到前端。
print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))

print("\n" + "=" * 70)
print("2) 逐项元数据详情")
print("=" * 70)

# content：正文内容。可能是字符串，也可能是 content blocks 列表。
print(f"content:\n{result.content}")

# response_metadata：厂商返回的原始元数据
# 包含 model_name、finish_reason、token_usage、system_fingerprint、id 等。
response_metadata = getattr(result, "response_metadata", None)
print(
    f"\nresponse_metadata:\n{json.dumps(response_metadata, indent=2, ensure_ascii=False, default=str)}"
)

# usage_metadata：标准化的 token 用量统计
# input_tokens（输入消耗）、output_tokens（输出消耗）、total_tokens（合计），
# 以及细分的 reasoning tokens（推理模型思考过程消耗的 token）。
usage_metadata = getattr(result, "usage_metadata", None)
print(
    f"\nusage_metadata:\n{json.dumps(usage_metadata, indent=2, ensure_ascii=False, default=str)}"
)
if usage_metadata:
    logger.info(
        "token 用量：输入=%s，输出=%s，合计=%s",
        usage_metadata.get("input_tokens"),
        usage_metadata.get("output_tokens"),
        usage_metadata.get("total_tokens"),
    )

# additional_kwargs：额外字段，不同厂商内容不同。
# DeepSeek 会在这里返回 reasoning_content（思维链）和 refusal（拒答标记）。
additional_kwargs = getattr(result, "additional_kwargs", None)
print(
    f"\nadditional_kwargs:\n{json.dumps(additional_kwargs, indent=2, ensure_ascii=False, default=str)}"
)

# 消息 ID 与类型标记，便于做链路追踪
print(f"\nid: {getattr(result, 'id', None)}")
print(f"type: {getattr(result, 'type', None)}")

print("\n" + "=" * 70)
print("3) 快速取用：常用字段速查")
print("=" * 70)
print(f"最终答案      : {result.content}")
print(
    f"实际模型名    : {response_metadata.get('model_name') if response_metadata else None}"
)
print(
    f"结束原因      : {response_metadata.get('finish_reason') if response_metadata else None}"
)
print(
    f"总 token 数   : {usage_metadata.get('total_tokens') if usage_metadata else None}"
)
print(
    f"输入 token 数 : {usage_metadata.get('input_tokens') if usage_metadata else None}"
)
print(
    f"输出 token 数 : {usage_metadata.get('output_tokens') if usage_metadata else None}"
)

# --------------------------------------------------------------------------- #
# 附：其他常用取数方式
# --------------------------------------------------------------------------- #
#   result.content                       -> 模型最终回答（字符串）
#   result.usage_metadata["total_tokens"]-> 本次调用的总 token 数
#   result.response_metadata["model_name"]-> 实际使用的模型名
#
# 若想拿到「填充后的完整提示词」（即塞进 Context 后的那条 user 消息），
# 有两种办法：
#
#   ① 用 stream_mode="values" 观察链上每一步的中间状态（推荐，能看到所有节点）：
#        for chunk in rag_chain.stream(question, stream_mode="values"):
#            print(chunk)   # 依次是 字典分支结果 → ChatPromptValue → AIMessage
#
#   ② 在链尾接一个「返回 prompt + answer」的字典分支：
#        rag_chain = (
#            {"context": retriever, "question": RunnablePassthrough()}
#            | prompt
#            | llm
#            | {"answer": StrOutputParser(), "prompt": RunnablePassthrough()}
#        )
#      这样 invoke() 返回 {"answer": ..., "prompt": ...}，
#      result["prompt"].to_messages()[0].content 就是填充后的提示词。
#
# 需要流式输出时改用 rag_chain.stream(..., stream_mode="messages") 逐 token 获取。
