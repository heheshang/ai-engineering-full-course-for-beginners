import os

from dotenv import load_dotenv
from langchain_apify import ApifyWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

# 加载环境变量
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["APIFY_API_TOKEN"] = os.getenv("APIFY_API_TOKEN")

# 1. 爬取网页
apify = ApifyWrapper()
print("开始爬取网站内容...")
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [
            {"url": "https://en.wikipedia.org/wiki/Generative_artificial_intelligence"}
        ]
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

# 2. 构建向量存储与检索器
documents = loader.load()
print(f"已加载 {len(documents)} 个文档，正在计算嵌入...")


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

vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# 3. 用 LCEL 手动组合 RAG 链（不用 langchain-classic）
llm = ChatDeepSeek(
    model="deepseek-chat", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY")
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "请根据以下上下文简洁回答问题，最多三句话。如果不知道答案，请如实说不知道。\n\n上下文：\n{context}",
        ),
        ("human", "{input}"),
    ]
)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 查询
query = "What is Generative artificial intelligence?"
answer = rag_chain.invoke(query)

print("\n--- 答案 ---")
print(answer)

# 如需来源，单独调用 retriever
print("\n--- 来源 ---")
for doc in retriever.invoke(query):
    print(doc.metadata.get("source", "未知来源"))
