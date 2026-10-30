"""DeepSeek -> OpenAI 兼容代理。

背景
----
DeepSeek 官方 API 只支持 ``response_format={"type": "json_object"}``，
**不支持** ``{"type": "json_schema", ...}``。而 OpenAI SDK / CrewAI 在做结构化
输出（``response_format=PydanticModel``）时走的正是 ``json_schema``，直接调用
DeepSeek 会报::

    400 This response_format type is unavailable now

本模块提供一个本地代理，把 OpenAI 的请求转换成 DeepSeek 能接受的形态：

* ``response_format.type == "json_schema"``
  -> 改用 DeepSeek 支持的 ``tool_choice="required"`` + 一个名为
  ``__structured_output__`` 的 function tool，拿到结果后把 arguments 反序列化成
  JSON 字符串塞回 ``message.content``。客户端（OpenAI SDK）随后会按
  ``json_schema`` 解析 content，行为与直连 OpenAI 完全一致。
* ``response_format.type == "json_object"``
  -> 原样透传给 DeepSeek。
* 其它字段（tools / stream / temperature ...）原样透传。

用法::

    # 终端 1
    python src/utils/deepseek_proxy.py          # 默认监听 127.0.0.1:8765
    # 终端 2
    LLM(model="openai/deepseek-chat", base_url="http://127.0.0.1:8765/v1", api_key="local")
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

load_dotenv()

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
PROXY_HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8765"))

# 代理内部用来承载结构化输出的虚拟工具名
_STRUCTURED_TOOL_NAME = "__structured_output__"


class _UpstreamError(Exception):
    """DeepSeek 上游返回了错误。"""

    def __init__(self, status_code: int, payload: Any) -> None:
        super().__init__(f"upstream error {status_code}")
        self.status_code = status_code
        self.payload = payload


def _split_response_format(
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """把 json_schema 的 response_format 拆成 (deepseek_response_format, schema)。"""
    response_format = body.get("response_format")
    if not isinstance(response_format, dict):
        return None, None
    if response_format.get("type") != "json_schema":
        return response_format, None

    json_schema = response_format.get("json_schema") or {}
    schema = json_schema.get("schema") or {}
    name = json_schema.get("name") or "response"
    return None, {"name": name, "schema": schema}


def _build_tool_from_schema(schema_info: dict[str, Any]) -> dict[str, Any]:
    """把 JSON Schema 包成一个 function tool 定义。"""
    schema = schema_info.get("schema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": _STRUCTURED_TOOL_NAME,
            "description": "Return the final structured response matching the schema.",
            "parameters": schema,
        },
    }


def _normalize_json_text(text: str) -> str:
    """从可能带噪声的文本里提取出合法的 JSON。

    DeepSeek 偶尔会在 tool-call 的 arguments 后面追加多余的字符（例如重复的
    结尾片段），导致 ``json.loads`` 报 "trailing characters"。这里用
    ``json.JSONDecoder().raw_decode`` 截取第一个完整的 JSON 值。
    """
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1:
        return text

    try:
        _, end = decoder.raw_decode(text, start)
        return text[start:end]
    except json.JSONDecodeError:
        return text


def _extract_structured_content(message: dict[str, Any]) -> str | None:
    """从 DeepSeek 返回的 tool_calls 中取出结构化 JSON 字符串。"""
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        if function.get("name") == _STRUCTURED_TOOL_NAME:
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                cleaned = _normalize_json_text(arguments)
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    return cleaned
                return json.dumps(parsed, ensure_ascii=False)
            return json.dumps(arguments, ensure_ascii=False)
    return None


def _prepare_upstream_body(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """构造发给 DeepSeek 的请求体。

    Returns:
        (upstream_body, used_structured_tool)
    """
    upstream = dict(body)

    # DeepSeek 不接受 OpenAI 的一些顶层字段
    for key in ("service_tier", "store", "metadata", "parallel_tool_calls"):
        upstream.pop(key, None)

    deepseek_format, schema_info = _split_response_format(body)

    if schema_info is not None:
        # 用 tool-calling 承载结构化输出
        upstream.pop("response_format", None)
        tools = list(upstream.get("tools") or [])
        # 移除 OpenAI SDK 为 strict 模式加的 wrapper
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters", {}),
                },
            }
            for t in tools
            if t.get("type") == "function"
            and t.get("function", {}).get("name") != _STRUCTURED_TOOL_NAME
        ]
        tools.append(_build_tool_from_schema(schema_info))
        upstream["tools"] = tools
        upstream["tool_choice"] = {
            "type": "function",
            "function": {"name": _STRUCTURED_TOOL_NAME},
        }
        return upstream, True

    if deepseek_format is not None:
        upstream["response_format"] = deepseek_format

    return upstream, False


def _restore_response(
    response: dict[str, Any], used_structured_tool: bool
) -> dict[str, Any]:
    """把 DeepSeek 的响应还原成 OpenAI 客户端期待的形态。

    DeepSeek 的 tool-calling 已经和 OpenAI 兼容，所以这里主要做两件事：

    1. 把 ``__structured_output__`` 工具调用里的 arguments 取出来，塞进
       ``message.content``，这样 OpenAI SDK 能按 ``json_schema`` 解析。
    2. DeepSeek 偶尔忽略 ``tool_choice`` 直接把 JSON 写在 content 里，
       此时也做一次规范化，让 SDK 能解析。
    """
    if not used_structured_tool:
        return response

    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        structured = _extract_structured_content(message)
        if structured is None:
            # 兜底：模型可能直接把 JSON 写在 content 里
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                stripped = content.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    structured = _normalize_json_text(stripped)
        if structured is not None:
            message["content"] = structured
            message["tool_calls"] = None
            message["refusal"] = None
            choice["message"] = message
            choice["finish_reason"] = choice.get("finish_reason") or "stop"
    return response


def _response_is_truncated(payload: dict[str, Any]) -> bool:
    """判断响应是否因为 max_tokens 上限被截断。

    DeepSeek 在 ``finish_reason == "length"`` 时会返回不完整的 tool-call
    arguments（JSON 字符串被截断），下游解析必然失败。
    """
    for choice in payload.get("choices", []):
        if choice.get("finish_reason") != "length":
            continue
        message = choice.get("message") or {}
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or ""
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                return True
        # 没有 tool_calls 但正文被截断，也算截断
        if not message.get("tool_calls") and choice.get("finish_reason") == "length":
            return True
    return False


async def _proxy_to_deepseek(body: dict[str, Any]) -> dict[str, Any]:
    upstream_body, used_structured_tool = _prepare_upstream_body(body)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=180.0) as client:
        for attempt in range(2):
            response = await client.post(
                f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
                json=upstream_body,
                headers=headers,
            )
            if response.status_code >= 400:
                raise _UpstreamError(
                    response.status_code,
                    response.json() if response.content else {"message": response.text},
                )
            payload = response.json()

            if not _response_is_truncated(payload):
                break

            # 被 max_tokens 截断 -> 提高上限重试一次
            current = upstream_body.get("max_tokens")
            base = current if isinstance(current, int) else 2000
            upstream_body = {**upstream_body, "max_tokens": base * 4}
            print(
                f"[proxy] response truncated by max_tokens={base}, "
                f"retrying with max_tokens={base * 4}",
                flush=True,
            )
    if os.getenv("PROXY_DEBUG"):
        for choice in payload.get("choices", []):
            message = choice.get("message") or {}
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments") or ""
                try:
                    json.loads(arguments)
                    status = "ok"
                except Exception as exc:
                    status = f"INVALID ({exc})"
                print(
                    f"[proxy-debug] tool={function.get('name')} "
                    f"len={len(arguments)} finish={choice.get('finish_reason')} -> {status}",
                    flush=True,
                )
                if status != "ok":
                    print(f"[proxy-debug] raw={arguments!r}", flush=True)
    return _restore_response(payload, used_structured_tool)


async def chat_completions(request: Request) -> Any:
    """POST /v1/chat/completions"""
    body = await request.json()
    if body.get("stream"):
        return await _stream_to_deepseek(body)
    try:
        payload = await _proxy_to_deepseek(body)
    except _UpstreamError as exc:
        return JSONResponse({"error": exc.payload}, status_code=exc.status_code)
    except Exception as exc:  # pragma: no cover - 兜底，避免 500 无信息
        return JSONResponse(
            {"error": {"message": f"proxy error: {exc}", "type": "proxy_error"}},
            status_code=502,
        )
    return JSONResponse(payload)


async def _stream_to_deepseek(body: dict[str, Any]) -> Any:
    """流式转发。

    结构化输出（json_schema）场景下不做特殊处理：DeepSeek 的流式响应里
    tool_calls 的 arguments 是分片增量，直接透传即可；OpenAI SDK 的
    ``chat.completions.parse`` 在流式模式下会自行拼接。
    """
    upstream_body, _ = _prepare_upstream_body(body)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    async def event_generator() -> Any:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
                json=upstream_body,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    payload = await response.aread()
                    error = {
                        "error": {
                            "message": payload.decode("utf-8", "replace"),
                            "type": "upstream_error",
                        }
                    }
                    yield f"data: {json.dumps(error)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def models(request: Request) -> Any:
    """GET /v1/models - 透传 DeepSeek 的模型列表。"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{DEEPSEEK_BASE_URL.rstrip('/')}/models", headers=headers
        )
    return JSONResponse(response.json(), status_code=response.status_code)


async def health(request: Request) -> Any:
    """GET /health"""
    return JSONResponse(
        {
            "status": "ok",
            "upstream": DEEPSEEK_BASE_URL,
            "key_configured": bool(DEEPSEEK_API_KEY),
        }
    )


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ]
)


def main() -> None:
    import uvicorn

    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未找到 API Key，请在 .env 中设置 DEEPSEEK_API_KEY")
    print(f"DeepSeek proxy listening on http://{PROXY_HOST}:{PROXY_PORT}/v1")
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="info")


if __name__ == "__main__":
    main()
