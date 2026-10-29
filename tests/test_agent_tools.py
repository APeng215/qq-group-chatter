import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from qq_group_chatter.agent.deepseek_llm import DeepSeekChatLLM, ToolModelResponse
from qq_group_chatter.agent.tool_loop import AgentLimits, run_tools
from qq_group_chatter.agent.tools import AgentTool, ToolRegistry
from qq_group_chatter.services.deepseek_search import DeepSeekSearchClient


def call(name, arguments, identifier="c1"):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

class Model:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.messages = []
    async def ainvoke_messages(self, messages, **kwargs):
        self.messages.append(json.loads(json.dumps(messages)))
        content, calls = next(self.steps)
        return ToolModelResponse(content, calls, "private thought", "tool_calls" if calls else "stop", None)

async def run(model, registry, **kwargs):
    return await run_tools(model, registry, kwargs.pop("limits", AgentLimits()), prompt="hello",
        system_prompt="system", context=SimpleNamespace(round_id="round"), **kwargs)

async def test_tools_continue_cache_and_keep_reasoning_private():
    executed = []
    async def tool(query):
        executed.append(query)
        return {"answer": query}
    registry = ToolRegistry()
    registry.register(AgentTool("web_search", "search", {"type": "object", "required": ["query"]}, tool))
    model = Model([( "", [call("web_search", {"query": "one"})]),
                   ("", [call("web_search", {"query": "one"}, "c2")]),
                   ("", [call("web_search", {"query": "two"}, "c3")]), ("final", [])])
    notices = []
    assert await run(model, registry, on_search_start=notices.append) == "final"
    assert executed == ["one", "two"]
    assert len(notices) == 1
    assert model.messages[1][2]["reasoning_content"] == "private thought"
    assert model.messages[1][3]["tool_call_id"] == "c1"

async def test_schema_errors_unknown_tools_and_timeout_recover():
    async def slow(query):
        await asyncio.sleep(1)
    registry = ToolRegistry()
    registry.register(AgentTool("slow", "slow", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, slow))
    assert (await registry.call("slow", {"query": 2}))["error"] == "invalid_arguments"
    assert (await registry.call("missing", {}))["error"] == "unknown_tool"
    model = Model([("", [call("slow", {"query": "x"})]), ("recovered", [])])
    assert await run(model, registry, limits=AgentLimits(tool_timeout=.01)) == "recovered"
    assert "tool_timeout" in model.messages[1][-1]["content"]

async def test_limits_and_round_timeout():
    registry = ToolRegistry()
    model = Model([("", [call("missing", {})]), ("partial results", [])])
    assert await run(model, registry, limits=AgentLimits(model_calls=2, tool_calls=1)) == "partial results"
    class Slow:
        async def ainvoke_messages(self, *args, **kwargs):
            await asyncio.sleep(1)
    assert "限制" in await run(Slow(), registry, limits=AgentLimits(round_timeout=.01))

async def test_search_fetch_retry_and_truncation(monkeypatch):
    requests = []
    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429)
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"results": [{"title": "t", "url": "https://example.com", "snippet": "s", "content": "abcd", "date": "2026"}, {"url": "https://example.com/empty"}]})
        return httpx.Response(200, json={"title": "page", "content": "abcdef"})
    monkeypatch.setenv("WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT", "3")
    monkeypatch.setenv("WEB_FETCH_MAX_CONTENT_CHARS", "4")
    client = DeepSeekSearchClient("test", transport=httpx.MockTransport(respond))
    data = await client.search("神奈")
    assert len(requests) == 2
    assert json.loads(requests[-1].content) == {"query": "神奈", "count": 5, "need_content": True}
    assert data["results"][0]["content"] == "abc"
    assert data["results"][0]["truncated"] is True
    assert len(data["results"]) == 2
    assert (await client.fetch("https://example.com"))["content"] == "abcd"
    assert (await client.search("x" * 129))["error"] == "invalid_query"
    assert (await client.fetch("file:///secret"))["error"] == "invalid_url"

async def test_non_retryable_http_errors():
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(401)
    client = DeepSeekSearchClient("test", transport=httpx.MockTransport(respond))
    assert (await client.search("x"))["error"] == "http_401"
    assert len(requests) == 1

def test_keyword_switch_and_preprocessed_events(monkeypatch):
    from qq_group_chatter.plugins import chat
    def event(text, *, to_me=False, original=None):
        message = [{"type": "text", "data": {"text": text}}]
        return SimpleNamespace(message_type="group", message=message, self_id=123,
            to_me=to_me, original_message=original if original is not None else message)
    monkeypatch.setattr(chat, "CHAT_KEYWORD_TRIGGER_ENABLED", True)
    assert chat.should_reply_to_message(event("  神奈在吗"))
    assert not chat.should_reply_to_message(event("我觉得神奈很可爱"))
    monkeypatch.setattr(chat, "CHAT_KEYWORD_TRIGGER_ENABLED", False)
    assert not chat.should_reply_to_message(event("神奈在吗"))
    original = [{"type": "text", "data": {"text": "神奈在吗"}}]
    assert not chat.should_reply_to_message(event("在吗", to_me=True, original=original))
    original.insert(0, {"type": "at", "data": {"qq": "123"}})
    assert chat.should_reply_to_message(event("在吗", to_me=True, original=original))

def test_duplicate_and_invalid_plugin_config():
    from pathlib import Path
    import uuid
    tmp_path = Path('tests/.tmp/tools') / uuid.uuid4().hex
    tmp_path.mkdir(parents=True)
    registry = ToolRegistry()
    async def execute():
        return "ok"
    tool = AgentTool("x", "x", {"type": "object"}, execute)
    registry.register(tool)
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(tool)
    path = tmp_path / "tools.json"
    path.write_text(json.dumps({"python": [{"factory": "missing_plugin_xyz:create"}]}))
    with pytest.raises(ModuleNotFoundError):
        registry.load_config(path)

@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
async def test_mcp_transport_lifecycle_and_mixed_python_tools(monkeypatch, transport):
    import mcp
    import mcp.client.stdio
    import mcp.client.streamable_http
    from contextlib import asynccontextmanager
    events = []
    @asynccontextmanager
    async def connect(*args, **kwargs):
        events.append("connect")
        yield (object(), object(), None)
        events.append("close")
    class Session:
        def __init__(self, *args): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): events.append("session_closed")
        async def initialize(self): pass
        async def list_tools(self, cursor=None):
            return SimpleNamespace(tools=[SimpleNamespace(name="echo", description="echo", inputSchema={"type": "object"})], nextCursor=None)
        async def call_tool(self, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="remote")], isError=False)
    monkeypatch.setattr(mcp, "ClientSession", Session)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", connect)
    monkeypatch.setattr(mcp.client.streamable_http, "streamablehttp_client", connect)
    registry = ToolRegistry()
    registry.servers["server"] = {"transport": transport, "command": "python", "url": "https://example.com/mcp"}
    async def local(): return "local"
    registry.register(AgentTool("local", "local", {"type": "object"}, local))
    await registry.start()
    model = Model([("", [call("server__echo", {})]), ("", [call("local", {}, "c2")]), ("mixed", [])])
    assert await run(model, registry) == "mixed"
    await registry.stop()
    assert "server__echo" not in registry.tools
    assert "local" in registry.tools
    assert events == ["connect", "session_closed", "close"]

async def test_failed_mcp_server_does_not_break_python_tools(monkeypatch):
    import mcp.client.stdio
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def broken(*args, **kwargs):
        raise OSError("disconnected")
        yield
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", broken)
    registry = ToolRegistry()
    registry.servers["broken"] = {"transport": "stdio", "command": "python"}
    await registry.start()
    assert registry.definitions() == []
    await registry.stop()

async def test_search_then_fetch_in_same_round():
    registry = ToolRegistry()
    async def search(query): return {"results": [{"url": "https://example.com"}]}
    async def fetch(url): return {"content": "verified page"}
    registry.register(AgentTool("web_search", "search", {"type": "object"}, search))
    registry.register(AgentTool("web_fetch", "fetch", {"type": "object"}, fetch))
    model = Model([("", [call("web_search", {"query": "x"})]),
                   ("", [call("web_fetch", {"url": "https://example.com"}, "c2")]), ("verified", [])])
    assert await run(model, registry) == "verified"
    assert "verified page" in model.messages[-1][-1]["content"]

def test_real_onebot_nickname_preprocessing_respects_disabled_switch(monkeypatch):
    from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message
    from nonebot.adapters.onebot.v11.bot import _check_nickname
    from qq_group_chatter.plugins import chat
    event = GroupMessageEvent(time=1, self_id=123, post_type="message", sub_type="normal",
        user_id=456, message_type="group", message_id=1, group_id=789,
        message=Message("神奈，在吗"), original_message=Message("神奈，在吗"),
        raw_message="神奈，在吗", font=0, sender={"user_id": 456, "nickname": "tester"})
    bot = SimpleNamespace(config=SimpleNamespace(nickname={"神奈"}))
    _check_nickname(bot, event)
    assert event.to_me
    monkeypatch.setattr(chat, "CHAT_KEYWORD_TRIGGER_ENABLED", False)
    assert not chat.should_reply_to_message(event)
    monkeypatch.setattr(chat, "CHAT_KEYWORD_TRIGGER_ENABLED", True)
    assert chat.should_reply_to_message(event)
