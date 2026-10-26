import uuid
from types import SimpleNamespace
from pathlib import Path

import pytest

from qq_group_chatter.agent.deepseek_llm import (
    DeepSeekChatLLM,
    _read_dotenv_key,
    create_deepseek_chat_llm,
)
from qq_group_chatter.llm_tracing import LLMTraceStore
from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.models import LongTermMemoryBundle, build_group_conversation_context


def trace_dir(name):
    path = Path("tests/.tmp/deepseek-tracing") / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("content", [None, "", "not valid JSON"])
async def test_fallback_preserves_full_api_response_after_trace_reload(content):
    payload = {
        "id": "test-response",
        "choices": [{
            "finish_reason": "length",
            "message": {"content": content, "reasoning_content": "unfinished reasoning"},
        }],
    }

    class Response:
        choices = [SimpleNamespace(message=SimpleNamespace(**payload["choices"][0]["message"]))]

        def model_dump(self, *, mode):
            assert mode == "json"
            return payload

    class Client(FakeAsyncClient):
        async def create(self, **kwargs):
            return Response()

    store = LLMTraceStore(path=trace_dir("fallback") / "traces.jsonl")
    agent = ChatAgent(llm=DeepSeekChatLLM(api_key="secret", client=Client(), trace_store=store))
    reply = await agent.generate_reply(
        user_message="hello",
        context=build_group_conversation_context(
            group_id=888888, user_id=123456, message_id="m1", nickname="tester", timestamp=123.0,
        ),
        short_term_messages=[],
        long_term_memory=LongTermMemoryBundle(user_memories=[], conversation_memories=[]),
    )

    trace = LLMTraceStore(path=store.path).snapshot()["traces"][0]
    assert reply.content == "我刚刚没能整理好回复，稍后再试。"
    assert trace["fallback_reason"] == "invalid_chat_decision"
    assert trace["response_text"] == (content or "")
    assert trace["raw_response"] == payload
    assert trace["reasoning_content"] == "unfinished reasoning"


class FakeAsyncClient:
    def __init__(self):
        self.calls = []

        class Chat:
            def __init__(self, outer):
                self.completions = outer

        self.chat = Chat(self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Message:
            content = "OK"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


async def test_deepseek_llm_calls_v4_pro_with_thinking_enabled_by_default():
    client = FakeAsyncClient()
    llm = DeepSeekChatLLM(api_key="secret", client=client)

    result = await llm.ainvoke("hello")

    assert result == "OK"
    assert client.calls[0]["model"] == "deepseek-flash"
    assert "max_tokens" not in client.calls[0]
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert client.calls[0]["stream"] is False
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "牧野神奈" in client.calls[0]["messages"][0]["content"]
    assert "神奈" in client.calls[0]["messages"][0]["content"]
    assert "人类" not in client.calls[0]["messages"][0]["content"]
    assert "不要自称" in client.calls[0]["messages"][0]["content"]
    assert "AI" in client.calls[0]["messages"][0]["content"]
    assert "助手" in client.calls[0]["messages"][0]["content"]
    assert "模型" in client.calls[0]["messages"][0]["content"]
    assert "不要输出或解释思考过程" in client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["messages"][1] == {"role": "user", "content": "hello"}


async def test_deepseek_llm_can_pass_response_format():
    client = FakeAsyncClient()
    llm = DeepSeekChatLLM(
        api_key="secret",
        client=client,
        response_format={"type": "json_object"},
    )

    await llm.ainvoke("return json")

    assert client.calls[0]["response_format"] == {"type": "json_object"}


async def test_deepseek_llm_can_override_response_format_per_call():
    client = FakeAsyncClient()
    llm = DeepSeekChatLLM(api_key="secret", client=client)

    await llm.ainvoke("return json", response_format={"type": "json_object"})

    assert client.calls[0]["response_format"] == {"type": "json_object"}


async def test_deepseek_llm_can_override_system_prompt_per_call():
    client = FakeAsyncClient()
    llm = DeepSeekChatLLM(api_key="secret", client=client)

    await llm.ainvoke("return json", system_prompt="你是长期记忆规划器。")

    assert client.calls[0]["messages"][0] == {
        "role": "system",
        "content": "你是长期记忆规划器。",
    }
    assert client.calls[0]["messages"][1] == {"role": "user", "content": "return json"}


async def test_deepseek_llm_records_prompt_response_and_usage():
    class UsageClient(FakeAsyncClient):
        async def create(self, **kwargs):
            self.calls.append(kwargs)

            class Message:
                content = "traced response"

            class Choice:
                message = Message()

            class Usage:
                prompt_tokens = 10
                completion_tokens = 3
                total_tokens = 13

                def model_dump(self):
                    return {
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": self.completion_tokens,
                        "total_tokens": self.total_tokens,
                    }

            class Response:
                choices = [Choice()]
                usage = Usage()

            return Response()

    store = LLMTraceStore(path=trace_dir("success") / "traces.jsonl", max_records=10)
    llm = DeepSeekChatLLM(
        api_key="secret",
        client=UsageClient(),
        trace_store=store,
    )

    result = await llm.ainvoke(
        "hello trace",
        response_format={"type": "json_object"},
        trace_context={
            "component": "chat_agent",
            "operation": "decision",
            "current_user_message": "[QQ:123456 昵称:tester] hello",
        },
    )

    assert result == "traced response"
    trace = store.snapshot()["traces"][0]
    assert trace["status"] == "success"
    assert llm.last_trace_id == trace["trace_id"]
    assert trace["component"] == "chat_agent"
    assert trace["operation"] == "decision"
    assert trace["current_user_message"] == "[QQ:123456 昵称:tester] hello"
    assert trace["model"] == "deepseek-flash"
    assert trace["thinking"] == "enabled"
    assert trace["temperature"] == 0.7
    assert trace["response_format"] == {"type": "json_object"}
    assert trace["messages"][1] == {"role": "user", "content": "hello trace"}
    assert trace["response_text"] == "traced response"
    assert trace["usage"]["total_tokens"] == 13
    assert trace["duration_ms"] >= 0


async def test_deepseek_llm_records_reasoning_content_when_available():
    class ReasoningClient(FakeAsyncClient):
        async def create(self, **kwargs):
            self.calls.append(kwargs)

            class Message:
                content = "final answer"
                reasoning_content = "先判断上下文，再组织回复。"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    store = LLMTraceStore(path=trace_dir("reasoning") / "traces.jsonl", max_records=10)
    llm = DeepSeekChatLLM(
        api_key="secret",
        client=ReasoningClient(),
        trace_store=store,
    )

    result = await llm.ainvoke(
        "hello trace",
        trace_context={"component": "chat_agent", "operation": "decision"},
    )

    assert result == "final answer"
    trace = store.snapshot()["traces"][0]
    assert trace["response_text"] == "final answer"
    assert trace["reasoning_content"] == "先判断上下文，再组织回复。"


async def test_deepseek_llm_records_error_trace_and_reraises():
    class FailingClient(FakeAsyncClient):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("api_key=sk-secret123456 failed")

    store = LLMTraceStore(path=trace_dir("error") / "traces.jsonl", max_records=10)
    llm = DeepSeekChatLLM(
        api_key="secret",
        client=FailingClient(),
        trace_store=store,
    )

    try:
        await llm.ainvoke(
            "hello trace",
            trace_context={"component": "memory_planner", "operation": "plan_memory"},
        )
    except RuntimeError as exc:
        assert "failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    trace = store.snapshot()["traces"][0]
    assert trace["status"] == "error"
    assert trace["component"] == "memory_planner"
    assert trace["operation"] == "plan_memory"
    assert trace["error_type"] == "RuntimeError"
    assert "[REDACTED]" in trace["error_message"]


def test_factory_uses_env_key_and_default_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.model == "deepseek-flash"
    assert llm.thinking == "enabled"


def test_factory_reads_model_from_env_and_allows_explicit_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    assert create_deepseek_chat_llm().model == "deepseek-v4-flash"
    assert create_deepseek_chat_llm(model="deepseek-v4-pro").model == "deepseek-v4-pro"


def test_factory_reads_model_from_dotenv(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setattr(
        "qq_group_chatter.agent.deepseek_llm._read_dotenv_key",
        lambda name="DEEPSEEK_API_KEY": "deepseek-v4-flash" if name == "DEEPSEEK_MODEL" else None,
    )

    assert create_deepseek_chat_llm().model == "deepseek-v4-flash"


def test_factory_allows_disabling_thinking_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", "false")

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.thinking == "disabled"


def test_factory_allows_enabling_thinking_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", "true")

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.thinking == "enabled"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("enabled", "enabled"), ("disabled", "disabled")],
)
def test_factory_allows_legacy_thinking_mode_aliases(monkeypatch, raw_value, expected):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", raw_value)

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.thinking == expected


def test_factory_reads_thinking_from_dotenv(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
    monkeypatch.setattr(
        "qq_group_chatter.agent.deepseek_llm._read_dotenv_key",
        lambda name="DEEPSEEK_API_KEY": "false" if name == "DEEPSEEK_THINKING" else None,
    )

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.thinking == "disabled"


def test_factory_allows_flash_for_background_tasks(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)

    llm = create_deepseek_chat_llm(model="deepseek-v4-flash")

    assert llm is not None
    assert llm.model == "deepseek-v4-flash"
    assert llm.thinking == "enabled"


def test_factory_allows_optional_max_tokens(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    llm = create_deepseek_chat_llm(max_tokens=1200)

    assert llm is not None
    assert llm.max_tokens == 1200


def test_factory_returns_none_when_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("qq_group_chatter.agent.deepseek_llm._read_dotenv_key", lambda: None)

    assert create_deepseek_chat_llm() is None


def test_factory_reads_key_from_dotenv_file(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        "qq_group_chatter.agent.deepseek_llm._read_dotenv_key",
        lambda name="DEEPSEEK_API_KEY": "from-dotenv" if name == "DEEPSEEK_API_KEY" else None,
    )

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.api_key == "from-dotenv"


def test_read_dotenv_key_allows_utf8_bom(monkeypatch):
    class FakePath:
        def __init__(self, path):
            self.path = path

        def exists(self):
            return True

        def read_text(self, *, encoding):
            assert encoding == "utf-8-sig"
            return "DEEPSEEK_API_KEY=from-bom-dotenv\n"

    monkeypatch.setattr("qq_group_chatter.agent.deepseek_llm.Path", FakePath)

    assert _read_dotenv_key(path=".env") == "from-bom-dotenv"
