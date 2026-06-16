import uuid
from pathlib import Path

from qq_group_chatter.agent.deepseek_llm import (
    DeepSeekChatLLM,
    _read_dotenv_key,
    create_deepseek_chat_llm,
)
from qq_group_chatter.llm_tracing import LLMTraceStore


def trace_dir(name):
    path = Path("tests/.tmp/deepseek-tracing") / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


async def test_deepseek_llm_calls_v4_pro_with_thinking_disabled():
    client = FakeAsyncClient()
    llm = DeepSeekChatLLM(api_key="secret", client=client)

    result = await llm.ainvoke("hello")

    assert result == "OK"
    assert client.calls[0]["model"] == "deepseek-v4-pro"
    assert "max_tokens" not in client.calls[0]
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
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
        trace_context={"component": "chat_agent", "operation": "decision"},
    )

    assert result == "traced response"
    trace = store.snapshot()["traces"][0]
    assert trace["status"] == "success"
    assert trace["component"] == "chat_agent"
    assert trace["operation"] == "decision"
    assert trace["model"] == "deepseek-v4-pro"
    assert trace["thinking"] == "disabled"
    assert trace["temperature"] == 0.7
    assert trace["response_format"] == {"type": "json_object"}
    assert trace["messages"][1] == {"role": "user", "content": "hello trace"}
    assert trace["response_text"] == "traced response"
    assert trace["usage"]["total_tokens"] == 13
    assert trace["duration_ms"] >= 0


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

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.model == "deepseek-v4-pro"
    assert llm.thinking == "disabled"


def test_factory_allows_flash_for_background_tasks(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    llm = create_deepseek_chat_llm(model="deepseek-v4-flash")

    assert llm is not None
    assert llm.model == "deepseek-v4-flash"
    assert llm.thinking == "disabled"


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
        lambda: "from-dotenv",
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
