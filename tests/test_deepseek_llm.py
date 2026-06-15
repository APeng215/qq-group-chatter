import pytest
from pathlib import Path

from qq_group_chatter.agent.deepseek_llm import DeepSeekChatLLM, create_deepseek_chat_llm


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
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert client.calls[0]["stream"] is False
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert client.calls[0]["messages"][1] == {"role": "user", "content": "hello"}


def test_factory_uses_env_key_and_default_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.model == "deepseek-v4-pro"
    assert llm.thinking == "disabled"


def test_factory_returns_none_when_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    temp_dir = Path("tests/.tmp/no_env_deepseek")
    temp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_dir)

    assert create_deepseek_chat_llm() is None


def test_factory_reads_key_from_dotenv_file(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    temp_dir = Path("tests/.tmp/deepseek_env_case")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.joinpath(".env").write_text("DEEPSEEK_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(temp_dir)

    llm = create_deepseek_chat_llm()

    assert llm is not None
    assert llm.api_key == "from-dotenv"
