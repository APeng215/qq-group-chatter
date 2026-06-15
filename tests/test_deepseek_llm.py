from qq_group_chatter.agent.deepseek_llm import (
    DeepSeekChatLLM,
    _read_dotenv_key,
    create_deepseek_chat_llm,
)


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
    assert "牧野神奈" in client.calls[0]["messages"][0]["content"]
    assert "神奈" in client.calls[0]["messages"][0]["content"]
    assert "人类" not in client.calls[0]["messages"][0]["content"]
    assert "不要自称" in client.calls[0]["messages"][0]["content"]
    assert "AI" in client.calls[0]["messages"][0]["content"]
    assert "助手" in client.calls[0]["messages"][0]["content"]
    assert "模型" in client.calls[0]["messages"][0]["content"]
    assert "不要输出或解释思考过程" in client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["messages"][1] == {"role": "user", "content": "hello"}


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
