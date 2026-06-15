import sys
from types import SimpleNamespace

import pytest

from qq_group_chatter.app import MemoryConfigurationError, NoopMem0Client, create_default_mem0_client


class FakeMemory:
    created = []

    @classmethod
    def from_config(cls, config):
        cls.created.append(config)
        return {"client": "mem0", "config": config}


class BrokenMemory:
    @classmethod
    def from_config(cls, config):
        raise ImportError("FastEmbed is not installed")


def test_default_mem0_client_uses_mem0_when_deepseek_key_exists(monkeypatch):
    FakeMemory.created.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=FakeMemory))

    client = create_default_mem0_client()

    assert client["client"] == "mem0"
    assert client["config"]["llm"]["provider"] == "deepseek"
    assert client["config"]["llm"]["config"]["api_key"] == "secret"
    assert client["config"]["llm"]["config"]["model"] == "deepseek-v4-flash"
    assert client["config"]["llm"]["config"]["deepseek_base_url"] == "https://api.deepseek.com"
    assert client["config"]["embedder"]["provider"] == "fastembed"
    assert client["config"]["embedder"]["config"]["model"] == "BAAI/bge-small-zh-v1.5"


def test_default_mem0_client_wraps_mem0_initialization_errors(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=BrokenMemory))

    with pytest.raises(MemoryConfigurationError, match="Failed to initialize Mem0"):
        create_default_mem0_client()


def test_default_mem0_client_raises_without_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("qq_group_chatter.app._read_dotenv_key", lambda name="DEEPSEEK_API_KEY": None)

    with pytest.raises(MemoryConfigurationError, match="DEEPSEEK_API_KEY"):
        create_default_mem0_client()


def test_default_mem0_client_uses_configured_fastembed_model(monkeypatch):
    FakeMemory.created.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("MEM0_FASTEMBED_MODEL", "jinaai/jina-embeddings-v2-base-zh")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=FakeMemory))

    client = create_default_mem0_client()

    assert client["client"] == "mem0"
    assert client["config"]["embedder"]["provider"] == "fastembed"
    assert client["config"]["embedder"]["config"]["model"] == "jinaai/jina-embeddings-v2-base-zh"


def test_default_long_term_memory_service_uses_deepseek_extractor(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    from qq_group_chatter.app import create_default_long_term_memory_service

    service = create_default_long_term_memory_service(mem0_client=NoopMem0Client())

    assert service._extractor._llm.model == "deepseek-v4-flash"
    assert service._extractor._llm.thinking == "disabled"


def test_noop_mem0_client_accepts_real_add_signature():
    client = NoopMem0Client()

    assert client.add(
        [{"role": "user", "content": "用户不吃辣"}],
        user_id="qq_user:123456",
        metadata={"scope": "user"},
        infer=False,
    ) == {"id": None}
