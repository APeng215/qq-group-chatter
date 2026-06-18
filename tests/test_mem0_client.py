import os
import sys
from types import SimpleNamespace

import pytest

from qq_group_chatter.app import (
    MemoryConfigurationError,
    NoopMem0Client,
    create_default_conversation_archive_service,
    create_default_mem0_client,
)


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
    monkeypatch.delenv("MEM0_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=FakeMemory))

    client = create_default_mem0_client()

    assert client["client"] == "mem0"
    assert client["config"]["llm"]["provider"] == "deepseek"
    assert client["config"]["llm"]["config"]["api_key"] == "secret"
    assert client["config"]["llm"]["config"]["model"] == "deepseek-v4-pro"
    assert client["config"]["llm"]["config"]["deepseek_base_url"] == "https://api.deepseek.com"
    assert client["config"]["embedder"]["provider"] == "fastembed"
    assert client["config"]["embedder"]["config"]["model"] == "BAAI/bge-small-zh-v1.5"
    assert client["config"]["vector_store"]["config"]["embedding_model_dims"] == 512
    assert client["config"]["history_db_path"].endswith(".mem0\\history.db") or client["config"][
        "history_db_path"
    ].endswith(".mem0/history.db")
    assert "MEM0_DIR" in os.environ


def test_default_mem0_client_disables_mem0_telemetry_by_default(monkeypatch):
    FakeMemory.created.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("MEM0_TELEMETRY", raising=False)
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=FakeMemory))

    create_default_mem0_client()

    assert os.environ["MEM0_TELEMETRY"] == "false"


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


def test_default_mem0_client_collection_name_tracks_fastembed_model(monkeypatch):
    FakeMemory.created.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=FakeMemory))

    monkeypatch.setenv("MEM0_FASTEMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    first = create_default_mem0_client()
    monkeypatch.setenv("MEM0_FASTEMBED_MODEL", "jinaai/jina-embeddings-v2-base-zh")
    second = create_default_mem0_client()

    first_collection = first["config"]["vector_store"]["config"]["collection_name"]
    second_collection = second["config"]["vector_store"]["config"]["collection_name"]
    assert first_collection.startswith("qq_group_chatter_memories_")
    assert "bge_small_zh_v1_5" in first_collection
    assert first_collection.endswith("_512d")
    assert second_collection.startswith("qq_group_chatter_memories_")
    assert first_collection != second_collection


def test_default_conversation_archive_service_uses_separate_mem0_namespace(monkeypatch):
    FakeMemory.created.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=FakeMemory))

    service = create_default_conversation_archive_service()

    assert service is not None
    config = service._mem0["config"]
    collection = config["vector_store"]["config"]["collection_name"]
    assert collection.startswith("qq_group_chatter_archive_")
    assert config["vector_store"]["config"]["path"].endswith(".mem0\\qdrant-archive") or config[
        "vector_store"
    ]["config"]["path"].endswith(".mem0/qdrant-archive")
    assert config["history_db_path"].endswith(".mem0\\archive-history.db") or config[
        "history_db_path"
    ].endswith(".mem0/archive-history.db")


def test_default_long_term_memory_service_uses_deepseek_planner(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)

    from qq_group_chatter.app import create_default_long_term_memory_service

    service = create_default_long_term_memory_service(mem0_client=NoopMem0Client())

    assert service._planner._llm.model == "deepseek-v4-pro"
    assert service._planner._llm.thinking == "enabled"


def test_default_long_term_memory_service_can_disable_deepseek_thinking(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_THINKING", "false")

    from qq_group_chatter.app import create_default_long_term_memory_service

    service = create_default_long_term_memory_service(mem0_client=NoopMem0Client())

    assert service._planner._llm.thinking == "disabled"


def test_noop_mem0_client_accepts_real_add_signature():
    client = NoopMem0Client()

    assert client.add(
        [{"role": "user", "content": "用户不吃辣"}],
        user_id="qq_user:123456",
        metadata={"scope": "user"},
        infer=False,
    ) == {"id": None}


def test_noop_mem0_client_accepts_mutation_and_listing_signatures():
    client = NoopMem0Client()

    assert client.update("mem-1", "用户不吃辣", metadata={"scope": "user"}) == {"id": "mem-1"}
    assert client.delete("mem-1") == {"id": "mem-1"}
    assert client.get_all(filters={"user_id": "qq_user:123456"}, top_k=1000) == {"results": []}
