import asyncio
import base64
import gc
import json
import socket
import re
import shutil
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qq_group_chatter.agent.chat_agent import ChatAgent
from qq_group_chatter.agent.deepseek_llm import DeepSeekChatLLM
from qq_group_chatter.app import NoopMem0Client, create_default_image_service
from qq_group_chatter.llm_tracing import LLMTraceStore
from qq_group_chatter.memory_dashboard import (
    build_llm_traces_snapshot, image_preview_api, memory_dashboard_html,
)
from qq_group_chatter.models import (
    ChatMessage, ImageAttachment, LongTermMemoryBundle,
    LongTermMemoryIngestionJob, build_group_conversation_context,
)
from qq_group_chatter.orchestrator import ChatOrchestrator
from qq_group_chatter.plugins.chat import _handle_message_event, _image_attachments_from_event
from qq_group_chatter.services.image_understanding import (
    ImageReadError, ImageUnderstandingService, public_connection, validate_public_url,
)
from qq_group_chatter.services.long_term_memory import LongTermMemoryService
from qq_group_chatter.services.long_term_memory_planner import LongTermMemoryPlanner
from qq_group_chatter.services.short_term_memory import ShortTermMemoryService
from qq_group_chatter.services.conversation_archive import ConversationArchiveService


# A real 1x1 PNG; no network or production memory is used by these tests.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


def workdir():
    path = Path("tests/.tmp/images") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def context(message_id="current", reply_to=None):
    return build_group_conversation_context(
        group_id=888, user_id=123, message_id=message_id, nickname="tester",
        timestamp=123.0, reply_to_message_id=reply_to,
    )


def message(message_id, attachments):
    return ChatMessage("qq_group:888", "user", "[图片]", "123", "tester", message_id, 123.0, attachments)


async def download(url):
    return PNG + url.encode()  # distinct resources, PNG signature retained


async def prepared(service, number=1):
    return await service.prepare([
        ImageAttachment(i + 1, f"https://example.com/{i}.png", file=f"{i}.png")
        for i in range(number)
    ])


def test_default_config_and_disable(monkeypatch):
    monkeypatch.setattr("qq_group_chatter.app._read_dotenv_key", lambda *args: None)
    for name in ("IMAGE_UNDERSTANDING_ENABLED", "IMAGE_UNDERSTANDING_MAX_IMAGES_PER_REQUEST"):
        monkeypatch.delenv(name, raising=False)
    service = create_default_image_service()
    assert service.enabled and service.max_images == 2
    monkeypatch.setenv("IMAGE_UNDERSTANDING_ENABLED", "false")
    assert not create_default_image_service().enabled


async def test_receive_image_metadata_and_persist_across_restart():
    root = workdir()
    event = SimpleNamespace(message=[
        {"type": "text", "data": {"text": "看图"}},
        {"type": "image", "data": {"url": "https://example.com/a", "file": "a"}},
        {"type": "image", "data": {"file": "b"}},
    ])
    attachments = _image_attachments_from_event(event)
    assert [(a.position, a.file) for a in attachments] == [(1, "a"), (2, "b")]
    service = ImageUnderstandingService(path=root / "cache", downloader=download)
    attachments = await service.prepare(attachments)
    assert [a.status for a in attachments] == ["ready", "missing_url"]
    short = ShortTermMemoryService(path=root / "short.jsonl")
    await short.add_message(message("current", attachments))
    reloaded = ShortTermMemoryService(path=root / "short.jsonl")
    history = await reloaded.get_recent("qq_group:888")
    snapshot = service.snapshot(history, context("followup", "current"))
    assert len(snapshot.images) == 1
    assert snapshot.attachments[0]["is_current"] is False
    assert history[0].attachments == attachments
    assert "base64" not in (root / "short.jsonl").read_text(encoding="utf-8")
    assert "https://example.com" not in (root / "short.jsonl").read_text(encoding="utf-8")


async def test_snapshot_priority_deduplication_and_limits():
    service = ImageUnderstandingService(path=workdir(), downloader=download, max_images=4)
    images = await prepared(service, 8)
    messages = [message("quoted", [images[2]]), message("older", [images[3]]),
                message("newer", [images[4], images[0]]), message("current", images[:2])]
    snapshot = service.snapshot(messages, context(reply_to="quoted"))
    assert [i.resource_id for i in snapshot.images] == [images[i].resource_id for i in [0, 1, 2, 4]]
    assert snapshot.attachments[-1]["status"] == "request_limit"
    assert any(a["status"] == "sent_duplicate" for a in snapshot.attachments)
    assert [a["is_current"] for a in snapshot.attachments[:2]] == [True, True]


async def test_disable_skips_download_and_all_historical_images():
    root = workdir()
    service = ImageUnderstandingService(path=root, downloader=download)
    images = await prepared(service)

    async def forbidden(url):
        raise AssertionError("disabled service downloaded an image")

    disabled = ImageUnderstandingService(path=root, enabled=False, downloader=forbidden)
    incoming = await disabled.prepare([ImageAttachment(1, "https://example.com/new")])
    assert incoming[0].status == "disabled"
    snapshot = disabled.snapshot([message("old", images), message("current", incoming)], context())
    assert not snapshot.images
    assert all(a["status"] == "disabled" for a in snapshot.attachments)
    assert service.read_resource(images[0].resource_id) is not None


@pytest.mark.parametrize("url", ["file:///a.png", "http://127.0.0.1/a", "http://[::1]/a",
                                "http://192.168.0.1/a", "https://localhost/a", "http://a:b@example.com/a"])
def test_private_and_invalid_urls_rejected(url):
    with pytest.raises(ImageReadError):
        validate_public_url(url)


def test_dns_private_address_rejected_before_socket_connect(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
    ])
    with pytest.raises(ImageReadError, match="private_address"):
        public_connection(("example.com", 80))


async def test_download_failures_and_per_message_cap():
    async def fetch(url):
        if url.endswith("timeout"):
            raise TimeoutError()
        if url.endswith("large"):
            return PNG * 100
        return b"not an image"

    service = ImageUnderstandingService(path=workdir(), downloader=fetch, max_bytes=100, max_per_message=4)
    results = await service.prepare([
        ImageAttachment(1), ImageAttachment(2, "https://example.com/timeout"),
        ImageAttachment(3, "https://example.com/large"), ImageAttachment(4, "https://example.com/bad"),
        ImageAttachment(5, "https://example.com/ignored"),
    ])
    assert [a.status for a in results] == ["missing_url", "timeout", "too_large", "unsupported_format", "message_limit"]


async def test_cleanup_preserves_short_term_and_live_snapshot():
    service = ImageUnderstandingService(path=workdir(), downloader=download)
    short = ShortTermMemoryService()
    images = await prepared(service)
    await short.add_message(message("current", images))
    await service.start(short.referenced_image_ids)
    try:
        service.collect_unused(grace_seconds=0)
        assert service.read_resource(images[0].resource_id)
        snapshot = service.snapshot(await short.get_recent("qq_group:888"), context())
        resource_id = images[0].resource_id
        del images
        await short.clear("qq_group:888")
        service.collect_unused(grace_seconds=0)
        assert service.read_resource(resource_id)
        del snapshot
        await asyncio.sleep(0)  # Let gather callbacks release completed download tasks.
        gc.collect()
        service.collect_unused(grace_seconds=0)
        assert service.read_resource(resource_id) is None
    finally:
        await service.stop()


class Client:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reply.startswith('{"action":"web_search"'):
            data = json.loads(self.reply)
            message = SimpleNamespace(content="", tool_calls=[{"id": "search-1", "type": "function",
                "function": {"name": "web_search", "arguments": json.dumps({"query": data["query"]})}}], reasoning_content="thinking")
        else:
            message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_full_chat_search_ingestion_share_snapshot_and_trace_has_no_base64():
    root = workdir()
    service = ImageUnderstandingService(path=root / "cache", downloader=download)
    store = LLMTraceStore(root / "trace.jsonl")
    client = Client('{"action":"web_search","notice":"查一下","query":"图片中的错误"}')
    llm = DeepSeekChatLLM(api_key="test", client=client, trace_store=store)
    planner_client = Client('{"operations":[]}')
    planner = LongTermMemoryPlanner(llm=DeepSeekChatLLM(api_key="test", client=planner_client, trace_store=store))

    class Memory:
        job = None

        async def search(self, *args):
            return LongTermMemoryBundle([], [])

        async def enqueue_ingestion(self, job):
            self.job = job

    class Search:
        async def search_sources(self, query):
            client.reply = "截图里的错误可以这样处理。"
            return [SimpleNamespace(title="result", content="text", raw_content="body")]

    class Archive:
        def __init__(self):
            self.messages = []

        async def enqueue_message(self, msg):
            self.messages.append(msg)

        async def search(self, *args):
            return []

    from qq_group_chatter.agent.tools import ToolRegistry, AgentTool
    registry = ToolRegistry()
    async def search(query):
        await Search().search_sources(query)
        return {"results": [{"title": "result", "content": "body"}]}
    registry.register(AgentTool("web_search", "search", {"type": "object"}, search))
    memory = Memory()
    archive = Archive()
    orchestrator = ChatOrchestrator(
        short_term_memory=ShortTermMemoryService(), long_term_memory=memory,
        chat_agent=ChatAgent(llm, tools=registry), web_search=Search(), image_service=service,
        conversation_archive=archive,
    )
    reply = await orchestrator.handle_message(context=context(), user_message="看看报错 [图片]",
        attachments=[ImageAttachment(1, "https://example.com/error")])
    await orchestrator.record_assistant_reply(reply)
    assert memory.job.image_snapshot is reply.image_snapshot
    ingestion = LongTermMemoryService(mem0_client=NoopMem0Client(), planner=planner)
    assert await ingestion._process_job(memory.job) is None
    calls = [*client.calls, *planner_client.calls]
    assert len(calls) == 3
    round_ids = {trace["round_id"] for trace in store.snapshot()["traces"]}
    assert len(round_ids) == 1 and None not in round_ids
    assert reply.context.round_id in round_ids
    await orchestrator.handle_message(context=context(), user_message="另一次请求")
    assert len({trace["round_id"] for trace in store.snapshot()["traces"]}) == 2
    assert all("response_format" not in call for call in client.calls)
    assert planner_client.calls[0]["response_format"] == {"type": "json_object"}
    image_blocks = [call["messages"][1]["content"][1:] for call in calls]
    assert image_blocks[0] == image_blocks[1] == image_blocks[2]
    assert image_blocks[0][0] == {"type": "text", "text": "图1"}
    assert image_blocks[0][1]["image_url"]["url"].startswith("data:image/png;base64,")
    traces = store.snapshot()["traces"]
    assert all(t["image_context"]["sent_count"] == 1 for t in traces)
    assert "data:image" not in store.path.read_text(encoding="utf-8")
    assert all(t["messages"][1]["content"][2]["type"] == "image_ref" for t in traces)
    assert "历史图片仅供指代理解" in planner_client.calls[0]["messages"][0]["content"]
    for call in calls:
        prompt = call["messages"][1]["content"][0]["text"]
        assert "[图1]" in prompt
        assert "resource_id" not in prompt and "duration_ms" not in prompt


async def test_console_preview_settings_and_unavailable_resources():
    service = ImageUnderstandingService(path=workdir(), downloader=download)
    images = await prepared(service)
    app = SimpleNamespace(image_service=service, llm_trace_store=LLMTraceStore(workdir() / "trace.jsonl"))
    snapshot = build_llm_traces_snapshot(app)
    assert snapshot["image_understanding"]["enabled"] is True
    assert snapshot["image_understanding"]["max_images_per_request"] == 2
    response = await image_preview_api(app, images[0].resource_id)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert (await image_preview_api(app, "../secret")).status_code == 404
    assert (await image_preview_api(app, "0" * 64)).status_code == 404


async def test_unaddressed_group_image_is_cached_without_llm_or_napcat_calls(monkeypatch):
    import qq_group_chatter.plugins.chat as plugin

    service = ImageUnderstandingService(path=workdir(), downloader=download)
    short = ShortTermMemoryService()
    client = Client('{"action":"reply","content":"ok"}')
    orchestrator = ChatOrchestrator(
        short_term_memory=short, long_term_memory=SimpleNamespace(), image_service=service,
        chat_agent=ChatAgent(DeepSeekChatLLM(api_key="test", client=client)),
    )
    monkeypatch.setattr(plugin, "_context_from_event", lambda event: context("earlier"))
    event = SimpleNamespace(user_id=123, self_id=456, message_type="group", to_me=False,
        message=[{"type": "image", "data": {"url": "https://example.com/cat", "file": "cat"}}])

    class ForbiddenBot:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected bot call: {name}")

    await _handle_message_event(ForbiddenBot(), event, orchestrator)
    history = await short.get_recent(context().conversation_id)
    assert history[0].attachments[0].status == "ready"
    assert client.calls == []
    followup = service.snapshot(history, context("followup", "earlier"))
    assert len(followup.images) == 1


@pytest.mark.parametrize("enabled", [True, False])
async def test_actual_llm_respects_two_image_limit_and_global_disable(enabled):
    root = workdir()
    service = ImageUnderstandingService(path=root, downloader=download)
    attachments = await prepared(service, 8)
    service.enabled = enabled
    snapshot = service.snapshot([message("current", attachments)], context())
    chat_client = Client('{"action":"reply","content":"ok"}')
    planner_client = Client('{"operations":[]}')
    await ChatAgent(DeepSeekChatLLM(api_key="test", client=chat_client)).generate_reply(
        user_message="看图", context=context(), short_term_messages=[],
        long_term_memory=LongTermMemoryBundle([], []), image_snapshot=snapshot,
    )
    await LongTermMemoryPlanner(llm=DeepSeekChatLLM(api_key="test", client=planner_client)).plan(
        user_message="看图", context=context(), user_memories=[], conversation_memories=[], image_snapshot=snapshot,
    )
    for client in (chat_client, planner_client):
        blocks = client.calls[0]["messages"][1]["content"]
        assert sum(block["type"] == "image_url" for block in blocks) == (2 if enabled else 0)
        assert ("[图片未读取：超限]" if enabled else "[图片未读取：功能关闭]") in blocks[0]["text"]
        if enabled:
            assert snapshot.render_content("", "current") == "[图1] [图2] " + " ".join(["[图片未读取：超限]"] * 6)
            assert [block["text"] for block in blocks[1:] if block["type"] == "text"] == ["图1", "图2"]
            assert "[图3]" not in blocks[0]["text"]


async def test_long_term_and_archive_store_only_text_not_image_references():
    class RecordingMem0(NoopMem0Client):
        def __init__(self):
            self.writes = []

        def add(self, messages, **kwargs):
            self.writes.append({"messages": messages, **kwargs})
            return {"id": None}

    service = ImageUnderstandingService(path=workdir(), downloader=download)
    attachments = await prepared(service)
    snapshot = service.snapshot([message("current", attachments)], context())
    client = Client(json.dumps({"operations": [{"action": "add", "scope": "user", "content": "养了一只橘猫",
        "kind": "other", "confidence": 0.95}]}))
    mem0 = RecordingMem0()
    memory = LongTermMemoryService(mem0_client=mem0,
        planner=LongTermMemoryPlanner(llm=DeepSeekChatLLM(api_key="test", client=client)))
    job = LongTermMemoryIngestionJob(context=context(), user_message="这是我的猫 [图片]",
        existing_memories=LongTermMemoryBundle([], []), image_snapshot=snapshot)
    assert await memory._process_job(job) is None
    archive = ConversationArchiveService(mem0_client=mem0)
    await archive.enqueue_message(message("current", attachments))
    serialized = json.dumps(mem0.writes, ensure_ascii=False)
    assert len(mem0.writes) == 2
    assert "养了一只橘猫" in serialized
    assert attachments[0].resource_id not in serialized
    assert all(write["infer"] is False for write in mem0.writes)


def test_download_redirect_to_private_address_is_not_followed(monkeypatch):
    calls = []

    class Connection:
        def __init__(self, host, port, **kwargs):
            calls.append(host)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(status=302, getheader=lambda *args: "http://127.0.0.1/private")

        def close(self):
            pass

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    service = ImageUnderstandingService(path=workdir())
    with pytest.raises(ImageReadError, match="private_address"):
        service._download_sync("https://example.com/image")
    assert calls == ["example.com"]


async def test_prepared_images_are_pinned_before_short_term_handoff():
    service = ImageUnderstandingService(path=workdir(), downloader=download)
    images = await prepared(service)
    resource_id = images[0].resource_id
    service.collect_unused(grace_seconds=0)
    assert service.read_resource(resource_id)
    del images
    await asyncio.sleep(0)
    gc.collect()
    service.collect_unused(grace_seconds=0)
    assert service.read_resource(resource_id) is None


async def test_download_concurrency_and_timeout_are_enforced():
    active = 0
    maximum = 0

    async def fetch(url):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.005)
            return PNG
        finally:
            active -= 1

    service = ImageUnderstandingService(path=workdir(), downloader=fetch, concurrency=3)
    assert all(a.status == "ready" for a in await prepared(service, 8))
    assert maximum == 3

    async def stuck(url):
        await asyncio.Future()

    service = ImageUnderstandingService(path=workdir(), downloader=stuck, timeout=0.1)
    result = await asyncio.wait_for(prepared(service), timeout=1)
    assert result[0].status == "timeout"


async def test_planner_image_reference_is_rejected_before_mem0_write():
    service = ImageUnderstandingService(path=workdir(), downloader=download)
    images = await prepared(service)
    snapshot = service.snapshot([message("current", images)], context())

    class RecordingMem0(NoopMem0Client):
        def add(self, *args, **kwargs):
            raise AssertionError("image reference reached Mem0")

    client = Client(json.dumps({"operations": [{"action": "add", "scope": "user",
        "content": "猫的照片：" + images[0].resource_id, "kind": "other", "confidence": 0.95}]}))
    memory = LongTermMemoryService(mem0_client=RecordingMem0(),
        planner=LongTermMemoryPlanner(llm=DeepSeekChatLLM(api_key="test", client=client)))
    assert await memory._process_job(LongTermMemoryIngestionJob(context=context(), user_message="记住我的猫",
        existing_memories=LongTermMemoryBundle([], []), image_snapshot=snapshot)) is None




async def test_message_labels_link_current_history_quote_and_archive_without_mutation():
    from qq_group_chatter.models import ConversationArchiveRecord

    service = ImageUnderstandingService(path=workdir(), downloader=download, max_images=3)
    images = await prepared(service, 3)
    ctx = context(reply_to="quoted")
    history = [replace(message("quoted", [images[2]]), content="引用正文", user_id="456"),
               replace(message("duplicate", [images[0]]), content="重复正文"),
               replace(message("current", images[:2]), content="字面量[图片]")]
    snapshot = service.snapshot(history, ctx)
    records = [ConversationArchiveRecord("引用正文", "user", "456", "other", "quoted", 123),
               ConversationArchiveRecord("[图片]", "user", "123", "tester", "old", 123)]
    agent = ChatAgent()
    prompt = agent._build_prompt(user_message="字面量[图片]", context=ctx,
        short_term_messages=history, long_term_memory=LongTermMemoryBundle([], []),
        conversation_archive=records, image_snapshot=snapshot)
    assert "字面量[图片] [图1] [图2]" in prompt
    assert "重复正文 [图1]" in prompt
    assert prompt.count("引用正文 [图3]") == 3  # history, quote, semantic archive
    assert "[历史图片未提供]" in prompt
    assert "resource_id" not in prompt and "input_index" not in prompt
    assert history[-1].content == "字面量[图片]"
    assert records[0].content == "引用正文"
    assert snapshot.render_content("literal [图片]", "unknown") == "literal [图片]"


@pytest.mark.parametrize("status,reason", [
    ("download_failed", "下载失败"), ("request_limit", "超限"),
    ("disabled", "功能关闭"), ("unavailable", "缓存不可用"),
    ("missing_url", "缺少下载地址"), ("timeout", "下载超时"),
])
def test_unread_images_have_reason_and_no_number(status, reason):
    from qq_group_chatter.models import ImageSnapshot

    snapshot = ImageSnapshot(attachments=[{"message_id": "m", "position": 1, "status": status}])
    assert snapshot.render_content("正文", "m") == f"正文 [图片未读取：{reason}]"
    assert snapshot.render_content("其他消息", "other") == "其他消息"
    assert snapshot.render_content("[图片] 回答我", "m") == f"[图片未读取：{reason}] 回答我"


async def test_image_placeholders_are_rendered_in_place_without_duplicate_labels():
    service = ImageUnderstandingService(path=workdir(), downloader=download)
    images = await prepared(service, 2)
    snapshot = service.snapshot([message("current", images)], context())
    assert snapshot.render_content("[图片] 回答我 [图片]", "current") == "[图1] 回答我 [图2]"
    assert snapshot.render_content("[图片]", "current") == "[图1] [图2]"
    assert snapshot.render_content("[图片] [图片] [图片]", "current") == "[图1] [图2] [图片]"
    assert snapshot.render_content("字面量[图片]", "current") == "字面量[图片] [图1] [图2]"
    assert snapshot.render_content("[图片] 回答我", "unknown") == "[图片] 回答我"
