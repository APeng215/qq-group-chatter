import json

import pytest

from qq_group_chatter.services.web_search import (
    SearchAnswerAgent,
    SearchSource,
    TavilySearchClient,
    WebSearchConfigurationError,
    WebSearchResult,
    WebSearchService,
    create_default_web_search_service,
    parse_search_command,
)


def test_parse_search_command_accepts_low_risk_chinese_prefixes():
    assert parse_search_command("搜一下 DeepSeek 最新消息") == "DeepSeek 最新消息"
    assert parse_search_command("搜索 Tavily API") == "Tavily API"
    assert parse_search_command("查一下 今天新闻") == "今天新闻"


def test_parse_search_command_rejects_slash_prefixes_regular_chat_and_empty_query():
    assert parse_search_command("/搜 DeepSeek") is None
    assert parse_search_command("/搜索 Tavily") is None
    assert parse_search_command("帮我查一下 DeepSeek") is None
    assert parse_search_command("搜一下   ") is None


def test_tavily_payload_uses_basic_raw_content_and_no_answer(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"results": []}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("qq_group_chatter.services.web_search.request.urlopen", fake_urlopen)
    client = TavilySearchClient(
        api_key="secret",
        search_depth="basic",
        include_raw_content="markdown",
        include_answer=False,
        timeout_seconds=7,
    )

    assert client._search_sync("DeepSeek", max_results=3) == []
    assert captured["timeout"] == 7
    assert captured["payload"] == {
        "query": "DeepSeek",
        "search_depth": "basic",
        "max_results": 3,
        "include_answer": False,
        "include_raw_content": "markdown",
        "auto_parameters": False,
    }


def test_tavily_records_include_raw_content():
    record = TavilySearchClient._result_from_item(
        {
            "title": "标题",
            "url": "https://example.com/a",
            "content": "摘要",
            "raw_content": "# 正文\n很多内容",
        }
    )

    assert record == WebSearchResult(
        title="标题",
        url="https://example.com/a",
        content="摘要",
        raw_content="# 正文\n很多内容",
    )


class FakeSearchClient:
    def __init__(self):
        self.calls = []

    async def search(self, query, *, max_results):
        self.calls.append({"query": query, "max_results": max_results})
        return [
            WebSearchResult(
                title="来源一",
                url="https://example.com/one",
                content="摘要一",
                raw_content="正文一" * 20,
            ),
            WebSearchResult(
                title="来源二",
                url="https://example.com/two",
                content="摘要二",
                raw_content="正文二",
            ),
        ]


class FakeAnswerAgent:
    def __init__(self):
        self.calls = []

    async def answer(self, *, query, sources, include_urls):
        self.calls.append(
            {
                "query": query,
                "sources": sources,
                "include_urls": include_urls,
            }
        )
        return "整理后的答案\n参考来源：来源一；来源二"


async def test_web_search_service_uses_answer_agent_with_truncated_raw_content():
    client = FakeSearchClient()
    answer_agent = FakeAnswerAgent()
    service = WebSearchService(
        client=client,
        answer_agent=answer_agent,
        max_results=3,
        max_raw_content_chars_per_result=6,
        include_urls=False,
    )

    reply = await service.search_reply("天气")

    assert reply == "整理后的答案\n参考来源：来源一；来源二"
    assert client.calls == [{"query": "天气", "max_results": 3}]
    assert answer_agent.calls[0]["include_urls"] is False
    assert answer_agent.calls[0]["sources"][0] == SearchSource(
        title="来源一",
        url="https://example.com/one",
        content="摘要一",
        raw_content="正文一正文一",
    )


async def test_web_search_service_search_sources_returns_truncated_sources_without_answer_agent():
    client = FakeSearchClient()
    answer_agent = FakeAnswerAgent()
    service = WebSearchService(
        client=client,
        answer_agent=answer_agent,
        max_results=3,
        max_raw_content_chars_per_result=6,
        include_urls=False,
    )

    sources = await service.search_sources("天气")

    assert client.calls == [{"query": "天气", "max_results": 3}]
    assert answer_agent.calls == []
    assert sources == [
        SearchSource(
            title="来源一",
            url="https://example.com/one",
            content="摘要一",
            raw_content="正文一正文一",
        ),
        SearchSource(
            title="来源二",
            url="https://example.com/two",
            content="摘要二",
            raw_content="正文二",
        ),
    ]


async def test_web_search_service_search_sources_skips_results_without_raw_content():
    class MixedClient:
        async def search(self, query, *, max_results):
            return [
                WebSearchResult(
                    title="空来源",
                    url="https://example.com/empty",
                    content="摘要",
                    raw_content="   ",
                ),
                WebSearchResult(
                    title="有效来源",
                    url="https://example.com/valid",
                    content="摘要",
                    raw_content="正文",
                ),
            ]

    service = WebSearchService(client=MixedClient())

    sources = await service.search_sources("天气")

    assert sources == [
        SearchSource(
            title="有效来源",
            url="https://example.com/valid",
            content="摘要",
            raw_content="正文",
        )
    ]


async def test_web_search_service_returns_no_result_without_calling_answer_agent():
    class EmptyClient:
        async def search(self, query, *, max_results):
            return []

    answer_agent = FakeAnswerAgent()
    service = WebSearchService(client=EmptyClient(), answer_agent=answer_agent)

    reply = await service.search_reply("不存在的问题")

    assert "没有找到" in reply
    assert answer_agent.calls == []


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return "基于网页正文的答案"


async def test_search_answer_agent_builds_prompt_from_raw_content_without_urls():
    llm = RecordingLLM()
    agent = SearchAnswerAgent(llm=llm)

    reply = await agent.answer(
        query="DeepSeek 最新消息",
        sources=[
            SearchSource(
                title="来源标题",
                url="https://example.com/news",
                content="摘要",
                raw_content="原网页正文",
            )
        ],
        include_urls=False,
    )

    assert reply == "基于网页正文的答案"
    assert "DeepSeek 最新消息" in llm.prompts[0]
    assert "原网页正文" in llm.prompts[0]
    assert "https://example.com/news" not in llm.prompts[0]


def test_default_web_search_service_is_enabled_by_default_with_thinking_answer_agent(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    monkeypatch.setenv("WEB_SEARCH_ANSWER_MAX_TOKENS", "1200")
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("WEB_SEARCH_MAX_RESULTS", raising=False)

    service = create_default_web_search_service()

    assert service is not None
    assert isinstance(service._client, TavilySearchClient)
    assert service._max_results == 3
    assert service._answer_agent._llm.model == "deepseek-v4-pro"
    assert service._answer_agent._llm.thinking == "enabled"
    assert service._answer_agent._llm.max_tokens is None


def test_default_web_search_service_can_be_disabled(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    assert create_default_web_search_service() is None


def test_default_web_search_service_requires_tavily_key_when_enabled(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.setattr("qq_group_chatter.services.web_search._read_dotenv_key", lambda name="DEEPSEEK_API_KEY": None)

    with pytest.raises(WebSearchConfigurationError):
        create_default_web_search_service()
