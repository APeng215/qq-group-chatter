import json

import pytest

from qq_group_chatter.services.web_search import (
    SearchSource,
    TavilySearchClient,
    WebSearchConfigurationError,
    WebSearchResult,
    WebSearchService,
    create_default_web_search_service,
)


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


async def test_web_search_service_search_sources_returns_truncated_sources_without_answer_agent():
    client = FakeSearchClient()
    service = WebSearchService(
        client=client,
        max_results=3,
        max_raw_content_chars_per_result=6,
    )

    sources = await service.search_sources("天气")

    assert client.calls == [{"query": "天气", "max_results": 3}]
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


def test_default_web_search_service_is_enabled_by_default(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret")
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("WEB_SEARCH_MAX_RESULTS", raising=False)
    monkeypatch.delenv("WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT", raising=False)

    service = create_default_web_search_service()

    assert service is not None
    assert isinstance(service._client, TavilySearchClient)
    assert service._max_results == 3
    assert service._max_raw_content_chars_per_result == 5000


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
