from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from qq_group_chatter.agent.deepseek_llm import _read_dotenv_key


class WebSearchConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    content: str
    raw_content: str


@dataclass(frozen=True)
class SearchSource:
    title: str
    url: str
    content: str
    raw_content: str


class WebSearchClient(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]: ...


class TavilySearchClient:
    def __init__(
        self,
        *,
        api_key: str,
        search_depth: str = "basic",
        include_raw_content: str | bool = "markdown",
        include_answer: bool = False,
        timeout_seconds: float = 8.0,
    ):
        self._api_key = api_key
        self._search_depth = search_depth
        self._include_raw_content = include_raw_content
        self._include_answer = include_answer
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        return await asyncio.to_thread(
            self._search_sync,
            query,
            max_results=max_results,
        )

    def _search_sync(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        payload = json.dumps(
            {
                "query": query,
                "search_depth": self._search_depth,
                "max_results": max_results,
                "include_answer": self._include_answer,
                "include_raw_content": self._include_raw_content,
                "auto_parameters": False,
            }
        ).encode("utf-8")
        req = request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.HTTPError, json.JSONDecodeError) as exc:
            raise RuntimeError("Tavily search request failed.") from exc

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        records = []
        for item in results:
            if not isinstance(item, dict):
                continue
            record = self._result_from_item(item)
            if record is not None:
                records.append(record)
        return records

    @staticmethod
    def _result_from_item(item: dict[str, Any]) -> WebSearchResult | None:
        url = str(item.get("url") or "").strip()
        if not url:
            return None
        return WebSearchResult(
            title=str(item.get("title") or "").strip(),
            url=url,
            content=str(item.get("content") or "").strip(),
            raw_content=str(item.get("raw_content") or "").strip(),
        )


class WebSearchService:
    def __init__(
        self,
        *,
        client: WebSearchClient,
        max_results: int = 3,
        max_raw_content_chars_per_result: int = 5000,
    ):
        self._client = client
        self._max_results = max_results
        self._max_raw_content_chars_per_result = max_raw_content_chars_per_result

    async def search_sources(self, query: str) -> list[SearchSource]:
        results = await self._client.search(query, max_results=self._max_results)
        return _sources_from_results(
            results,
            max_raw_content_chars_per_result=self._max_raw_content_chars_per_result,
        )


def create_default_web_search_service() -> WebSearchService | None:
    if not _read_bool("WEB_SEARCH_ENABLED", True):
        return None

    provider = (
        os.getenv("WEB_SEARCH_PROVIDER")
        or _read_dotenv_key("WEB_SEARCH_PROVIDER")
        or "tavily"
    ).lower()
    if provider != "tavily":
        raise WebSearchConfigurationError(f"Unsupported web search provider: {provider}")

    tavily_key = os.getenv("TAVILY_API_KEY") or _read_dotenv_key("TAVILY_API_KEY")
    if not tavily_key:
        raise WebSearchConfigurationError("TAVILY_API_KEY is required when web search is enabled.")

    return WebSearchService(
        client=TavilySearchClient(
            api_key=tavily_key,
            search_depth=os.getenv("WEB_SEARCH_DEPTH")
            or _read_dotenv_key("WEB_SEARCH_DEPTH")
            or "basic",
            include_raw_content=_read_raw_content_mode("WEB_SEARCH_INCLUDE_RAW_CONTENT", "markdown"),
            include_answer=_read_bool("WEB_SEARCH_INCLUDE_ANSWER", False),
            timeout_seconds=_read_float("WEB_SEARCH_TIMEOUT_SECONDS", 8.0),
        ),
        max_results=_read_int("WEB_SEARCH_MAX_RESULTS", 3),
        max_raw_content_chars_per_result=_read_int(
            "WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT",
            5000,
        ),
    )


def _sources_from_results(
    results: list[WebSearchResult],
    *,
    max_raw_content_chars_per_result: int,
) -> list[SearchSource]:
    sources = []
    for result in results:
        raw_content = result.raw_content.strip()
        if not raw_content:
            continue
        sources.append(
            SearchSource(
                title=result.title,
                url=result.url,
                content=result.content,
                raw_content=raw_content[:max_raw_content_chars_per_result],
            )
        )
    return sources


def _read_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _read_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _read_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _read_raw_content_mode(env_name: str, default: str) -> str | bool:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return raw.strip()
