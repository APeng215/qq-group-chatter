from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from qq_group_chatter.agent.deepseek_llm import _read_dotenv_key, create_deepseek_chat_llm
from qq_group_chatter.prompt_loader import load_prompt


SEARCH_PREFIXES = ("搜一下", "查一下", "搜索")
SEARCH_ANSWER_PROMPT_TEMPLATE = load_prompt("search_answer.txt")


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


class SearchAnswerAgent:
    def __init__(self, *, llm: Any | None = None):
        self._llm = llm

    async def answer(
        self,
        *,
        query: str,
        sources: list[SearchSource],
        include_urls: bool = False,
    ) -> str:
        if self._llm is None:
            return _fallback_search_answer(query, sources, include_urls=include_urls)
        prompt = SEARCH_ANSWER_PROMPT_TEMPLATE.format(
            query=query,
            sources=_format_sources_for_prompt(sources, include_urls=include_urls),
        )
        raw = await self._call_llm(prompt)
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw)

    async def _call_llm(self, prompt: str) -> Any:
        if hasattr(self._llm, "ainvoke"):
            return await self._llm.ainvoke(prompt)
        if hasattr(self._llm, "invoke"):
            return self._llm.invoke(prompt)
        if callable(self._llm):
            result = self._llm(prompt)
            if hasattr(result, "__await__"):
                return await result
            return result
        raise TypeError("llm must be callable or expose invoke/ainvoke")


class WebSearchService:
    def __init__(
        self,
        *,
        client: WebSearchClient,
        answer_agent: SearchAnswerAgent | Any | None = None,
        max_results: int = 3,
        max_raw_content_chars_per_result: int = 3000,
        include_urls: bool = False,
    ):
        self._client = client
        self._answer_agent = answer_agent or SearchAnswerAgent()
        self._max_results = max_results
        self._max_raw_content_chars_per_result = max_raw_content_chars_per_result
        self._include_urls = include_urls

    async def search_reply(self, query: str) -> str:
        results = await self._client.search(query, max_results=self._max_results)
        sources = _sources_from_results(
            results,
            max_raw_content_chars_per_result=self._max_raw_content_chars_per_result,
        )
        if not sources:
            return f"没有找到和「{query}」相关的可读网页正文。"
        return await self._answer_agent.answer(
            query=query,
            sources=sources,
            include_urls=self._include_urls,
        )


def parse_search_command(text: str) -> str | None:
    content = text.strip()
    for prefix in SEARCH_PREFIXES:
        if content == prefix:
            return None
        if content.startswith(prefix):
            query = content[len(prefix) :].strip()
            return query or None
    return None


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

    answer_llm = create_deepseek_chat_llm(
        model=os.getenv("WEB_SEARCH_ANSWER_MODEL")
        or _read_dotenv_key("WEB_SEARCH_ANSWER_MODEL")
        or "deepseek-v4-pro",
        thinking=_read_thinking("WEB_SEARCH_ANSWER_THINKING", "enabled"),
    )
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
        answer_agent=SearchAnswerAgent(llm=answer_llm),
        max_results=_read_int("WEB_SEARCH_MAX_RESULTS", 3),
        max_raw_content_chars_per_result=_read_int(
            "WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT",
            3000,
        ),
        include_urls=_read_bool("WEB_SEARCH_INCLUDE_URLS", False),
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


def _format_sources_for_prompt(sources: list[SearchSource], *, include_urls: bool) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        url_line = f"\nURL: {source.url}" if include_urls else ""
        blocks.append(
            f"[来源 {index}]\n"
            f"标题: {source.title or '无标题'}"
            f"{url_line}\n"
            f"摘要: {source.content or '无摘要'}\n"
            f"原网页正文:\n{source.raw_content}"
        )
    return "\n\n".join(blocks)


def _fallback_search_answer(
    query: str,
    sources: list[SearchSource],
    *,
    include_urls: bool,
) -> str:
    lines = [f"我找到了和「{query}」相关的网页正文，但搜索回答模型没有配置。"]
    lines.append("参考来源：" + "；".join(source.title or source.url for source in sources))
    if include_urls:
        lines.extend(source.url for source in sources if source.url)
    return "\n".join(lines)


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


def _read_thinking(env_name: str, default: str) -> str:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name) or default
    normalized = raw.strip().lower()
    return "enabled" if normalized == "enabled" else "disabled"
