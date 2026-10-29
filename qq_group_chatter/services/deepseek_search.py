from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import httpx
from qq_group_chatter.agent.tools import AgentTool, setting


class DeepSeekSearchClient:
    def __init__(self, api_key: str, *, timeout_seconds=8, transport=None):
        self._key = api_key
        self._timeout = timeout_seconds
        self._transport = transport

    async def _post(self, endpoint, payload):
        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            for attempt in range(2):
                response = await client.post(f"https://api.deepseek.com/v1/tools/{endpoint}",
                    headers={"Authorization": f"Bearer {self._key}"}, json=payload)
                if response.status_code == 200:
                    return response.json()
                if attempt == 0 and (response.status_code in {408, 429} or response.status_code >= 500):
                    await asyncio.sleep(0.25)
                    continue
                return {"error": f"http_{response.status_code}", "message": "DeepSeek search/fetch request failed"}

    async def search(self, query: str, *, max_results=5):
        if not 1 <= max_results <= 50:
            raise ValueError("WEB_SEARCH_MAX_RESULTS must be between 1 and 50")
        if not 1 <= len(query) <= 128 or not query.strip():
            return {"error": "invalid_query", "message": "Query must contain 1-128 characters and not be blank"}
        data = await self._post("search", {"query": query, "count": max_results, "need_content": True})
        if "error" in data:
            return data
        limit = int(setting("WEB_SEARCH_MAX_RAW_CONTENT_CHARS_PER_RESULT", "8000"))
        results = []
        for index, item in enumerate(data.get("results", [])[:max_results], 1):
            content = str(item.get("content") or "")
            results.append({**item, "source_id": f"source_{index}",
                            "content": content[:limit], "truncated": len(content) > limit})
        return {"results": results}

    async def fetch(self, url: str):
        try:
            parsed = urlsplit(url)
            valid = len(url) <= 8192 and parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if not valid:
            return {"error": "invalid_url", "message": "Use an HTTP(S) URL without embedded credentials (maximum 8192 characters)"}
        data = await self._post("fetch", {"url": url})
        if "error" in data:
            return data
        limit = int(setting("WEB_FETCH_MAX_CONTENT_CHARS", "20000"))
        content = str(data.get("content") or "")
        return {"url": url, "title": data.get("title", ""), "content": content[:limit],
                "truncated": len(content) > limit}

    def tools(self):
        async def search(query):
            return await self.search(query, max_results=int(setting("WEB_SEARCH_MAX_RESULTS", "5")))
        return [AgentTool("web_search", "Search current web sources; read results and search again or fetch pages when necessary.",
                    {"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"\S"}},
                     "required": ["query"], "additionalProperties": False}, search),
                AgentTool("web_fetch", "Fetch one HTTP(S) webpage as Markdown.",
                    {"type": "object", "properties": {"url": {"type": "string", "minLength": 1, "maxLength": 8192}},
                     "required": ["url"], "additionalProperties": False}, self.fetch)]
