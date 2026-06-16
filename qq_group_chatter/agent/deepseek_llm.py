from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qq_group_chatter.agent.identity import BOT_IDENTITY_PROMPT
from qq_group_chatter.prompt_loader import load_prompt


ThinkingMode = Literal["enabled", "disabled"]
DEEPSEEK_SYSTEM_PROMPT_TEMPLATE = load_prompt("deepseek_system.txt")


@dataclass
class DeepSeekChatLLM:
    api_key: str
    model: str = "deepseek-v4-pro"
    thinking: ThinkingMode = "disabled"
    temperature: float = 0.7
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    base_url: str = "https://api.deepseek.com"
    client: Any | None = None
    trace_store: Any | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is required for DeepSeekChatLLM. "
                "Install it with `python -m pip install openai`."
            ) from exc
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def ainvoke(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> str:
        resolved_response_format = (
            response_format if response_format is not None else self.response_format
        )
        messages = [
            {
                "role": "system",
                "content": DEEPSEEK_SYSTEM_PROMPT_TEMPLATE.format(
                    bot_identity_prompt=BOT_IDENTITY_PROMPT,
                ),
            },
            {"role": "user", "content": prompt},
        ]
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "extra_body": {"thinking": {"type": self.thinking}},
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if resolved_response_format is not None:
            params["response_format"] = resolved_response_format

        trace_id = None
        start = time.perf_counter()
        if self.trace_store is not None:
            context = trace_context or {}
            trace_id = self.trace_store.record_start(
                component=str(context.get("component") or "unknown"),
                operation=str(context.get("operation") or "unknown"),
                model=self.model,
                thinking=self.thinking,
                temperature=self.temperature,
                response_format=resolved_response_format,
                messages=messages,
            )
        try:
            response = await self.client.chat.completions.create(**params)
        except Exception as exc:
            if self.trace_store is not None and trace_id is not None:
                self.trace_store.record_error(
                    trace_id=trace_id,
                    error=exc,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            raise

        content = response.choices[0].message.content or ""
        if self.trace_store is not None and trace_id is not None:
            self.trace_store.record_success(
                trace_id=trace_id,
                response_text=content,
                usage=_response_usage(response),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        return content


def create_deepseek_chat_llm(
    *,
    api_key: str | None = None,
    model: str = "deepseek-v4-pro",
    thinking: ThinkingMode = "disabled",
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    trace_store: Any | None = None,
) -> DeepSeekChatLLM | None:
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY") or _read_dotenv_key()
    if not resolved_key:
        return None
    return DeepSeekChatLLM(
        api_key=resolved_key,
        model=model,
        thinking=thinking,
        max_tokens=max_tokens,
        response_format=response_format,
        trace_store=trace_store,
    )


def _read_dotenv_key(name: str = "DEEPSEEK_API_KEY", path: str = ".env") -> str | None:
    env_path = Path(path)
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result or None
