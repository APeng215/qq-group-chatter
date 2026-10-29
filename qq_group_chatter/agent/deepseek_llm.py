from __future__ import annotations

import os
import base64
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qq_group_chatter.agent.identity import BOT_IDENTITY_PROMPT
from qq_group_chatter.prompt_loader import load_prompt
from qq_group_chatter.models import ImageSnapshot


ThinkingMode = Literal["enabled", "disabled"]


@dataclass(frozen=True)
class ToolModelResponse:
    content: str
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None
    finish_reason: str | None
    trace_id: str | None

    def assistant_message(self) -> dict[str, Any]:
        message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        return message
DEEPSEEK_SYSTEM_PROMPT_TEMPLATE = load_prompt("deepseek_system.txt")


@dataclass
class DeepSeekChatLLM:
    api_key: str
    model: str = "deepseek-flash"
    thinking: ThinkingMode = "enabled"
    temperature: float = 0.7
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    base_url: str = "https://api.deepseek.com"
    client: Any | None = None
    trace_store: Any | None = None
    last_trace_id: str | None = None

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

    async def ainvoke_messages(self, messages: list[dict[str, Any]], *,
                               tools: list[dict[str, Any]], trace_context=None,
                               image_snapshot: ImageSnapshot | None = None,
                               tool_choice: str = "auto") -> ToolModelResponse:
        messages = copy.deepcopy(messages)
        if image_snapshot is not None and (image_snapshot.images or image_snapshot.attachments):
            for message in messages:
                if message["role"] == "user" and isinstance(message.get("content"), str):
                    blocks = [{"type": "text", "text": message["content"]}]
                    pictures = image_snapshot.images[:image_snapshot.max_images] if image_snapshot.enabled else []
                    for index, picture in enumerate(pictures, 1):
                        blocks.extend([{ "type": "text", "text": f"图{index}"},
                            {"type": "image_url", "image_url": {"url":
                             f"data:{picture.media_type};base64,{base64.b64encode(picture.data).decode()}", "detail": "auto"}}])
                    message["content"] = blocks
                    break
        safe_messages = copy.deepcopy(messages)
        for message in safe_messages:
            if isinstance(message.get("content"), list):
                message["content"] = [({"type": "image_ref"} if part.get("type") == "image_url" else part)
                                      for part in message["content"]]
        context = trace_context or {}
        trace_id = None
        if self.trace_store is not None:
            trace_id = self.trace_store.record_start(component="chat_agent", operation="agent_step",
                model=self.model, thinking=self.thinking, temperature=self.temperature,
                response_format=None, messages=safe_messages, round_id=context.get("round_id"),
                attempt=context.get("attempt"), current_user_message=context.get("current_user_message"),
                image_context=({"enabled": image_snapshot.enabled, "max_images": image_snapshot.max_images,
                    "sent_count": min(len(image_snapshot.images), image_snapshot.max_images) if image_snapshot.enabled else 0,
                    "attachments": image_snapshot.attachments} if image_snapshot is not None else None))
        params = dict(model=self.model, messages=messages, stream=False, temperature=self.temperature,
                      extra_body={"thinking": {"type": self.thinking}})
        if tools:
            params.update(tools=tools, tool_choice=tool_choice)
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        start = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**params)
            choice = response.choices[0]
            raw_response = _raw_response(response)
            message = choice.message
            calls = (message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)) or []
            calls = [call.model_dump(exclude_none=True) if hasattr(call, "model_dump") else call for call in calls]
            result = ToolModelResponse(_message_value(message, "content") or "", calls,
                _message_value(message, "reasoning_content"), getattr(choice, "finish_reason", None) or
                raw_response.get("choices", [{}])[0].get("finish_reason"), trace_id)
            if trace_id:
                self.trace_store.record_success(trace_id=trace_id, response_text=result.content,
                    raw_response=raw_response,
                    reasoning_content=result.reasoning_content, usage=_response_usage(response),
                    duration_ms=(time.perf_counter()-start)*1000)
            return result
        except BaseException as exc:
            if trace_id:
                self.trace_store.record_error(trace_id=trace_id, error=exc,
                    duration_ms=(time.perf_counter()-start)*1000)
            raise

    async def ainvoke(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        trace_context: dict[str, Any] | None = None,
        image_snapshot: ImageSnapshot | None = None,
    ) -> str:
        resolved_response_format = (
            response_format if response_format is not None else self.response_format
        )
        resolved_system_prompt = system_prompt or DEEPSEEK_SYSTEM_PROMPT_TEMPLATE.format(
            bot_identity_prompt=BOT_IDENTITY_PROMPT,
        )
        messages = [
            {
                "role": "system",
                "content": resolved_system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        trace_messages = messages
        if image_snapshot is not None and image_snapshot.attachments:
            text = prompt
            blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
            trace_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
            if image_snapshot.enabled:
                for index, image in enumerate(image_snapshot.images[:image_snapshot.max_images], 1):
                    label = {"type": "text", "text": f"图{index}"}
                    blocks.append(label)
                    trace_blocks.append(label.copy())
                    blocks.append({"type": "image_url", "image_url": {
                        "url": f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}",
                        "detail": "auto",
                    }})
                    trace_blocks.append({"type": "image_ref", "input_index": index, "resource_id": image.resource_id,
                                         "media_type": image.media_type})
            messages = [messages[0], {"role": "user", "content": blocks}]
            trace_messages = [messages[0], {"role": "user", "content": trace_blocks}]
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
                messages=trace_messages,
                **({"round_id": context["round_id"]} if context.get("round_id") else {}),
                **({"attempt": context["attempt"]} if context.get("attempt") else {}),
                **({"image_context": {
                    "enabled": image_snapshot.enabled, "max_images": image_snapshot.max_images,
                    "sent_count": min(len(image_snapshot.images), image_snapshot.max_images) if image_snapshot.enabled else 0,
                    "attachments": image_snapshot.attachments,
                }} if image_snapshot is not None else {}),
                current_user_message=_clean_trace_value(
                    context.get("current_user_message")
                ),
            )
            self.last_trace_id = trace_id
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

        message = response.choices[0].message
        content = _message_value(message, "content") or ""
        reasoning_content = _message_value(message, "reasoning_content")
        if self.trace_store is not None and trace_id is not None:
            self.trace_store.record_success(
                trace_id=trace_id,
                response_text=content,
                raw_response=_raw_response(response),
                reasoning_content=reasoning_content,
                usage=_response_usage(response),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        # Restore this invocation's ID after awaiting the response. Other
        # concurrent calls may have started while it was in flight.
        self.last_trace_id = trace_id
        return content


def create_deepseek_chat_llm(
    *,
    api_key: str | None = None,
    model: str | None = None,
    thinking: ThinkingMode | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    trace_store: Any | None = None,
) -> DeepSeekChatLLM | None:
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY") or _read_dotenv_key()
    if not resolved_key:
        return None
    return DeepSeekChatLLM(
        api_key=resolved_key,
        model=model or _read_model(),
        thinking=thinking or _read_thinking_mode(),
        max_tokens=max_tokens,
        response_format=response_format,
        trace_store=trace_store,
    )


def _read_model() -> str:
    return (
        os.getenv("DEEPSEEK_MODEL")
        or _read_dotenv_key("DEEPSEEK_MODEL")
        or "deepseek-flash"
    )


def _read_thinking_mode(env_name: str = "DEEPSEEK_THINKING") -> ThinkingMode:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return "enabled"
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return "disabled"
    raise ValueError(f"{env_name} must be true or false.")


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


def _raw_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    # Also support lightweight clients used in tests and custom assembly.
    return {
        "choices": [
            {
                "finish_reason": getattr(choice, "finish_reason", None),
                "message": {
                    "content": _message_value(choice.message, "content"),
                    "reasoning_content": _message_value(choice.message, "reasoning_content"),
                },
            }
            for choice in response.choices
        ],
        "usage": _response_usage(response),
    }


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


def _message_value(message: Any, key: str) -> str | None:
    if isinstance(message, dict):
        value = message.get(key)
    else:
        value = getattr(message, key, None)
    if value is None:
        return None
    return str(value)


def _clean_trace_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
