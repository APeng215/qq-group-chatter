from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any

from qq_group_chatter.agent.tools import ToolRegistry, setting
from qq_group_chatter.observability import LLM_LATENCY_SECONDS, observe_duration


@dataclass(frozen=True)
class AgentLimits:
    model_calls: int = 8
    tool_calls: int = 12
    tool_timeout: float = 20
    round_timeout: float = 120

    def __post_init__(self):
        if min(self.model_calls, self.tool_calls, self.tool_timeout, self.round_timeout) <= 0:
            raise ValueError("Agent limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(int(setting("AGENT_MAX_MODEL_CALLS", "8")),
                   int(setting("AGENT_MAX_TOOL_CALLS", "12")),
                   float(setting("AGENT_TOOL_TIMEOUT_SECONDS", "20")),
                   float(setting("AGENT_ROUND_TIMEOUT_SECONDS", "120")))


async def run_tools(llm, registry: ToolRegistry, limits: AgentLimits, *, prompt: str,
                    system_prompt: str, context, image_snapshot=None, on_search_start=None) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    cache: dict[str, str] = {}
    executions = 0
    noticed = False
    last_trace = None
    termination = "model_call_limit"
    store = getattr(llm, "trace_store", None)
    def record(action, **kwargs):
        if store is not None and last_trace:
            store.record_result(trace_id=last_trace, parsed_action=action, **kwargs)
    try:
        async with asyncio.timeout(limits.round_timeout):
            for step in range(limits.model_calls):
                final_step = step == limits.model_calls - 1 or executions >= limits.tool_calls
                if final_step:
                    messages.append({"role": "user", "content":
                        "本轮执行预算即将用尽。停止调用工具，依据已有结果回答，并明确尚未完成的部分。"})
                with observe_duration(metric=LLM_LATENCY_SECONDS, labels={"component": "chat_agent"},
                                      log_name="llm_call", log_fields={"component": "chat_agent"}):
                    result = await llm.ainvoke_messages(messages, tools=registry.definitions(),
                        tool_choice="none" if final_step else "auto", image_snapshot=image_snapshot,
                        trace_context={"round_id": context.round_id, "attempt": step + 1, "current_user_message": prompt})
                last_trace = result.trace_id
                if not result.tool_calls:
                    if result.content.strip() and result.finish_reason != "length":
                        record("reply", final_reply=result.content.strip())
                        return result.content.strip()
                    record("retry", fallback_reason="empty_or_truncated_response")
                    continue
                messages.append(result.assistant_message())
                records = []
                for call in result.tool_calls:
                    started = time.perf_counter()
                    function = call["function"]
                    name = function["name"]
                    cached = False
                    try:
                        arguments = json.loads(function["arguments"])
                        if not isinstance(arguments, dict):
                            raise ValueError("Arguments must be an object")
                        key = name + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
                        if key in cache:
                            output = cache[key]
                            cached = True
                        elif final_step or executions >= limits.tool_calls:
                            output = json.dumps({"error": "tool_call_limit", "message": "Stop using tools and answer with available evidence"})
                        else:
                            executions += 1
                            if name in {"web_search", "web_fetch"} and not noticed:
                                noticed = True
                                if on_search_start:
                                    try:
                                        notice = on_search_start("我查一下，稍等。")
                                        if inspect.isawaitable(notice):
                                            await notice
                                    except Exception:
                                        pass
                            async with asyncio.timeout(limits.tool_timeout):
                                value = await registry.call(name, arguments)
                            output = json.dumps(value, ensure_ascii=False, default=str)
                            cache[key] = output
                    except (ValueError, TypeError):
                        output = json.dumps({"error": "invalid_arguments", "message": "Provide a JSON object matching the tool schema"})
                    except TimeoutError:
                        output = json.dumps({"error": "tool_timeout"})
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                    records.append({"tool_call_id": call["id"], "name": name,
                        "arguments": function["arguments"], "result": output, "cached": cached,
                        "duration_ms": round((time.perf_counter()-started)*1000, 3)})
                record("tools", tool_events=records)
    except TimeoutError:
        termination = "round_timeout"
    answer = "这次处理达到时间或调用次数限制，还没能完成可靠的整理。你可以缩小问题范围再让我试试。"
    record("fallback", fallback_reason=termination, final_reply=answer)
    return answer
