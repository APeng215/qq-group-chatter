from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from datetime import UTC, datetime
from typing import Any

from yarl import URL

from qq_group_chatter.agent.deepseek_llm import _read_dotenv_key


def setup_memory_dashboard(driver: Any, application: Any) -> None:
    if not _read_bool("QQ_GROUP_CHATTER_MEMORY_DASHBOARD_ENABLED", True):
        return

    from nonebot.drivers import HTTPServerSetup
    from starlette.staticfiles import StaticFiles

    driver.server_app.mount(
        "/console-assets", StaticFiles(directory=str(CONSOLE_DIR), check_dir=False),
        name="qq_group_chatter_console_assets",
    )

    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/api/todo"), method="GET", name="qq_group_chatter_todo_api",
            handle_func=lambda request: todo_api(),
        )
    )
    for path, name in (
        ("/console", "qq_group_chatter_console_dashboard"),
        ("/memory", "qq_group_chatter_memory_dashboard"),
    ):
        driver.setup_http_server(
            HTTPServerSetup(
                path=URL(path),
                method="GET",
                name=name,
                handle_func=lambda request: memory_dashboard_response(application),
            )
        )
    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/api/prompts"), method="GET", name="qq_group_chatter_prompts_api",
            handle_func=lambda request: prompts_api(),
        )
    )
    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/api/memory"),
            method="GET",
            name="qq_group_chatter_memory_api",
            handle_func=lambda request: memory_dashboard_api(application),
        )
    )
    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/api/llm-traces"),
            method="GET",
            name="qq_group_chatter_llm_traces_api",
            handle_func=lambda request: llm_traces_api(application),
        )
    )
    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/api/llm-traces/clear"),
            method="POST",
            name="qq_group_chatter_llm_traces_clear_api",
            handle_func=lambda request: clear_llm_traces_api(application),
        )
    )
    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/api/images"), method="GET", name="qq_group_chatter_image_preview",
            handle_func=lambda request: image_preview_api(application, request.url.query.get("resource_id", "")),
        )
    )


TODO_PATH = Path(__file__).resolve().parents[1] / "todo" / "todo.md"


async def todo_api() -> Any:
    from nonebot.drivers import Response

    try:
        content = await asyncio.to_thread(TODO_PATH.read_text, encoding="utf-8")
        snapshot = {"content": content, "exists": True}
        status = 200
    except FileNotFoundError:
        snapshot = {"content": "", "exists": False}
        status = 200
    except OSError:
        snapshot = {"error": "无法读取 TODO 文件"}
        status = 500
    return Response(status, headers={"content-type": "application/json; charset=utf-8", "cache-control": "no-store"},
                    content=json.dumps(snapshot, ensure_ascii=False))


def build_prompts_snapshot() -> dict[str, Any]:
    from importlib.resources import files
    from qq_group_chatter.prompt_loader import parse_prompt_bundle
    from qq_group_chatter.agent.chat_agent import (
        CHAT_AGENT_SYSTEM_PROMPT, CHAT_AGENT_PROMPT_TEMPLATE,
        CHAT_SEARCH_GROUNDED_SYSTEM_PROMPT, CHAT_SEARCH_GROUNDED_PROMPT_TEMPLATE,
        ERROR_NOTICE_PROMPT_TEMPLATE,
    )
    from qq_group_chatter.services.long_term_memory_planner import PLANNER_SYSTEM_PROMPT, PLANNER_PROMPT_TEMPLATE

    groups = [
        {"key": "chat", "title": "普通聊天", "system": CHAT_AGENT_SYSTEM_PROMPT, "user": CHAT_AGENT_PROMPT_TEMPLATE},
        {"key": "search", "title": "搜索增强（兼容路径）", "system": CHAT_SEARCH_GROUNDED_SYSTEM_PROMPT, "user": CHAT_SEARCH_GROUNDED_PROMPT_TEMPLATE},
        {"key": "planner", "title": "长期记忆 Planner", "system": PLANNER_SYSTEM_PROMPT, "user": PLANNER_PROMPT_TEMPLATE},
        {"key": "notice", "title": "记忆异常通知", "system": CHAT_AGENT_SYSTEM_PROMPT, "user": ERROR_NOTICE_PROMPT_TEMPLATE},
    ]
    errors = []
    try:
        source = parse_prompt_bundle(files("qq_group_chatter").joinpath("prompts", "prompts.md").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        source = {}
        errors.append(f"无法读取提示词文件：{type(exc).__name__}")
    return {"groups": groups, "source": source, "errors": errors}


async def prompts_api() -> Any:
    from nonebot.drivers import Response
    snapshot = await asyncio.to_thread(build_prompts_snapshot)
    return Response(200, headers={"content-type": "application/json; charset=utf-8", "cache-control": "no-store"},
                    content=json.dumps(snapshot, ensure_ascii=False))


def _read_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


async def memory_dashboard_response(application: Any) -> Any:
    from nonebot.drivers import Response

    try:
        html = memory_dashboard_html()
    except FileNotFoundError:
        return Response(503, headers={"content-type": "text/plain; charset=utf-8"},
                        content="控制台尚未构建，请在项目根目录运行 python scripts/build_console.py")
    return Response(200, headers={"content-type": "text/html; charset=utf-8", "cache-control": "no-store"}, content=html)


async def memory_dashboard_api(application: Any) -> dict[str, Any]:
    from nonebot.drivers import Response

    snapshot = await asyncio.to_thread(build_memory_dashboard_snapshot, application)
    return Response(
        200,
        headers={"content-type": "application/json; charset=utf-8"},
        content=json.dumps(snapshot, ensure_ascii=False),
    )


async def llm_traces_api(application: Any) -> Any:
    from nonebot.drivers import Response

    snapshot = await asyncio.to_thread(build_llm_traces_snapshot, application)
    return Response(
        200,
        headers={"content-type": "application/json; charset=utf-8"},
        content=json.dumps(snapshot, ensure_ascii=False),
    )


async def clear_llm_traces_api(application: Any) -> Any:
    from nonebot.drivers import Response

    trace_store = getattr(application, "llm_trace_store", None)
    if trace_store is not None:
        await asyncio.to_thread(trace_store.clear)
    return Response(
        200,
        headers={"content-type": "application/json; charset=utf-8"},
        content=json.dumps({"ok": True}, ensure_ascii=False),
    )


async def image_preview_api(application: Any, resource_id: str) -> Any:
    from nonebot.drivers import Response

    service = getattr(application, "image_service", None)
    resource = await asyncio.to_thread(service.read_resource, resource_id) if service else None
    headers = {"cache-control": "no-store", "x-content-type-options": "nosniff"}
    if resource is None:
        return Response(404, headers=headers, content="图片已不可用")
    data, media_type = resource
    return Response(200, headers={**headers, "content-type": media_type}, content=data)


def build_llm_traces_snapshot(application: Any) -> dict[str, Any]:
    trace_store = getattr(application, "llm_trace_store", None)
    if trace_store is None:
        return {
            "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            "summary": {
                "total": 0,
                "running": 0,
                "success": 0,
                "error": 0,
                "average_duration_ms": 0,
            },
            "traces": [],
            "errors": ["LLM trace store is not available."],
        }
    try:
        snapshot = trace_store.snapshot()
        image_service = getattr(application, "image_service", None)
        snapshot["image_understanding"] = image_service.settings() if image_service else {
            "enabled": False, "max_images_per_request": 4,
        }
        return snapshot
    except Exception as exc:
        return {
            "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            "summary": {
                "total": 0,
                "running": 0,
                "success": 0,
                "error": 0,
                "average_duration_ms": 0,
            },
            "traces": [],
            "errors": [f"Failed to read LLM traces: {type(exc).__name__}: {exc}"],
        }


def build_memory_dashboard_snapshot(application: Any) -> dict[str, Any]:
    mem0 = getattr(getattr(application, "long_term_memory", None), "_mem0", None)
    errors: list[str] = []
    memories = []
    if mem0 is None:
        errors.append("Mem0 client is not available.")
    else:
        try:
            memories = _list_memories(mem0)
        except Exception as exc:
            errors.append(f"Failed to read memories: {type(exc).__name__}: {exc}")

    summary = {
        "total": len(memories),
        "user": sum(1 for item in memories if item["scope"] == "user"),
        "conversation": sum(1 for item in memories if item["scope"] == "conversation"),
        "other": sum(1 for item in memories if item["scope"] == "other"),
        "queue_size": _memory_queue_size(getattr(application, "long_term_memory", None)),
    }
    return {
        "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "summary": summary,
        "memories": memories,
        "errors": errors,
    }


def _memory_queue_size(long_term_memory: Any) -> int:
    queue = getattr(long_term_memory, "_queue", None)
    if queue is None or not hasattr(queue, "qsize"):
        return 0
    try:
        return int(queue.qsize())
    except Exception:
        return 0


CONSOLE_DIR = Path(__file__).parent / "static" / "console"


def memory_dashboard_html(snapshot: dict[str, Any] | None = None) -> str:
    """Return the static SPA shell; data is fetched lazily from the APIs."""
    return (CONSOLE_DIR / "index.html").read_text(encoding="utf-8")


def _list_memories(mem0: Any) -> list[dict[str, Any]]:
    vector_store = getattr(mem0, "vector_store", None)
    if vector_store is None or not hasattr(vector_store, "list"):
        return []

    raw = vector_store.list(filters=None, top_k=1000)
    points = _unwrap_vector_points(raw)
    memories = []
    for point in points:
        payload = dict(getattr(point, "payload", None) or {})
        memory_id = str(getattr(point, "id", ""))
        item = _memory_item_from_payload(memory_id, payload)
        item["history"] = _memory_history(mem0, memory_id)
        memories.append(item)
    return sorted(
        memories,
        key=lambda item: str(item.get("updated_at") or item.get("created_at") or item.get("source_created_at") or ""),
        reverse=True,
    )


def _unwrap_vector_points(raw: Any) -> list[Any]:
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return raw[0]
    if isinstance(raw, list):
        return raw
    return []


def _memory_item_from_payload(memory_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    owner_id = str(payload.get("user_id") or payload.get("agent_id") or payload.get("run_id") or "")
    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "data",
            "hash",
            "id",
            "text_lemmatized",
            "created_at",
            "updated_at",
            "user_id",
            "agent_id",
            "run_id",
        }
    }
    scope = str(metadata.get("scope") or _scope_from_owner(owner_id))
    return {
        "id": memory_id,
        "content": str(payload.get("data") or payload.get("memory") or ""),
        "owner_id": owner_id,
        "scope": scope,
        "kind": str(metadata.get("kind") or ""),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "source_created_at": payload.get("source_created_at"),
        "metadata": metadata,
    }


def _scope_from_owner(owner_id: str) -> str:
    if owner_id.startswith("qq_user:"):
        return "user"
    if owner_id.startswith("qq_conversation:"):
        return "conversation"
    return "other"


def _memory_history(mem0: Any, memory_id: str) -> list[dict[str, Any]]:
    if not memory_id or not hasattr(mem0, "history"):
        return []
    try:
        raw = mem0.history(memory_id)
    except Exception as exc:
        return [{"event": "history_error", "error": f"{type(exc).__name__}: {exc}"}]
    if isinstance(raw, list):
        return [dict(item) if isinstance(item, dict) else {"value": item} for item in raw]
    return [{"value": raw}]
