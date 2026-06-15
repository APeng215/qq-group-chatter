from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from yarl import URL


def setup_memory_dashboard(driver: Any, application: Any) -> None:
    from nonebot.drivers import HTTPServerSetup

    driver.setup_http_server(
        HTTPServerSetup(
            path=URL("/memory"),
            method="GET",
            name="qq_group_chatter_memory_dashboard",
            handle_func=lambda request: memory_dashboard_response(application),
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


async def memory_dashboard_response(application: Any) -> Any:
    from nonebot.drivers import Response

    snapshot = await asyncio.to_thread(build_memory_dashboard_snapshot, application)
    return Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=memory_dashboard_html(snapshot),
    )


async def memory_dashboard_api(application: Any) -> dict[str, Any]:
    from nonebot.drivers import Response

    snapshot = await asyncio.to_thread(build_memory_dashboard_snapshot, application)
    return Response(
        200,
        headers={"content-type": "application/json; charset=utf-8"},
        content=json.dumps(snapshot, ensure_ascii=False),
    )


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
    }
    return {
        "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "summary": summary,
        "memories": memories,
        "errors": errors,
    }


def memory_dashboard_html(snapshot: dict[str, Any]) -> str:
    snapshot_json = json.dumps(snapshot, ensure_ascii=False)
    escaped_snapshot_json = (
        snapshot_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>长期记忆</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --text: #1f2328;
      --muted: #667085;
      --border: #d7dce2;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --danger: #b42318;
      --code: #344054;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    main {{ padding: 20px 24px 32px; max-width: 1280px; margin: 0 auto; }}
    .toolbar, .summary, .memory, .history-item {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 160px 160px auto;
      gap: 10px;
      padding: 12px;
      margin-bottom: 16px;
      align-items: center;
    }}
    input, select, button {{
      height: 36px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }}
    button {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      cursor: pointer;
      font-weight: 600;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      overflow: hidden;
      margin-bottom: 16px;
    }}
    .summary div {{ padding: 14px 16px; background: #fff; }}
    .summary strong {{ display: block; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .error {{
      color: var(--danger);
      background: #fff4f2;
      border: 1px solid #fecdca;
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 12px;
    }}
    .memory {{
      padding: 14px 16px;
      margin-bottom: 12px;
    }}
    .memory-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 600;
    }}
    .id {{
      font-family: Consolas, "Cascadia Mono", monospace;
      color: var(--code);
      word-break: break-all;
      font-size: 12px;
    }}
    .content {{
      font-size: 16px;
      margin: 10px 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    details {{
      border-top: 1px solid var(--border);
      margin-top: 12px;
      padding-top: 10px;
    }}
    summary {{ cursor: pointer; font-weight: 700; }}
    pre {{
      overflow: auto;
      background: #f3f4f6;
      border-radius: 6px;
      padding: 10px;
      max-height: 260px;
      font-size: 12px;
    }}
    .history-item {{
      padding: 10px;
      margin-top: 8px;
      background: #fcfcfd;
    }}
    .empty {{
      text-align: center;
      padding: 48px 16px;
      color: var(--muted);
    }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .toolbar, .summary, .grid {{ grid-template-columns: 1fr; }}
      .memory-head {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>长期记忆</h1>
    <div class="muted">只读视图，数据来自当前 Mem0 本地存储。</div>
  </header>
  <main>
    <section class="toolbar">
      <input id="search" type="search" placeholder="搜索内容、ID、metadata">
      <select id="scope">
        <option value="">全部 scope</option>
        <option value="user">用户记忆</option>
        <option value="conversation">会话记忆</option>
        <option value="other">其他</option>
      </select>
      <select id="kind">
        <option value="">全部 kind</option>
      </select>
      <button id="refresh" type="button">刷新</button>
    </section>
    <section id="summary" class="summary"></section>
    <section id="errors"></section>
    <section id="list"></section>
  </main>
  <script>
    window.__MEMORY_SNAPSHOT__ = {escaped_snapshot_json};
    let snapshot = window.__MEMORY_SNAPSHOT__;

    const summaryEl = document.querySelector("#summary");
    const errorsEl = document.querySelector("#errors");
    const listEl = document.querySelector("#list");
    const searchEl = document.querySelector("#search");
    const scopeEl = document.querySelector("#scope");
    const kindEl = document.querySelector("#kind");
    const refreshEl = document.querySelector("#refresh");

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[char]));
    }}

    function renderSummary() {{
      const s = snapshot.summary || {{}};
      summaryEl.innerHTML = [
        ["总数", s.total || 0],
        ["用户", s.user || 0],
        ["会话", s.conversation || 0],
        ["其他", s.other || 0],
      ].map(([label, value]) => `<div><span class="muted">${{label}}</span><strong>${{value}}</strong></div>`).join("");
    }}

    function renderErrors() {{
      errorsEl.innerHTML = (snapshot.errors || []).map(error =>
        `<div class="error">${{escapeHtml(error)}}</div>`
      ).join("");
    }}

    function syncKinds() {{
      const current = kindEl.value;
      const kinds = [...new Set((snapshot.memories || []).map(item => item.kind).filter(Boolean))].sort();
      kindEl.innerHTML = '<option value="">全部 kind</option>' +
        kinds.map(kind => `<option value="${{escapeHtml(kind)}}">${{escapeHtml(kind)}}</option>`).join("");
      kindEl.value = kinds.includes(current) ? current : "";
    }}

    function filteredMemories() {{
      const q = searchEl.value.trim().toLowerCase();
      const scope = scopeEl.value;
      const kind = kindEl.value;
      return (snapshot.memories || []).filter(item => {{
        if (scope && item.scope !== scope) return false;
        if (kind && item.kind !== kind) return false;
        if (!q) return true;
        return JSON.stringify(item).toLowerCase().includes(q);
      }});
    }}

    function renderList() {{
      const memories = filteredMemories();
      if (!memories.length) {{
        listEl.innerHTML = '<div class="empty">没有匹配的长期记忆</div>';
        return;
      }}
      listEl.innerHTML = memories.map(item => {{
        const history = item.history || [];
        const metadata = JSON.stringify(item.metadata || {{}}, null, 2);
        return `<article class="memory">
          <div class="memory-head">
            <div class="badges">
              <span class="badge">${{escapeHtml(item.scope)}}</span>
              <span class="badge">${{escapeHtml(item.kind || "unknown")}}</span>
              <span class="id">${{escapeHtml(item.id)}}</span>
            </div>
            <div class="muted">${{escapeHtml(item.updated_at || item.created_at || "")}}</div>
          </div>
          <div class="content">${{escapeHtml(item.content)}}</div>
          <div class="grid">
            <div><span class="muted">owner</span><div class="id">${{escapeHtml(item.owner_id)}}</div></div>
            <div><span class="muted">created</span><div>${{escapeHtml(item.created_at || item.source_created_at || "")}}</div></div>
          </div>
          <details>
            <summary>metadata</summary>
            <pre>${{escapeHtml(metadata)}}</pre>
          </details>
          <details>
            <summary>变更历史 ${{history.length}}</summary>
            ${{history.length ? history.map(entry => `
              <div class="history-item">
                <div><strong>${{escapeHtml(entry.event || entry.action || "")}}</strong> <span class="muted">${{escapeHtml(entry.created_at || entry.updated_at || "")}}</span></div>
                <pre>${{escapeHtml(JSON.stringify(entry, null, 2))}}</pre>
              </div>
            `).join("") : '<div class="muted">没有历史记录</div>'}}
          </details>
        </article>`;
      }}).join("");
    }}

    async function refresh() {{
      refreshEl.disabled = true;
      try {{
        const response = await fetch("/api/memory", {{ cache: "no-store" }});
        snapshot = await response.json();
        syncKinds();
        render();
      }} finally {{
        refreshEl.disabled = false;
      }}
    }}

    function render() {{
      renderSummary();
      renderErrors();
      renderList();
    }}

    searchEl.addEventListener("input", renderList);
    scopeEl.addEventListener("change", renderList);
    kindEl.addEventListener("change", renderList);
    refreshEl.addEventListener("click", refresh);
    syncKinds();
    render();
  </script>
</body>
</html>
"""


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
