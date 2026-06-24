from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any

from yarl import URL

from qq_group_chatter.agent.deepseek_llm import _read_dotenv_key


def setup_memory_dashboard(driver: Any, application: Any) -> None:
    if not _read_bool("QQ_GROUP_CHATTER_MEMORY_DASHBOARD_ENABLED", True):
        return

    from nonebot.drivers import HTTPServerSetup

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


def _read_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name) or _read_dotenv_key(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
        return trace_store.snapshot()
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
  <title>运行控制台</title>
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
      --success: #027a48;
      --warning: #b54708;
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
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 12px 24px 0;
      max-width: 1280px;
      margin: 0 auto;
    }}
    .tab {{
      background: #fff;
      color: var(--text);
      border: 1px solid var(--border);
    }}
    .tab.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .panel[hidden] {{ display: none; }}
    .toolbar, .summary, .memory, .history-item, .trace {{
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
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
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
    .status-success {{
      background: #ecfdf3;
      color: var(--success);
    }}
    .status-error {{
      background: #fff4f2;
      color: var(--danger);
    }}
    .status-running {{
      background: #fffaeb;
      color: var(--warning);
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
    .trace {{
      padding: 14px 16px;
      margin-bottom: 12px;
    }}
    .trace-headline {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .trace-actions {{
      display: inline-flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .trace-action {{
      height: 28px;
      padding: 0 8px;
      border-radius: 6px;
      background: #f8fafc;
      border-color: var(--border);
      color: var(--code);
      font-size: 12px;
      font-weight: 600;
    }}
    .trace-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .duration-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #f2f4f7;
      color: var(--code);
      font-size: 12px;
      font-weight: 700;
    }}
    .duration-ok {{
      background: #ecfdf3;
      color: var(--success);
    }}
    .duration-warn {{
      background: #fffaeb;
      color: var(--warning);
    }}
    .duration-slow {{
      background: #fff4f2;
      color: var(--danger);
    }}
    .trace-filter-status {{
      color: var(--muted);
      margin: -6px 0 12px;
      min-height: 20px;
    }}
    .trace-component-chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: -4px 0 14px;
    }}
    .chip {{
      height: 30px;
      padding: 0 10px;
      border-radius: 999px;
      background: #fff;
      border: 1px solid var(--border);
      color: var(--code);
      font-size: 12px;
      font-weight: 700;
    }}
    .chip.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .trace pre {{
      max-height: 360px;
    }}
    .trace-group {{
      margin-bottom: 18px;
    }}
    .trace-group-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      padding: 10px 12px;
      margin-bottom: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #f8fafc;
    }}
    .trace-group-title {{
      margin: 0;
      font-weight: 700;
      word-break: break-word;
    }}
    .trace-group-count {{
      flex: 0 0 auto;
    }}
    .trace-message {{
      margin-top: 12px;
    }}
    .trace-message-role {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 1px 7px;
      border-radius: 6px;
      background: var(--accent-soft);
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
    }}
    .trace-message-body {{
      margin-top: 6px;
      padding-left: 12px;
      border-left: 3px solid #bfdbfe;
    }}
    .trace-message-content, .trace-json-block {{
      margin-top: 6px;
      background: #f8fafc;
      border: 1px solid #e4e7ec;
      border-radius: 6px;
      color: var(--code);
      line-height: 1.55;
    }}
    .trace-reasoning-content {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #7c2d12;
      line-height: 1.6;
    }}
    .trace-tools {{
      grid-template-columns: minmax(220px, 1fr) 160px 160px auto auto;
    }}
    .danger-button {{
      background: var(--danger);
      border-color: var(--danger);
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
      .trace-headline {{ flex-direction: column; }}
      .trace-actions {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>运行控制台</h1>
    <div class="muted">只读视图，包含长期记忆、记忆队列和本地 LLM trace。</div>
  </header>
  <nav class="tabs" aria-label="dashboard sections">
    <button id="memory-tab" class="tab active" type="button">记忆库</button>
    <button id="llm-tab" class="tab" type="button">LLM 控制台</button>
  </nav>
  <main>
    <section id="memory-panel" class="panel">
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
    </section>
    <section id="llm-panel" class="panel" hidden>
      <section class="toolbar trace-tools">
        <input id="trace-search" type="search" placeholder="搜索 prompt / response / error">
        <select id="trace-component">
          <option value="">全部 component</option>
        </select>
        <select id="trace-status">
          <option value="">全部 status</option>
          <option value="running">running</option>
          <option value="success">success</option>
          <option value="error">error</option>
        </select>
        <button id="trace-refresh" type="button">刷新</button>
        <button id="trace-export" type="button">导出当前筛选</button>
        <button id="trace-clear" class="danger-button" type="button">清空</button>
      </section>
      <section id="trace-summary" class="summary"></section>
      <section id="trace-errors"></section>
      <section id="trace-filter-status" class="trace-filter-status"></section>
      <section id="trace-component-chips" class="trace-component-chips"></section>
      <section id="trace-diagnostic-chips" class="trace-component-chips"></section>
      <section id="trace-list"></section>
    </section>
  </main>
  <script>
    window.__MEMORY_SNAPSHOT__ = {escaped_snapshot_json};
    let snapshot = window.__MEMORY_SNAPSHOT__;
    let traceSnapshot = {{ summary: {{}}, traces: [], errors: [] }};

    const memoryTabEl = document.querySelector("#memory-tab");
    const llmTabEl = document.querySelector("#llm-tab");
    const memoryPanelEl = document.querySelector("#memory-panel");
    const llmPanelEl = document.querySelector("#llm-panel");
    const summaryEl = document.querySelector("#summary");
    const errorsEl = document.querySelector("#errors");
    const listEl = document.querySelector("#list");
    const searchEl = document.querySelector("#search");
    const scopeEl = document.querySelector("#scope");
    const kindEl = document.querySelector("#kind");
    const refreshEl = document.querySelector("#refresh");
    const traceSummaryEl = document.querySelector("#trace-summary");
    const traceErrorsEl = document.querySelector("#trace-errors");
    const traceFilterStatusEl = document.querySelector("#trace-filter-status");
    const traceComponentChipsEl = document.querySelector("#trace-component-chips");
    const traceListEl = document.querySelector("#trace-list");
    const traceSearchEl = document.querySelector("#trace-search");
    const traceComponentEl = document.querySelector("#trace-component");
    const traceStatusEl = document.querySelector("#trace-status");
    const traceRefreshEl = document.querySelector("#trace-refresh");
    const traceExportEl = document.querySelector("#trace-export");
    const traceClearEl = document.querySelector("#trace-clear");
    const traceDiagnosticChipsEl = document.querySelector("#trace-diagnostic-chips");
    let activeTraceDiagnostic = "";

    const TRACE_DIAGNOSTICS = [
      {{ id: "fallback", label: "失败回复", match: item => Boolean(item?.fallback_reason || item?.parsed_action === "fallback") }},
      {{ id: "web_search", label: "触发搜索", match: item => Boolean(item?.search_query || item?.parsed_action === "web_search" || item?.operation === "grounded_search_reply") }},
      {{ id: "thinking", label: "有 thinking", match: item => traceHasReasoningContent(item) }},
      {{ id: "slow", label: "慢调用 > 8s", match: item => typeof item?.duration_ms === "number" && item.duration_ms >= 8000 }},
      {{ id: "memory_planner", label: "记忆规划", match: item => item?.component === "memory_planner" }},
    ];

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[char]));
    }}

    function formatTraceText(value) {{
      return escapeHtml(String(value ?? "").replace(/\\\\n/g, "\\n"));
    }}

    function formatDurationMs(value) {{
      if (typeof value !== "number" || !Number.isFinite(value)) return "未完成";
      if (value >= 1000) return `${{(value / 1000).toFixed(value >= 10000 ? 1 : 2)}} s`;
      return `${{Math.round(value)}} ms`;
    }}

    function durationSeverityClass(value) {{
      if (typeof value !== "number" || !Number.isFinite(value)) return "";
      if (value >= 8000) return "duration-slow";
      if (value >= 3000) return "duration-warn";
      return "duration-ok";
    }}

    function traceStatusClass(status) {{
      if (status === "success") return "status-success";
      if (status === "error") return "status-error";
      if (status === "running") return "status-running";
      return "";
    }}

    function traceCopyPayload(item, kind) {{
      if (kind === "id") return String(item?.trace_id || "");
      if (kind === "response") return String(item?.response_text || "");
      return JSON.stringify(item || {{}}, null, 2);
    }}

    async function copyTraceText(button, text) {{
      const original = button.textContent;
      try {{
        await navigator.clipboard.writeText(text);
        button.textContent = "已复制";
      }} catch (error) {{
        button.textContent = "复制失败";
      }} finally {{
        window.setTimeout(() => {{ button.textContent = original; }}, 1200);
      }}
    }}

    function renderTraceMessages(messages) {{
      if (!Array.isArray(messages) || !messages.length) {{
        return '<div class="empty">没有 messages</div>';
      }}
      return messages.map((message, index) => {{
        const role = message && typeof message === "object" ? message.role : `message ${{index + 1}}`;
        const content = message && typeof message === "object" && "content" in message
          ? message.content
          : message;
        return `<div class="trace-message">
          <div class="trace-message-role">${{escapeHtml(role || `message ${{index + 1}}`)}}</div>
          <div class="trace-message-body">
            <pre class="trace-message-content">${{formatTraceText(content ?? "")}}</pre>
          </div>
        </div>`;
      }}).join("");
    }}

    function renderTraceUserQuestion(item) {{
      const question = String(item?.current_user_message || "").trim();
      if (!question) {{
        return '<div class="empty">没有用户提问</div>';
      }}
      return `<pre class="trace-message-content">${{formatTraceText(question)}}</pre>`;
    }}

    function renderTraceAssistantReply(item) {{
      const visibleReply = item.final_reply || item.search_notice || item.response_text;
      const blocks = [];
      if (String(visibleReply || "").trim()) {{
        blocks.push(`<pre class="trace-message-content">${{formatTraceText(visibleReply)}}</pre>`);
      }} else {{
        blocks.push('<div class="empty">没有神奈回复</div>');
      }}
      blocks.push(`<div class="muted">原始 response</div><pre class="trace-json-block">${{formatTraceText(item.response_text || "")}}</pre>`);
      return blocks.join("");
    }}

    function traceHasReasoningContent(item) {{
      return Boolean(String(item?.reasoning_content || "").trim());
    }}

    function renderTraceReasoning(item) {{
      if (!traceHasReasoningContent(item)) {{
        return '<div class="empty">没有 thinking 内容</div>';
      }}
      return `<pre class="trace-reasoning-content">${{formatTraceText(item.reasoning_content || "")}}</pre>`;
    }}

    function renderTraceResult(item) {{
      const result = {{
        parsed_action: item.parsed_action || null,
        final_reply: item.final_reply || null,
        fallback_reason: item.fallback_reason || null,
        search_notice: item.search_notice || null,
        search_query: item.search_query || null,
      }};
      if (!Object.values(result).some(value => value !== null && String(value).trim())) {{
        return '<div class="empty">没有解析后的最终输出</div>';
      }}
      return `<pre class="trace-json-block">${{escapeHtml(JSON.stringify(result, null, 2))}}</pre>`;
    }}

    function renderSummary() {{
      const s = snapshot.summary || {{}};
      summaryEl.innerHTML = [
        ["总数", s.total || 0],
        ["用户", s.user || 0],
        ["会话", s.conversation || 0],
        ["队列", s.queue_size || 0],
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

    function renderTraceSummary(traces = traceSnapshot.traces || []) {{
      const s = traceSnapshot.summary || {{}};
      const visibleCompleted = traces.filter(item => typeof item.duration_ms === "number");
      const visibleAverage = visibleCompleted.length
        ? Math.round(visibleCompleted.reduce((sum, item) => sum + item.duration_ms, 0) / visibleCompleted.length)
        : 0;
      traceSummaryEl.innerHTML = [
        ["总数", s.total || 0],
        ["运行中", s.running || 0],
        ["成功", s.success || 0],
        ["错误", s.error || 0],
      ].map(([label, value]) => `<div><span class="muted">${{label}}</span><strong>${{value}}</strong></div>`).join("") +
        `<div><span class="muted">当前匹配</span><strong>${{traces.length}}</strong></div>` +
        `<div><span class="muted">thinking 内容</span><strong>${{traces.filter(traceHasReasoningContent).length}}</strong></div>` +
        `<div><span class="muted">匹配平均耗时</span><strong>${{escapeHtml(formatDurationMs(visibleAverage))}}</strong></div>` +
        `<div><span class="muted">聊天平均耗时</span><strong>${{escapeHtml(formatDurationMs(s.average_duration_ms || 0))}}</strong></div>`;
    }}

    function renderTraceErrors() {{
      traceErrorsEl.innerHTML = (traceSnapshot.errors || []).map(error =>
        `<div class="error">${{escapeHtml(error)}}</div>`
      ).join("");
    }}

    function syncTraceComponents() {{
      const current = traceComponentEl.value;
      const components = [...new Set((traceSnapshot.traces || []).map(item => item.component).filter(Boolean))].sort();
      traceComponentEl.innerHTML = '<option value="">全部 component</option>' +
        components.map(component => `<option value="${{escapeHtml(component)}}">${{escapeHtml(component)}}</option>`).join("");
      traceComponentEl.value = components.includes(current) ? current : "";
      renderTraceComponentChips(components);
    }}

    function renderTraceComponentChips(components = []) {{
      const current = traceComponentEl.value;
      traceComponentChipsEl.innerHTML = [
        `<button class="chip ${{current ? "" : "active"}}" type="button" data-component-filter="">全部</button>`,
        ...components.map(component =>
          `<button class="chip ${{component === current ? "active" : ""}}" type="button" data-component-filter="${{escapeHtml(component)}}">${{escapeHtml(component)}}</button>`
        ),
      ].join("");
    }}

    function renderTraceDiagnosticChips() {{
      traceDiagnosticChipsEl.innerHTML = [
        `<button class="chip ${{activeTraceDiagnostic ? "" : "active"}}" type="button" data-diagnostic-filter="">全部诊断</button>`,
        ...TRACE_DIAGNOSTICS.map(diagnostic =>
          `<button class="chip ${{diagnostic.id === activeTraceDiagnostic ? "active" : ""}}" type="button" data-diagnostic-filter="${{escapeHtml(diagnostic.id)}}">${{escapeHtml(diagnostic.label)}}</button>`
        ),
      ].join("");
    }}

    function traceMatchesDiagnostic(item) {{
      if (!activeTraceDiagnostic) return true;
      const diagnostic = TRACE_DIAGNOSTICS.find(candidate => candidate.id === activeTraceDiagnostic);
      return diagnostic ? diagnostic.match(item) : true;
    }}

    function filteredTraces() {{
      const q = traceSearchEl.value.trim().toLowerCase();
      const component = traceComponentEl.value;
      const status = traceStatusEl.value;
      return (traceSnapshot.traces || []).filter(item => {{
        if (component && item.component !== component) return false;
        if (status && item.status !== status) return false;
        if (!traceMatchesDiagnostic(item)) return false;
        if (!q) return true;
        return JSON.stringify(item).toLowerCase().includes(q);
      }});
    }}

    function renderTraceFilterStatus(traces) {{
      const total = (traceSnapshot.traces || []).length;
      const filters = [];
      if (traceComponentEl.value) filters.push(`component=${{traceComponentEl.value}}`);
      if (traceStatusEl.value) filters.push(`status=${{traceStatusEl.value}}`);
      if (activeTraceDiagnostic) {{
        const diagnostic = TRACE_DIAGNOSTICS.find(candidate => candidate.id === activeTraceDiagnostic);
        filters.push(`诊断=${{diagnostic ? diagnostic.label : activeTraceDiagnostic}}`);
      }}
      if (traceSearchEl.value.trim()) filters.push("包含搜索词");
      const filterText = filters.length ? `筛选：${{filters.join("，")}}` : "未筛选";
      traceFilterStatusEl.textContent = `${{filterText}} · 显示 ${{traces.length}} / ${{total}} 条`;
    }}

    function captureTraceDetailState() {{
      const state = new Map();
      traceListEl.querySelectorAll("details[data-detail-key]").forEach(detail => {{
        state.set(detail.dataset.detailKey, detail.open);
      }});
      return state;
    }}

    function restoreTraceDetailState(state) {{
      traceListEl.querySelectorAll("details[data-detail-key]").forEach(detail => {{
        const key = detail.dataset.detailKey;
        if (state.has(key)) detail.open = state.get(key);
      }});
    }}

    function traceGroupKey(item) {{
      const question = String(item?.current_user_message || "").trim();
      if (question) return question;
      return `trace:${{item?.trace_id || item?.created_at || Math.random()}}`;
    }}

    function groupTracesByUserQuestion(traces) {{
      const groups = [];
      traces.forEach(item => {{
        const key = traceGroupKey(item);
        const previous = groups[groups.length - 1];
        if (previous && previous.key === key) {{
          previous.traces.push(item);
          return;
        }}
        groups.push({{
          key,
          title: String(item?.current_user_message || "").trim() || "没有用户提问",
          traces: [item],
        }});
      }});
      return groups;
    }}

    function renderTraceCard(item) {{
      const traceKey = String(item.trace_id || item.created_at || "");
      const usage = JSON.stringify(item.usage || {{}}, null, 2);
      return `<article class="trace">
        <div class="trace-headline">
          <div class="badges">
            <span class="badge ${{traceStatusClass(item.status)}}">${{escapeHtml(item.status || "running")}}</span>
            <span class="badge">${{escapeHtml(item.component || "unknown")}}</span>
            <span class="badge">${{escapeHtml(item.operation || "unknown")}}</span>
            <span class="id">${{escapeHtml(item.trace_id)}}</span>
          </div>
          <div class="trace-actions">
            <button class="trace-action" type="button" data-copy-trace-id="${{escapeHtml(traceKey)}}" data-copy-kind="id">复制 ID</button>
            <button class="trace-action" type="button" data-copy-trace-id="${{escapeHtml(traceKey)}}" data-copy-kind="response">复制响应</button>
            <button class="trace-action" type="button" data-copy-trace-id="${{escapeHtml(traceKey)}}" data-copy-kind="json">复制 JSON</button>
          </div>
        </div>
        <div class="trace-meta">
          <div class="muted">${{escapeHtml(item.created_at || "")}}</div>
          <span class="duration-pill ${{durationSeverityClass(item.duration_ms)}}">${{escapeHtml(formatDurationMs(item.duration_ms))}}</span>
        </div>
        <div class="grid">
          <div><span class="muted">model</span><div>${{escapeHtml(item.model || "")}}</div></div>
          <div><span class="muted">thinking</span><div>${{escapeHtml(item.thinking || "")}}</div></div>
        </div>
        ${{item.error_message ? `<div class="error">${{escapeHtml(item.error_type || "Error")}}: ${{escapeHtml(item.error_message)}}</div>` : ""}}
        <details data-detail-key="${{escapeHtml(traceKey)}}:user-question" open>
          <summary>用户提问</summary>
          ${{renderTraceUserQuestion(item)}}
        </details>
        <details data-detail-key="${{escapeHtml(traceKey)}}:assistant-reply" open>
          <summary>神奈回复</summary>
          ${{renderTraceAssistantReply(item)}}
        </details>
        <details data-detail-key="${{escapeHtml(traceKey)}}:reasoning">
          <summary>思考</summary>
          ${{renderTraceReasoning(item)}}
        </details>
        <details data-detail-key="${{escapeHtml(traceKey)}}:messages">
          <summary>上下文</summary>
          ${{renderTraceMessages(item.messages || [])}}
        </details>
        <details data-detail-key="${{escapeHtml(traceKey)}}:result">
          <summary>最终输出 / 解析结果</summary>
          ${{renderTraceResult(item)}}
        </details>
        <details data-detail-key="${{escapeHtml(traceKey)}}:options">
          <summary>usage / options</summary>
          <pre class="trace-json-block">${{escapeHtml(JSON.stringify({{
            response_format: item.response_format || null,
            temperature: item.temperature ?? null,
            usage: item.usage || null,
          }}, null, 2))}}</pre>
        </details>
      </article>`;
    }}

    function renderTraceList() {{
      const detailState = captureTraceDetailState();
      const traces = filteredTraces();
      renderTraceSummary(traces);
      renderTraceFilterStatus(traces);
      if (!traces.length) {{
        traceListEl.innerHTML = '<div class="empty">没有匹配的 LLM trace</div>';
        return;
      }}
      traceListEl.innerHTML = groupTracesByUserQuestion(traces).map(group => `
        <section class="trace-group">
          <div class="trace-group-head">
            <p class="trace-group-title">${{escapeHtml(group.title)}}</p>
            <span class="badge trace-group-count">${{group.traces.length}} 条 trace</span>
          </div>
          ${{group.traces.map(item => renderTraceCard(item)).join("")}}
        </section>
      `).join("");
      restoreTraceDetailState(detailState);
    }}

    function renderTraces() {{
      renderTraceErrors();
      renderTraceList();
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

    async function refreshTraces() {{
      traceRefreshEl.disabled = true;
      try {{
        const response = await fetch("/api/llm-traces", {{ cache: "no-store" }});
        traceSnapshot = await response.json();
        syncTraceComponents();
        renderTraces();
      }} finally {{
        traceRefreshEl.disabled = false;
      }}
    }}

    async function clearTraces() {{
      if (!confirm("确定清空当前所有 LLM trace？建议先导出当前筛选 JSON 留存排障证据。")) return;
      traceClearEl.disabled = true;
      try {{
        await fetch("/api/llm-traces/clear", {{ method: "POST", cache: "no-store" }});
        await refreshTraces();
      }} finally {{
        traceClearEl.disabled = false;
      }}
    }}

    function exportFilteredTraces() {{
      const traces = filteredTraces();
      const payload = JSON.stringify({{
        generated_at: new Date().toISOString(),
        filters: {{
          search: traceSearchEl.value.trim(),
          component: traceComponentEl.value,
          status: traceStatusEl.value,
          diagnostic: activeTraceDiagnostic,
        }},
        traces,
      }}, null, 2);
      const blob = new Blob([payload], {{ type: "application/json;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `llm-traces-${{new Date().toISOString().replace(/[:.]/g, "-")}}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    function onTraceComponentChipClick(event) {{
      const button = event.target.closest("[data-component-filter]");
      if (!button) return;
      traceComponentEl.value = button.dataset.componentFilter || "";
      renderTraceList();
      renderTraceComponentChips([...new Set((traceSnapshot.traces || []).map(item => item.component).filter(Boolean))].sort());
    }}

    function onTraceDiagnosticChipClick(event) {{
      const button = event.target.closest("[data-diagnostic-filter]");
      if (!button) return;
      activeTraceDiagnostic = button.dataset.diagnosticFilter || "";
      renderTraceList();
      renderTraceDiagnosticChips();
    }}

    function onTraceListClick(event) {{
      const button = event.target.closest("[data-copy-trace-id]");
      if (!button) return;
      const traceId = button.dataset.copyTraceId || "";
      const item = (traceSnapshot.traces || []).find(trace =>
        String(trace.trace_id || trace.created_at || "") === traceId
      );
      copyTraceText(button, traceCopyPayload(item, button.dataset.copyKind || "json"));
    }}

    function showPanel(name) {{
      const showMemory = name === "memory";
      memoryPanelEl.hidden = !showMemory;
      llmPanelEl.hidden = showMemory;
      memoryTabEl.classList.toggle("active", showMemory);
      llmTabEl.classList.toggle("active", !showMemory);
      if (!showMemory) refreshTraces();
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
    traceSearchEl.addEventListener("input", renderTraceList);
    traceComponentEl.addEventListener("change", renderTraceList);
    traceStatusEl.addEventListener("change", renderTraceList);
    traceRefreshEl.addEventListener("click", refreshTraces);
    traceExportEl.addEventListener("click", exportFilteredTraces);
    traceClearEl.addEventListener("click", clearTraces);
    traceComponentChipsEl.addEventListener("click", onTraceComponentChipClick);
    traceDiagnosticChipsEl.addEventListener("click", onTraceDiagnosticChipClick);
    traceListEl.addEventListener("click", onTraceListClick);
    memoryTabEl.addEventListener("click", () => showPanel("memory"));
    llmTabEl.addEventListener("click", () => showPanel("llm"));
    syncKinds();
    renderTraceDiagnosticChips();
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
