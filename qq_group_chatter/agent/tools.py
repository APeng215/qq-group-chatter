"""Configured Python and MCP tools. Configuration is trusted administrator input."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import Draft202012Validator
from qq_group_chatter.agent.deepseek_llm import _read_dotenv_key
from qq_group_chatter.observability import record_error


def setting(name: str, default: str) -> str:
    return os.getenv(name) or _read_dotenv_key(name) or default


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[..., Awaitable[Any]]

    def definition(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": self.parameters}}


def env_reference(value: str) -> str:
    match = re.fullmatch(r"${([A-Za-z_][A-Za-z0-9_]*)}", value)
    if not match:
        raise ValueError("Tool credentials must use ${ENV_VAR} references")
    resolved = os.getenv(match[1]) or _read_dotenv_key(match[1])
    if resolved is None:
        raise ValueError(f"Missing tool environment variable: {match[1]}")
    return resolved


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, AgentTool] = {}
        self.servers: dict[str, dict[str, Any]] = {}
        self._tasks: list[asyncio.Task] = []
        self._stop: asyncio.Event | None = None

    def register(self, tool: AgentTool) -> None:
        if not isinstance(tool, AgentTool):
            raise TypeError("Python tool factories must return AgentTool instances")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool.name):
            raise ValueError("Tool names must be 1-64 alphanumeric, underscore or hyphen characters")
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        Draft202012Validator.check_schema(tool.parameters)
        self.tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self.tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": "unknown_tool", "message": "Tool is unavailable"}
        errors = sorted(Draft202012Validator(tool.parameters).iter_errors(arguments),
                        key=lambda error: str(list(error.path)))
        if errors:
            # Do not echo supplied values: they may contain secrets.
            return {"error": "invalid_arguments", "fields": [list(e.path) for e in errors],
                    "message": "Arguments do not match the tool JSON schema"}
        try:
            return await tool.execute(**arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record_error("agent_tool", exc)
            return {"error": type(exc).__name__, "message": "Tool execution failed"}

    def load_config(self, path: str | Path | None = None) -> None:
        explicit = path is not None or setting("AGENT_TOOLS_CONFIG", "") != ""
        config_path = Path(path or setting("AGENT_TOOLS_CONFIG", "tools.json"))
        if not config_path.exists() and not explicit:
            return
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        for plugin in data.get("python", []):
            if not plugin.get("enabled", True):
                continue
            module, factory = plugin["factory"].split(":", 1)
            result = getattr(importlib.import_module(module), factory)()
            for tool in result if isinstance(result, (list, tuple)) else [result]:
                self.register(tool)
        for name, config in data.get("mcp", {}).items():
            if not config.get("enabled", True):
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", name):
                raise ValueError("Invalid MCP server name")
            if config.get("transport") not in {"stdio", "streamable_http"}:
                raise ValueError(f"Unsupported MCP transport: {name}")
            self.servers[name] = config

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop = asyncio.Event()
        ready = []
        for name, config in self.servers.items():
            event = asyncio.Event()
            ready.append(event.wait())
            self._tasks.append(asyncio.create_task(self._serve(name, config, event)))
        if ready:
            await asyncio.gather(*ready)

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _serve(self, name: str, config: dict, ready: asyncio.Event) -> None:
        registered: list[str] = []
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
            # Enter/exit SDK task groups in this same task (AnyIO requirement).
            async with AsyncExitStack() as stack:
                async with asyncio.timeout(20):
                    if config["transport"] == "stdio":
                        env = {key: env_reference(value) for key, value in config.get("env", {}).items()}
                        transport = stdio_client(StdioServerParameters(
                            command=config["command"], args=config.get("args", []), env=env))
                    else:
                        from urllib.parse import urlsplit
                        url = config["url"]
                        parsed = urlsplit(url)
                        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                            raise ValueError("MCP URLs must not contain credentials or query strings")
                        headers = {key: env_reference(value) for key, value in config.get("headers", {}).items()}
                        transport = streamablehttp_client(url, headers=headers)
                    streams = await stack.enter_async_context(transport)
                    session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                    await session.initialize()
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        for remote in page.tools:
                            tool_name = f"{name}__{remote.name}"
                            async def execute(_remote=remote.name, **arguments):
                                try:
                                    result = await session.call_tool(_remote, arguments=arguments)
                                except Exception:
                                    for registered_name in registered:
                                        self.tools.pop(registered_name, None)
                                    raise
                                # Tools return textual content only; never forward image Base64.
                                return {"content": [item.text for item in result.content if item.type == "text"],
                                        "is_error": bool(result.isError)}
                            self.register(AgentTool(tool_name, remote.description or tool_name, remote.inputSchema, execute))
                            registered.append(tool_name)
                        cursor = page.nextCursor
                        if not cursor:
                            break
                ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record_error("mcp_connection", exc)
        finally:
            for tool_name in registered:
                self.tools.pop(tool_name, None)
            ready.set()
