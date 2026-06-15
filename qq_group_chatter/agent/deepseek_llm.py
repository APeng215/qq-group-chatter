from __future__ import annotations

import os
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
    max_tokens: int = 320
    base_url: str = "https://api.deepseek.com"
    client: Any | None = None

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

    async def ainvoke(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": DEEPSEEK_SYSTEM_PROMPT_TEMPLATE.format(
                        bot_identity_prompt=BOT_IDENTITY_PROMPT,
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            extra_body={"thinking": {"type": self.thinking}},
        )
        return response.choices[0].message.content or ""


def create_deepseek_chat_llm(
    *,
    api_key: str | None = None,
    model: str = "deepseek-v4-pro",
    thinking: ThinkingMode = "disabled",
) -> DeepSeekChatLLM | None:
    resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY") or _read_dotenv_key()
    if not resolved_key:
        return None
    return DeepSeekChatLLM(
        api_key=resolved_key,
        model=model,
        thinking=thinking,
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
