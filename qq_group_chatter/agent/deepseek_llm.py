from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ThinkingMode = Literal["enabled", "disabled"]


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
                    "content": (
                        "\u4f60\u662f\u4e00\u4e2aQQ\u804a\u5929\u673a\u5668\u4eba\u3002"
                        "\u56de\u590d\u8981\u81ea\u7136\u3001\u7b80\u77ed\uff0c"
                        "\u4e0d\u8981\u89e3\u91ca\u601d\u8003\u8fc7\u7a0b\u3002"
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
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None
