from importlib.resources import files
import re


def parse_prompt_bundle(text: str) -> dict[str, str]:
    """Read named blocks; headings outside blocks never enter model input."""
    prompts = {}
    for name, body in re.findall(
        r"^<!-- prompt: ([\w.-]+) -->\n(.*?)\n<!-- /prompt -->$",
        text, re.MULTILINE | re.DOTALL,
    ):
        if name in prompts:
            raise ValueError(f"Duplicate prompt: {name}")
        prompts[name] = body.strip()
    if not prompts:
        raise ValueError("No prompt blocks found")
    return prompts


def load_prompt(name: str) -> str:
    text = files("qq_group_chatter").joinpath("prompts", "prompts.md").read_text(encoding="utf-8")
    try:
        return parse_prompt_bundle(text)[name]
    except KeyError:
        raise FileNotFoundError(f"Unknown prompt: {name}") from None
