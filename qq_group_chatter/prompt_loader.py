from importlib.resources import files


def load_prompt(name: str) -> str:
    return files("qq_group_chatter").joinpath("prompts", name).read_text(encoding="utf-8").strip()
