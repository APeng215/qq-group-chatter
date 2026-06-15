from pathlib import Path

from qq_group_chatter.agent.identity import BOT_IDENTITY_PROMPT, load_bot_identity_prompt
from qq_group_chatter.prompt_loader import load_prompt


def test_bot_identity_prompt_is_loaded_from_prompt_file():
    prompt_path = Path("qq_group_chatter/prompts/bot_identity.txt")

    assert prompt_path.exists()
    assert load_bot_identity_prompt() == prompt_path.read_text(encoding="utf-8").strip()
    assert BOT_IDENTITY_PROMPT == load_bot_identity_prompt()


def test_prompt_files_are_loaded_from_prompt_directory():
    for name in [
        "bot_identity.txt",
        "chat_agent.txt",
        "deepseek_system.txt",
        "long_term_memory_extractor.txt",
        "long_term_memory_section.txt",
    ]:
        prompt_path = Path("qq_group_chatter/prompts") / name

        assert prompt_path.exists()
        assert load_prompt(name) == prompt_path.read_text(encoding="utf-8").strip()
