from qq_group_chatter.prompt_loader import load_prompt


def load_bot_identity_prompt() -> str:
    return load_prompt("bot_identity.txt")


BOT_IDENTITY_PROMPT = load_bot_identity_prompt()
