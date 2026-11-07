import json

import pytest

from qq_group_chatter.agent.chat_agent import CHAT_AGENT_SYSTEM_PROMPT
from qq_group_chatter.memory_dashboard import build_prompts_snapshot, prompts_api


def test_prompt_snapshot_exposes_loaded_templates_separately_from_disk():
    snapshot = build_prompts_snapshot()
    assert snapshot['groups'][0]['system'] == CHAT_AGENT_SYSTEM_PROMPT
    assert '{chat_context_rules}' not in snapshot['groups'][0]['system']
    assert '{chat_context_rules}' in snapshot['source']['chat_agent_system.txt']
    assert '{user_message}' in snapshot['groups'][0]['user']
    assert len(snapshot['source']) == 11
    assert not snapshot['errors']


@pytest.mark.asyncio
async def test_prompt_api_is_not_cached():
    response = await prompts_api()
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert json.loads(response.content)['groups'][0]['key'] == 'chat'
