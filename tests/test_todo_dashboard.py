import json
from pathlib import Path
from uuid import uuid4

import pytest

from qq_group_chatter import memory_dashboard as dashboard


@pytest.mark.asyncio
async def test_todo_api_reads_edits_and_handles_missing_file(monkeypatch):
    path = Path('tests/.tmp') / f'todo-{uuid4().hex}.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dashboard, 'TODO_PATH', path)
    response = await dashboard.todo_api()
    assert json.loads(response.content) == {'content': '', 'exists': False}
    for text in ['# TODO\n- [ ] 图片编号', '# TODO\n- [x] 图片编号']:
        path.write_text(text, encoding='utf-8')
        response = await dashboard.todo_api()
        assert response.headers['cache-control'] == 'no-store'
        assert json.loads(response.content)['content'] == text
    path.unlink()
