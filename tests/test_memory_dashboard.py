from qq_group_chatter.memory_dashboard import (
    build_memory_dashboard_snapshot,
    memory_dashboard_api,
    memory_dashboard_response,
    memory_dashboard_html,
    setup_memory_dashboard,
)


class FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakeVectorStore:
    def list(self, filters=None, top_k=100):
        return (
            [
                FakePoint(
                    "mem-user-1",
                    {
                        "data": "用户喜欢咖啡",
                        "user_id": "qq_user:123456",
                        "kind": "preference",
                        "scope": "user",
                        "source_created_at": 100.0,
                        "created_at": "2026-06-15T13:00:00Z",
                    },
                ),
                FakePoint(
                    "mem-conv-1",
                    {
                        "data": "这个群默认中文交流",
                        "user_id": "qq_conversation:qq_group:888888",
                        "kind": "conversation_rule",
                        "scope": "conversation",
                        "updated_at": "2026-06-15T13:01:00Z",
                    },
                ),
            ],
            None,
        )


class FakeMem0:
    def __init__(self):
        self.vector_store = FakeVectorStore()

    def history(self, memory_id):
        return [
            {
                "memory_id": memory_id,
                "event": "ADD",
                "new_memory": f"created {memory_id}",
                "created_at": "2026-06-15T13:00:00Z",
            }
        ]


class FakeLongTermMemory:
    def __init__(self):
        self._mem0 = FakeMem0()


class FakeApplication:
    def __init__(self):
        self.long_term_memory = FakeLongTermMemory()


class FakeDriver:
    def __init__(self):
        self.routes = {}

    def setup_http_server(self, setup):
        self.routes[setup.path.path] = setup


def test_build_memory_dashboard_snapshot_lists_memories_and_history():
    snapshot = build_memory_dashboard_snapshot(FakeApplication())

    assert snapshot["summary"] == {
        "total": 2,
        "user": 1,
        "conversation": 1,
        "other": 0,
    }
    memories = {item["id"]: item for item in snapshot["memories"]}
    assert memories["mem-user-1"]["content"] == "用户喜欢咖啡"
    assert memories["mem-user-1"]["history"][0]["event"] == "ADD"
    assert memories["mem-conv-1"]["owner_id"] == "qq_conversation:qq_group:888888"


def test_memory_dashboard_html_contains_bootstrap_snapshot():
    html = memory_dashboard_html({"summary": {"total": 0}, "memories": [], "errors": []})

    assert "<!doctype html>" in html
    assert "window.__MEMORY_SNAPSHOT__" in html
    assert "长期记忆" in html


async def test_memory_dashboard_handlers_return_html_and_json_responses():
    html_response = await memory_dashboard_response(FakeApplication())
    api_response = await memory_dashboard_api(FakeApplication())

    assert html_response.status_code == 200
    assert "text/html" in html_response.headers["content-type"]
    assert "长期记忆" in html_response.content
    assert api_response.status_code == 200
    assert "application/json" in api_response.headers["content-type"]
    assert "用户喜欢咖啡" in api_response.content


def test_setup_memory_dashboard_registers_html_and_api_routes():
    driver = FakeDriver()

    setup_memory_dashboard(driver, FakeApplication())

    assert driver.routes["/memory"].method == "GET"
    assert driver.routes["/api/memory"].method == "GET"
