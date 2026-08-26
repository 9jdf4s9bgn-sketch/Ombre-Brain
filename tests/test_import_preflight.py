import asyncio
import json
import threading

import pytest

import web.import_api as import_api


class FakeMCP:
    def __init__(self):
        self.routes = {}

    def custom_route(self, path, methods):
        def decorator(fn):
            for method in methods:
                self.routes[(method, path)] = fn
            return fn

        return decorator


class BodyRequest:
    def __init__(
        self,
        body: str,
        filename: str = "upload.md",
        **query_params,
    ):
        self.headers = {}
        self.query_params = {"filename": filename, **query_params}
        self._body = body.encode("utf-8")

    async def body(self):
        return self._body


class FakeDehydrator:
    api_available = True


class FakeImportEngine:
    is_running = False
    dehydrator = FakeDehydrator()


class QueryRequest:
    def __init__(self, **query_params):
        self.query_params = query_params


class FakeBucketManager:
    def __init__(self, buckets):
        self.buckets = buckets

    async def list_all(self, include_archive=False):
        assert include_archive is False
        return list(self.buckets)


def test_preview_import_counts_turns_chunks_and_estimated_calls():
    from import_memory import preview_import

    raw = "Human: 我喜欢茶\nAssistant: 我记住了\nUser: 明天提醒我整理导入体验"

    preview = preview_import(raw, filename="chat.md", human_label="阿立")

    assert preview["ok"] is True
    assert preview["detected_format"] == "markdown"
    assert preview["turns_count"] == 3
    assert preview["chunks_count"] == 1
    assert preview["estimated_api_calls"] == 1
    assert "[阿立]" in preview["first_chunk_preview"]


def test_preview_import_warns_when_invalid_json_falls_back_to_text():
    from import_memory import preview_import

    preview = preview_import("{not json", filename="bad.json")

    assert preview["ok"] is True
    assert preview["detected_format"] == "text"
    assert preview["turns_count"] == 1
    assert any("JSON" in warning for warning in preview["warnings"])


def test_preview_import_handles_deep_json_without_internal_error():
    from import_memory import preview_import

    raw = "[" * 100_000 + "0" + "]" * 100_000
    preview = preview_import(raw, filename="deep.json")

    assert preview["ok"] is True
    assert preview["detected_format"] == "text"
    assert any("JSON" in warning for warning in preview["warnings"])


def test_preview_import_handles_out_of_range_chatgpt_timestamp():
    from import_memory import preview_import

    raw = json.dumps({
        "mapping": {
            "node": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["仍应正常预检"]},
                    "create_time": 1e100,
                }
            }
        }
    })
    preview = preview_import(raw, filename="chatgpt.json")

    assert preview["ok"] is True
    assert preview["turns_count"] == 1
    assert preview["sample_turns"][0]["timestamp"] == ""


@pytest.mark.asyncio
async def test_import_preflight_route_returns_preview_with_runtime_readiness(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})

    mcp = FakeMCP()
    import_api.register(mcp)

    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest("Human: hi\nAssistant: hello", filename="chat.md")
    )
    payload = json.loads(response.body)

    assert payload["ok"] is True
    assert payload["can_start"] is True
    assert payload["llm_ready"] is True
    assert payload["import_running"] is False
    assert payload["filename"] == "chat.md"
    assert payload["turns_count"] == 2
    assert payload["chunks_count"] == 1


@pytest.mark.asyncio
async def test_import_preflight_returns_resolved_echo_identity_snapshot(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "人类"})

    raw = """1｜2026-08-25 10:00:00.000 +08:00｜松松｜\\
你好

2｜2026-08-25 10:00:01.000 +08:00｜尼奥｜\\
你好呀
"""
    mcp = FakeMCP()
    import_api.register(mcp)

    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest(
            raw,
            filename="echo.md",
            human_label=" 松松 ",
            assistant_label=" 尼奥 ",
        )
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["identity_mapping"] == {
        "applied": True,
        "human_label": "松松",
        "assistant_label": "尼奥",
    }
    assert 'role=user speaker="松松"' in payload["first_chunk_preview"]
    assert 'role=assistant speaker="尼奥"' in payload["first_chunk_preview"]


@pytest.mark.asyncio
async def test_import_preflight_echo_identity_uses_global_fallback(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "人类"})
    monkeypatch.setattr(import_api, "get_ai_name", lambda: "AI")

    raw = """1｜2026-08-25 10:00:00.000 +08:00｜人类｜\\
你好

2｜2026-08-25 10:00:01.000 +08:00｜AI｜\\
你好呀
"""
    mcp = FakeMCP()
    import_api.register(mcp)
    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest(raw, human_label="   ", assistant_label="")
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["identity_mapping"] == {
        "applied": True,
        "human_label": "人类",
        "assistant_label": "AI",
    }


@pytest.mark.parametrize(
    ("raw", "expected_roles"),
    [
        ("Human: 你好\nAssistant: 你好呀", ["user", "assistant"]),
        (
            json.dumps({
                "chat_messages": [
                    {"sender": "human", "text": "Claude 用户消息"},
                    {"sender": "assistant", "text": "Claude AI 消息"},
                ]
            }),
            ["human", "assistant"],
        ),
        (
            json.dumps({
                "mapping": {
                    "user": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["ChatGPT 用户消息"]},
                        }
                    },
                    "assistant": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["ChatGPT AI 消息"]},
                        }
                    }
                }
            }),
            ["user", "assistant"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_import_preflight_ignores_echo_mapping_for_other_formats(
    raw,
    expected_roles,
    monkeypatch,
):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "人类"})
    monkeypatch.setattr(import_api, "get_ai_name", lambda: "AI")

    mcp = FakeMCP()
    import_api.register(mcp)
    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest(raw, human_label="松松", assistant_label="尼奥")
    )
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["identity_mapping"] == {
        "applied": False,
        "human_label": "人类",
        "assistant_label": "AI",
    }
    assert [turn["role"] for turn in payload["sample_turns"]] == expected_roles
    assert "松松" not in payload["first_chunk_preview"]
    assert "尼奥" not in payload["first_chunk_preview"]


@pytest.mark.parametrize(
    ("human_label", "assistant_label", "error"),
    [
        ("同名", "同名", "must be different"),
        ("松\n松", "尼奥", "control characters"),
        ("松" * 21, "尼奥", "at most 20"),
        ("松松", "尼" * 41, "at most 40"),
    ],
)
@pytest.mark.asyncio
async def test_import_preflight_validates_identity_query_parameters(
    human_label,
    assistant_label,
    error,
    monkeypatch,
):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "人类"})

    raw = "1｜2026-08-25 10:00:00.000 +08:00｜松松｜\\\n你好"
    mcp = FakeMCP()
    import_api.register(mcp)
    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest(
            raw,
            human_label=human_label,
            assistant_label=assistant_label,
        )
    )

    assert response.status_code == 400
    assert error in json.loads(response.body)["error"]


@pytest.mark.asyncio
async def test_import_preflight_route_hides_preview_worker_failure(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})

    def reject_preview(*_args, **_kwargs):
        raise RecursionError("malicious nesting detail")

    monkeypatch.setattr(import_api, "preview_import", reject_preview)
    mcp = FakeMCP()
    import_api.register(mcp)

    response = await mcp.routes[("POST", "/api/import/preflight")](
        BodyRequest("{}", filename="bad.json")
    )
    payload = json.loads(response.body)

    assert response.status_code == 400
    assert payload == {
        "ok": False,
        "error": "Import preview could not parse file",
    }
    assert "malicious nesting detail" not in response.body.decode("utf-8")


@pytest.mark.asyncio
async def test_preflight_is_off_loop_and_rejects_parallel_body_before_read(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {"human": "阿立"})
    entered = threading.Event()
    release = threading.Event()

    def blocking_preview(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2)
        return {"ok": True, "turns_count": 1, "chunks_count": 1}

    monkeypatch.setattr(import_api, "preview_import", blocking_preview)
    mcp = FakeMCP()
    import_api.register(mcp)
    preflight = mcp.routes[("POST", "/api/import/preflight")]

    class MustNotReadRequest(BodyRequest):
        async def body(self):
            raise AssertionError("parallel preflight must be rejected before body read")

    first = asyncio.create_task(preflight(BodyRequest("Human: one")))
    while not entered.is_set():
        await asyncio.sleep(0)

    second = await preflight(MustNotReadRequest("Human: two"))
    assert second.status_code == 409
    assert "active" in json.loads(second.body)["error"].lower()

    release.set()
    response = await asyncio.wait_for(first, timeout=2)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_import_results_filters_imported_buckets_and_paginates(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {})
    monkeypatch.setattr(import_api.sh, "bucket_mgr", FakeBucketManager([
        {
            "id": "ordinary-newest",
            "content": "not imported",
            "metadata": {"name": "ordinary", "created": "2026-07-26T12:00:00"},
        },
        {
            "id": "import-new",
            "content": "new imported body",
            "metadata": {
                "name": "new import",
                "created": "2026-07-26T11:00:00",
                "imported": True,
            },
        },
        {
            "id": "import-legacy-source",
            "content": "legacy source marker",
            "metadata": {
                "name": "legacy import",
                "created": "2026-07-26T10:00:00",
                "source_tool": "import",
            },
        },
    ]))

    mcp = FakeMCP()
    import_api.register(mcp)
    route = mcp.routes[("GET", "/api/import/results")]

    first = json.loads((await route(QueryRequest(limit="1", offset="0"))).body)
    second = json.loads((await route(QueryRequest(limit="1", offset="1"))).body)

    assert first["total"] == 2
    assert first["has_more"] is True
    assert first["next_offset"] == 1
    assert [bucket["id"] for bucket in first["buckets"]] == ["import-new"]
    assert first["buckets"][0]["imported"] is True
    assert second["has_more"] is False
    assert second["next_offset"] is None
    assert [bucket["id"] for bucket in second["buckets"]] == [
        "import-legacy-source"
    ]


@pytest.mark.asyncio
async def test_import_results_rejects_negative_or_non_integer_offset(monkeypatch):
    monkeypatch.setattr(import_api.sh, "_require_auth", lambda request: None)
    monkeypatch.setattr(import_api.sh, "import_engine", FakeImportEngine())
    monkeypatch.setattr(import_api.sh, "config", {})
    monkeypatch.setattr(import_api.sh, "bucket_mgr", FakeBucketManager([]))
    mcp = FakeMCP()
    import_api.register(mcp)
    route = mcp.routes[("GET", "/api/import/results")]

    for offset in ("-1", "nan"):
        response = await route(QueryRequest(offset=offset))
        assert response.status_code == 400
