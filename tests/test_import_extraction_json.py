import json
from types import SimpleNamespace

import pytest

from dehydrator import Dehydrator
from import_memory import (
    IMPORT_EXTRACT_JSON_OBJECT_PROMPT,
    ImportEngine,
    _EXTRACT_MAX_ITEMS,
)


def test_clean_llm_json_extracts_first_balanced_json_value():
    from utils import clean_llm_json

    raw = '说明：{"ok": true, "items": [1, 2]} done'

    assert clean_llm_json(raw) == '{"ok": true, "items": [1, 2]}'


def test_import_extraction_accepts_json_array_with_model_chatter():
    raw = """
可以，下面是我从这段对话里提取出的记忆：
[
  {
    "name": "偏好",
    "content": "用户更喜欢 Dashboard 批量导入能容忍模型在 JSON 外补充说明。",
    "domain": ["AI"],
    "valence": 0.6,
    "arousal": 0.4,
    "tags": ["导入", "DeepSeek"],
    "importance": 6,
    "preserve_raw": false,
    "is_pattern": false
  }
]
如果需要，我也可以继续帮你细分。
"""

    items = ImportEngine._parse_extraction(raw)

    assert len(items) == 1
    assert items[0]["name"] == "偏好"
    assert items[0]["tags"] == ["导入", "DeepSeek"]


def test_import_extraction_accepts_legacy_bare_array():
    items = ImportEngine._parse_extraction('[{"content":"legacy"}]')

    assert [item["content"] for item in items] == ["legacy"]


def test_import_extraction_accepts_memories_wrapper():
    items = ImportEngine._parse_extraction(
        '{"memories":[{"content":"wrapped"}]}'
    )

    assert [item["content"] for item in items] == ["wrapped"]


def test_import_extraction_accepts_empty_memories_wrapper():
    assert ImportEngine._parse_extraction('{"memories":[]}') == []


def test_import_extraction_rejects_object_without_memories():
    with pytest.raises(ValueError, match="memories"):
        ImportEngine._parse_extraction('{"items":[]}')


def test_import_extraction_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        ImportEngine._parse_extraction('{"memories":[')


@pytest.mark.asyncio
async def test_deepseek_extraction_uses_json_output_and_enables_thinking(tmp_path):
    captured = {}

    class DeepSeekDehydrator:
        api_available = True
        api_format = "openai_compat"
        base_url = "https://api.deepseek.com/v1"
        model = "deepseek-v4-pro"

        async def _chat(self, prompt, content, **kwargs):
            captured.update(prompt=prompt, content=content, kwargs=kwargs)
            return '{"memories":[]}'

    engine = ImportEngine(
        {"buckets_dir": str(tmp_path), "human": "用户"},
        bucket_mgr=None,
        dehydrator=DeepSeekDehydrator(),
    )

    assert await engine._extract_memories("用户：测试") == []
    assert captured["prompt"] == IMPORT_EXTRACT_JSON_OBJECT_PROMPT
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["request_extra_body"] == {
        "thinking": {"type": "enabled"}
    }
    assert "禁止解释文字、Markdown code fence、前缀或后缀文本" in captured["prompt"]


@pytest.mark.asyncio
async def test_openai_compat_forwards_per_request_json_output_options(tmp_path):
    dehydrator = Dehydrator(
        {
            "buckets_dir": str(tmp_path),
            "dehydration": {
                "api_key": "test-key",
                "extra_body": {"existing": True},
            },
        }
    )
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
            )

    dehydrator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    try:
        await dehydrator._chat_once(
            "system",
            "user",
            response_format={"type": "json_object"},
            request_extra_body={"thinking": {"type": "disabled"}},
        )
    finally:
        dehydrator.close()

    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {
        "existing": True,
        "thinking": {"type": "disabled"},
    }


@pytest.mark.parametrize("raw", ["not json", '{"content":"not an array"}'])
def test_import_extraction_rejects_malformed_or_non_array_results(raw):
    with pytest.raises(ValueError, match="JSON|array"):
        ImportEngine._parse_extraction(raw)


def test_import_extraction_normalizes_null_and_scalar_collection_fields():
    raw = """[
      {"content":"A", "domain":null, "tags":null},
      {"content":"B", "domain":"工作", "tags":"重要"},
      {"content":"C", "domain":{"bad":true}, "tags":123}
    ]"""

    items = ImportEngine._parse_extraction(raw)

    assert [item["domain"] for item in items] == [
        ["未分类"], ["工作"], ["未分类"]
    ]
    assert [item["tags"] for item in items] == [[], ["重要"], []]


def test_import_extraction_enforces_item_cap_after_validation():
    raw = json.dumps(
        [{"content": f"memory-{index}"} for index in range(100)]
    )

    items = ImportEngine._parse_extraction(raw)

    assert len(items) == _EXTRACT_MAX_ITEMS
    assert [item["content"] for item in items] == [
        f"memory-{index}" for index in range(_EXTRACT_MAX_ITEMS)
    ]
