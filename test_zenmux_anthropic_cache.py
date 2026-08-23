import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import httpx
import pytest

from gateway import GatewayService


ZENMUX_ANTHROPIC_BASE_URL = "https://zenmux.ai/api/anthropic"
SONNET_ALIAS = "cc-home-claude-sonnet"
SONNET_MODEL = "anthropic/claude-sonnet-4.6"
OPUS_ALIAS = "cc-home-claude-opus"
OPUS_MODEL = "anthropic/claude-opus-4.6"
REQUEST_ID = "019d3fd2-44a6-7c64-8e0d-455658bbb09d"


def build_service():
    service = object.__new__(GatewayService)
    service.gateway_cfg = {}
    service.identity = {"ai_name": "Identity Placeholder"}
    service.inject_total_budget = 100_000
    service._memory_reading_policy_context = lambda: "memory-policy-marker"
    return service


def zenmux_route():
    return {
        "public_model": SONNET_ALIAS,
        "upstream_model": SONNET_MODEL,
        "upstream": {
            "name": "zenmux-anthropic",
            "base_url": ZENMUX_ANTHROPIC_BASE_URL,
            "protocol": "anthropic",
            "prompt_cache": "anthropic_explicit",
        },
    }


def cache_test_payload():
    # The long, deterministic history satisfies the Gateway's conservative local
    # breakpoint thresholds without estimating or asserting provider usage.
    long_prefix = "stable-history-prefix " * 5_000
    long_tail = "current-turn-tail " * 5_000
    return {
        "request_id": REQUEST_ID,
        "model": SONNET_ALIAS,
        "messages": [
            {"role": "system", "content": "stable-system-marker"},
            {"role": "user", "content": long_prefix},
            {"role": "assistant", "content": long_prefix},
            {"role": "user", "content": long_tail},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "first_placeholder_tool",
                    "description": "local mock only",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "last_placeholder_tool",
                    "description": "local mock only",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "max_tokens": 64,
        "stream": False,
    }


def cache_control_from_content(content):
    assert isinstance(content, list)
    return content[-1].get("cache_control")


@pytest.mark.asyncio
async def test_zenmux_anthropic_messages_uses_explicit_five_minute_cache_once():
    service = build_service()
    route = zenmux_route()
    payload = cache_test_payload()

    assert service._pop_chat_request_id(payload) == REQUEST_ID

    post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_local_mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "local mock reply"}],
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "cache_read_input_tokens": 80,
                    "cache_creation_input_tokens": 10,
                },
            },
        )
    )
    service.http_client = SimpleNamespace(post=post)
    service._available_upstream_api_keys = lambda _upstream: [
        {"label": "fake-test-key", "value": "fake-zenmux-test-placeholder"}
    ]
    service._anthropic_upstream_headers = (
        lambda *_args, **_kwargs: {
            "x-api-key": "fake-zenmux-test-placeholder",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    )
    service._clear_upstream_key_cooldown = lambda *_args: None
    service._cool_down_upstream_key = lambda *_args: None
    service._should_retry_upstream_status = lambda _status: False

    response = await service._forward_anthropic_upstream(payload, route)

    assert response.status_code == 200
    assert post.await_count == 1
    call = post.await_args
    assert call.args[0] == f"{ZENMUX_ANTHROPIC_BASE_URL}/v1/messages"
    forwarded = call.kwargs["json"]
    assert REQUEST_ID not in json.dumps(forwarded)
    assert cache_control_from_content(forwarded["system"]) == {"type": "ephemeral"}
    assert forwarded["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    history_assistant = next(
        message for message in forwarded["messages"] if message["role"] == "assistant"
    )
    assert cache_control_from_content(history_assistant["content"]) == {"type": "ephemeral"}
    assert "ttl" not in json.dumps(forwarded)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://zenmux.ai/api/anthropic",
            "https://zenmux.ai/api/anthropic/v1/messages",
        ),
        (
            "https://api.anthropic.example.invalid/v1",
            "https://api.anthropic.example.invalid/v1/messages",
        ),
        (
            "https://legacy-anthropic.example.invalid/api",
            "https://legacy-anthropic.example.invalid/api/messages",
        ),
        (
            "https://custom-anthropic.example.invalid/v1/messages/",
            "https://custom-anthropic.example.invalid/v1/messages",
        ),
    ],
)
def test_anthropic_messages_url_preserves_other_provider_conventions(base_url, expected):
    assert GatewayService._anthropic_messages_url({"base_url": base_url}) == expected


def test_identity_persona_recent_and_recall_keep_current_injection_order():
    service = build_service()
    stable, dynamic = service._build_injected_context_messages(
        persona_block="persona-marker",
        core_memory="core-marker",
        portrait_memory="",
        recalled_memory="recall-marker",
        related_memory="diffusion-marker",
        recent_context="recent-marker",
        favorite_memory="favorite-marker",
    )

    assert "core-marker" in stable
    assert "Identity Placeholder Favorite Memory" in dynamic
    assert dynamic.index("recall-marker") < dynamic.index("diffusion-marker")
    assert dynamic.index("diffusion-marker") < dynamic.index("recent-marker")
    assert dynamic.index("recent-marker") < dynamic.index("persona-marker")

    original = [
        {"role": "user", "content": "older-user-marker"},
        {"role": "assistant", "content": "older-assistant-marker"},
        {"role": "user", "content": "current-user-marker"},
    ]
    injected = service._inject_context_messages(original, stable, dynamic)

    assert injected[0] == {"role": "system", "content": stable}
    assert injected[1:3] == original[:2]
    assert injected[3]["content"].startswith("<ombre_live_context>")
    assert injected[3]["content"].endswith("current-user-marker")
    assert original[-1]["content"] == "current-user-marker"


def test_current_shanghai_time_is_dynamic_and_authoritative():
    service = build_service()
    service.gateway_tz = ZoneInfo("Asia/Shanghai")
    current_time = service._current_time_context(
        datetime(2026, 8, 23, 7, 25, tzinfo=timezone.utc)
    )

    stable, dynamic = service._build_injected_context_messages(
        persona_block="persona-marker",
        core_memory="core-marker",
        portrait_memory="",
        current_time_context=current_time,
    )

    assert "2026-08-23 15:25" in current_time
    assert "Sunday (Asia/Shanghai)" in current_time
    assert current_time not in stable
    assert "Current Local Time" in dynamic
    assert current_time in dynamic
    injected = service._inject_context_messages(
        [{"role": "user", "content": "current-user-marker"}],
        stable,
        dynamic,
    )
    assert current_time not in injected[0]["content"]
    assert current_time in injected[-1]["content"]


def test_short_chat_marks_latest_completed_assistant_as_cache_breakpoint():
    service = build_service()
    route = zenmux_route()
    payload = {
        "model": SONNET_ALIAS,
        "messages": [
            {"role": "user", "content": "first short question"},
            {"role": "assistant", "content": "first short answer"},
            {"role": "user", "content": "second short question"},
        ],
    }

    forwarded = service._anthropic_payload_for_upstream(payload, route)

    assert cache_control_from_content(forwarded["messages"][1]["content"]) == {
        "type": "ephemeral"
    }
    assert "cache_control" not in json.dumps(forwarded["messages"][2])


def test_real_cache_usage_is_returned_and_missing_values_stay_null():
    usage = GatewayService._actual_usage_diagnostics(
        {
            "input_tokens": 120,
            "output_tokens": 30,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 10,
        }
    )
    assert usage == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": None,
        "cached_tokens": None,
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 10,
    }

    missing = GatewayService._actual_usage_diagnostics(
        {"input_tokens": 7, "output_tokens": 2}
    )
    assert missing["input_tokens"] == 7
    assert missing["output_tokens"] == 2
    assert missing["total_tokens"] is None
    assert missing["cached_tokens"] is None
    assert missing["cache_read_input_tokens"] is None
    assert missing["cache_creation_input_tokens"] is None


def test_cache_strategy_is_scoped_to_the_selected_upstream():
    service = build_service()
    route = zenmux_route()
    route["upstream"] = {
        **route["upstream"],
        "name": "other-anthropic-compatible-upstream",
        "prompt_cache": "",
    }
    payload = cache_test_payload()
    service._pop_chat_request_id(payload)

    forwarded = service._anthropic_payload_for_upstream(payload, route)

    assert "cache_control" not in forwarded
    assert all("cache_control" not in tool for tool in forwarded["tools"])
    assert all("cache_control" not in json.dumps(message) for message in forwarded["messages"])


def build_multi_upstream_service():
    service = build_service()
    service.gateway_cfg = {
        "upstream_default_model": "cc-home-default",
        "upstreams": [
            {
                "name": "existing-provider-placeholder",
                "protocol": "openai",
                "base_url": "https://existing-provider.example.invalid/v1",
                "api_key": "fake-existing-provider-key",
                "default_model": "cc-home-default",
                "models": [
                    {
                        "id": "cc-home-default",
                        "upstream_model": "existing-model-placeholder",
                    }
                ],
            },
            {
                "name": "zenmux-anthropic",
                "protocol": "anthropic",
                "base_url": ZENMUX_ANTHROPIC_BASE_URL,
                "api_key": "fake-zenmux-test-placeholder",
                "default_model": SONNET_ALIAS,
                "prompt_cache": "anthropic_explicit",
                "models": [
                    {"id": SONNET_ALIAS, "upstream_model": SONNET_MODEL},
                    {"id": OPUS_ALIAS, "upstream_model": OPUS_MODEL},
                ],
            },
        ],
    }
    service.upstream_base_url = ""
    service.upstream_default_model = "cc-home-default"
    service.upstream_models = []
    service.upstream_api_key = ""
    service.upstreams = service._load_upstreams()
    service._refresh_upstream_model_summary()
    return service


def test_public_aliases_route_without_replacing_existing_provider():
    service = build_multi_upstream_service()

    assert service.upstream_models == [
        "cc-home-default",
        SONNET_ALIAS,
        OPUS_ALIAS,
    ]

    default_route = service._resolve_upstream_for_model("cc-home-default")
    assert default_route["upstream"]["name"] == "existing-provider-placeholder"
    assert default_route["upstream_model"] == "existing-model-placeholder"

    sonnet_route = service._resolve_upstream_for_model(SONNET_ALIAS)
    assert sonnet_route["upstream"]["name"] == "zenmux-anthropic"
    assert sonnet_route["upstream_model"] == SONNET_MODEL

    opus_route = service._resolve_upstream_for_model(OPUS_ALIAS)
    assert opus_route["upstream"]["name"] == "zenmux-anthropic"
    assert opus_route["upstream_model"] == OPUS_MODEL


@pytest.mark.asyncio
async def test_model_list_exposes_only_public_aliases():
    service = build_multi_upstream_service()
    service.gateway_token = "fake-gateway-token"
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fake-gateway-token"}
    )

    response = await service.handle_models(request)
    body = json.loads(response.body)
    serialized = json.dumps(body)

    assert [item["id"] for item in body["data"]] == [
        "cc-home-default",
        SONNET_ALIAS,
        OPUS_ALIAS,
    ]
    assert "zenmux.ai" not in serialized
    assert "fake-" not in serialized
    assert SONNET_MODEL not in serialized
    assert OPUS_MODEL not in serialized
