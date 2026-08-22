import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway import GatewayService


ZENMUX_ANTHROPIC_BASE_URL = "https://zenmux.ai/api/anthropic"
MODEL_PLACEHOLDER = "ZENMUX_MODEL_PLACEHOLDER"
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
        "public_model": "zenmux-anthropic-cache-test-placeholder",
        "upstream_model": MODEL_PLACEHOLDER,
        "upstream": {
            "name": "zenmux-anthropic-cache-test",
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
        "model": "zenmux-anthropic-cache-test-placeholder",
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
    assert call.args[0] == f"{ZENMUX_ANTHROPIC_BASE_URL}/messages"
    forwarded = call.kwargs["json"]
    assert REQUEST_ID not in json.dumps(forwarded)
    assert cache_control_from_content(forwarded["system"]) == {"type": "ephemeral"}
    assert forwarded["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    history_assistant = next(
        message for message in forwarded["messages"] if message["role"] == "assistant"
    )
    assert cache_control_from_content(history_assistant["content"]) == {"type": "ephemeral"}
    assert "ttl" not in json.dumps(forwarded)


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
