import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway import GatewayService


REQUEST_ID = "019d3fd2-44a6-7c64-8e0d-455658bbb09d"


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {
            "Authorization": "Bearer fake-test-token",
            "X-Ombre-Session-Id": "cc-home-main",
        }

    async def json(self):
        return self._payload


def response_json(response):
    return json.loads(response.body.decode("utf-8"))


def build_service(*, usage=None, injection_debug=None, upstream_status=200, upstream_body=None):
    service = object.__new__(GatewayService)
    service.default_session_id = "default"
    service.upstream_default_model = "mock-model"
    service._authorize = lambda _authorization: None
    service._client_label_from_request = lambda _request, _route: "cc-home-test"
    service._summarize_messages_for_debug = lambda _messages: "redacted"
    service._strip_favorite_memory_marker_from_payload = lambda payload: (payload, False)
    service._truthy_header = lambda _value: False
    service._extract_last_user_query = lambda _messages: "hello"

    debug = injection_debug if injection_debug is not None else {
        "recent_context_injected": True,
        "recalled_bucket_ids": ["recall-a", "recall-b"],
        "diffused_bucket_ids": ["diffused-a"],
        "injected_bucket_ids": ["recall-a", "recall-b", "diffused-a"],
    }
    forward_payloads = []

    async def prepare_payload(payload, *_args, **_kwargs):
        forward_payloads.append(dict(payload))
        return dict(payload), ["recall-a", "recall-b"], debug

    service.prepare_payload = prepare_payload
    service._resolve_upstream_for_model = lambda _model: {"upstream": {"protocol": "openai"}}
    service._upstream_uses_anthropic_protocol = lambda _upstream: False

    body = upstream_body or {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock-model",
        "choices": [{"message": {"role": "assistant", "content": "safe reply"}}],
        "usage": usage or {},
    }
    upstream_response = httpx.Response(upstream_status, json=body)
    service._forward_upstream = AsyncMock(return_value=upstream_response)
    service._maybe_retry_with_memory_detail = AsyncMock(return_value=(upstream_response, None))
    service._log_cache_usage_from_response = lambda *_args, **_kwargs: body.get("usage") or None
    service._capture_reasoning_from_response = lambda *_args: None
    service._extract_assistant_message_from_response = (
        lambda response: response.json().get("choices", [{}])[0].get("message")
    )
    service._record_successful_round = AsyncMock(return_value=17)
    service._update_persona_after_assistant_message = AsyncMock()
    return service, forward_payloads, debug


@pytest.mark.asyncio
async def test_request_id_is_linked_without_forwarding_it_upstream():
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 80, "secret": "do-not-return"},
        "prompt_cache_hit_tokens": 70,
        "prompt_cache_miss_tokens": 50,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 10,
        "provider_private": "do-not-return",
    }
    service, forward_payloads, debug = build_service(usage=usage)

    response = await service.handle_chat(FakeRequest({
        "request_id": REQUEST_ID,
        "model": "mock-model",
        "messages": [{"role": "user", "content": "not diagnostic output"}],
        "stream": False,
    }))
    body = response_json(response)

    assert response.status_code == 200
    assert body["request_id"] == REQUEST_ID
    assert "request_id" not in forward_payloads[0]
    assert service._forward_upstream.await_count == 1
    assert debug["request_id"] == REQUEST_ID
    recorded_debug = service._record_successful_round.await_args.args[2]
    assert recorded_debug["request_id"] == REQUEST_ID
    assert body["diagnostics"] == {
        "gateway_round": 17,
        "recent_context_injected": True,
        "memory": {
            "recalled_count": 2,
            "diffused_count": 1,
            "injected_count": 3,
        },
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "cached_tokens": 80,
            "prompt_cache_hit_tokens": 70,
            "prompt_cache_miss_tokens": 50,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 10,
        },
    }
    serialized = json.dumps(body["diagnostics"])
    assert "provider_private" not in serialized
    assert "do-not-return" not in serialized
    assert "not diagnostic output" not in serialized


@pytest.mark.asyncio
async def test_missing_request_id_remains_compatible():
    service, forward_payloads, debug = build_service(usage={"input_tokens": 4, "output_tokens": 2})

    response = await service.handle_chat(FakeRequest({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }))
    body = response_json(response)

    assert response.status_code == 200
    assert "request_id" not in body
    assert "request_id" not in debug
    assert "request_id" not in forward_payloads[0]
    assert body["choices"][0]["message"]["content"] == "safe reply"
    assert body["diagnostics"]["usage"]["input_tokens"] == 4
    assert service._forward_upstream.await_count == 1


@pytest.mark.asyncio
async def test_missing_diagnostic_fields_are_null_and_never_estimated():
    service, _forward_payloads, _debug = build_service(
        usage={"prompt_tokens": 9, "completion_tokens": 3},
        injection_debug={},
    )

    response = await service.handle_chat(FakeRequest({
        "request_id": REQUEST_ID,
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }))
    diagnostics = response_json(response)["diagnostics"]

    assert diagnostics["recent_context_injected"] is None
    assert diagnostics["memory"] == {
        "recalled_count": None,
        "diffused_count": None,
        "injected_count": None,
    }
    assert diagnostics["usage"]["input_tokens"] == 9
    assert diagnostics["usage"]["output_tokens"] == 3
    assert diagnostics["usage"]["total_tokens"] is None
    assert diagnostics["usage"]["cached_tokens"] is None


@pytest.mark.asyncio
async def test_safe_error_does_not_proxy_sensitive_upstream_details():
    service, _forward_payloads, _debug = build_service(
        upstream_status=500,
        upstream_body={
            "error": "internal api-key fake-secret",
            "prompt": "private prompt",
            "headers": {"authorization": "Bearer fake-jwt"},
        },
    )

    response = await service.handle_chat(FakeRequest({
        "request_id": REQUEST_ID,
        "model": "mock-model",
        "messages": [{"role": "user", "content": "private message"}],
        "stream": False,
    }))
    body = response_json(response)
    serialized = json.dumps(body)

    assert response.status_code == 502
    assert body["request_id"] == REQUEST_ID
    assert body["error_stage"] == "upstream"
    assert body["error_code"] == "upstream_error"
    assert "fake-secret" not in serialized
    assert "private prompt" not in serialized
    assert "fake-jwt" not in serialized
    assert "private message" not in serialized
    assert service._forward_upstream.await_count == 1


@pytest.mark.asyncio
async def test_successful_round_persists_request_id_in_existing_debug_json():
    service = object.__new__(GatewayService)
    stored = {}
    service.state_store = SimpleNamespace(
        record_success=lambda _session_id, _recalled_ids: 23,
        record_recent_context_injection=lambda *_args: None,
        record_injection_debug=lambda session_id, round_id, payload: stored.update(
            session_id=session_id,
            round_id=round_id,
            payload=dict(payload),
        ),
    )
    service.reminder_store = SimpleNamespace(mark_reminded=lambda *_args, **_kwargs: None)
    service.bucket_mgr = SimpleNamespace(touch=AsyncMock())
    service._record_conversation_turn = lambda **_kwargs: None

    round_id = await service._record_successful_round(
        "cc-home-main",
        ["recall-a"],
        {"request_id": REQUEST_ID, "recent_context_injected": False},
        assistant_message={"role": "assistant", "content": "safe reply"},
    )

    assert round_id == 23
    assert stored == {
        "session_id": "cc-home-main",
        "round_id": 23,
        "payload": {"request_id": REQUEST_ID, "recent_context_injected": False},
    }
