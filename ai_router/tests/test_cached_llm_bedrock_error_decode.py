"""S2+S3 — Bedrock Error-Type Decoding + Retry-By-Error-Type.

Live-Probe 2026-05-10: Bedrock liefert `overloaded_error` als yielded
SSE `error`-Event vor `message_start`. SDK swallowt die Payload via
`accumulate_event` (anthropic/lib/streaming/_messages.py:454) — wir
sehen nur "Unexpected event order", nie den eigentlichen Error-Type.

Plan-S2: `_stream_once` bypasses `messages.stream(...).text_stream`,
nutzt `messages.create(stream=True)` direkt + iteriert raw
RawMessageStreamEvent. Decodiert `error`-Events in typed
`BedrockStream*Error`. Catched zusätzlich SDK-raised `APIStatusError`
(HTTP-4xx/5xx vor First-Event) und decodiert deren `.body` + falls
leer Status-Code-Fallback (529/503→overload, 429→rate_limit,
504→timeout, 401/403/404/400/402→typed).

Plan-S3: `_create_or_stream` retried ausschliesslich auf
`overloaded_error / rate_limit_error / api_error / timeout_error` mit
Backoff (5, 15, 45)s über max 4 Versuche. Non-retryable Error-Types
(`invalid_request_error / authentication_error / permission_error /
not_found_error / billing_error`) raisen sofort.

Diese Suite verifiziert:
1. Yielded error events (Object UND dict shape) → typed Exception.
2. APIStatusError vor First-Event (body decoded) → typed Exception.
3. APIStatusError mit leerem body → Status-Code-Fallback (529/429/...).
4. Unknown error_type → BedrockUnknownStreamError (NICHT OutOfOrder).
5. Retryable types → 4 Versuche, 3 Sleeps (5,15,45), kein Sleep nach
   letztem; non-retryable → sofort.
6. Erfolg auf 2./3. Versuch → korrekte Anzahl Sleeps + valide Response.
7. Content/Usage-Aggregation aus message_start + content_block_delta +
   message_delta — tolerant gegen dict|object Event-Shape.
"""

from unittest.mock import MagicMock, patch


def _make_anthropic_client(max_tokens: int = 96000):
    """CachedAnthropicClient mit gemocktem _setup_client."""
    from ai_router.cached_llm import CachedAnthropicClient

    with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
        client = CachedAnthropicClient(model="claude-sonnet-4-6", max_tokens=max_tokens)
    client._client = MagicMock()
    client._is_bedrock = True
    client._is_vertex = False
    client._supports_temp = False
    client._model_alias = "claude-sonnet-4-6"
    return client


class _IterableStream:
    """Simuliert anthropic.Stream — Context-Manager + iter über Events."""

    def __init__(self, events):
        self._events = list(events)

    def __iter__(self):
        for e in self._events:
            yield e

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def close(self):
        pass


def _obj_event(**kwargs):
    """Build a Pydantic-style event object via MagicMock with spec'd attrs."""
    m = MagicMock(spec=list(kwargs.keys()))
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def _err_obj(error_type, message):
    """Pydantic-style error sub-object."""
    return _obj_event(type=error_type, message=message)


def _yielded_error_event_obj(error_type, message, request_id="req_test_abc"):
    return _obj_event(type="error", error=_err_obj(error_type, message), request_id=request_id)


def _yielded_error_event_dict(error_type, message, request_id="req_test_dict"):
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": request_id,
    }


def _yielded_message_start(input_tokens=100, cache_read=0, cache_creation=0):
    usage = _obj_event(
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        output_tokens=0,
    )
    msg = _obj_event(usage=usage, content=[])
    return _obj_event(type="message_start", message=msg)


def _yielded_content_block_delta_obj(text):
    delta = _obj_event(type="text_delta", text=text)
    return _obj_event(type="content_block_delta", delta=delta, index=0)


def _yielded_content_block_delta_dict(text):
    return {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": text},
        "index": 0,
    }


def _yielded_message_delta(output_tokens):
    usage = _obj_event(output_tokens=output_tokens)
    return _obj_event(type="message_delta", usage=usage, delta=_obj_event(stop_reason="end_turn"))


def _make_api_status_error(status_code, body=None):
    """Build an anthropic.APIStatusError. SDK's constructor wants a
    Response; we patch __init__ to skip the response-dependency."""
    from anthropic import APIStatusError

    exc = APIStatusError.__new__(APIStatusError)
    exc.status_code = status_code
    exc.body = body
    exc.message = (body or {}).get("error", {}).get("message", "(status)") if isinstance(body, dict) else "(status)"
    exc.request_id = "req_status_test"
    return exc


# ---------------------------------------------------------------------------
# Yielded error events — typed Exception per Bedrock-Error-Type
# ---------------------------------------------------------------------------


class TestYieldedErrorEventDecoding:
    def test_overloaded_yielded_event_triggers_overload_exception(self):
        import pytest

        from ai_router.cached_llm import BedrockOverloadError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("overloaded_error", "Overloaded")]
        )

        with pytest.raises(BedrockOverloadError) as exc_info:
            client._stream_once({"messages": []})

        assert exc_info.value.bedrock_error_type == "overloaded_error"
        assert exc_info.value.bedrock_message == "Overloaded"
        assert exc_info.value.model == "claude-sonnet-4-6"
        assert exc_info.value.request_id == "req_test_abc"

    def test_rate_limit_yielded_event_triggers_rate_limit_exception(self):
        import pytest

        from ai_router.cached_llm import BedrockRateLimitError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("rate_limit_error", "rate limited")]
        )
        with pytest.raises(BedrockRateLimitError):
            client._stream_once({"messages": []})

    def test_invalid_request_yielded_event(self):
        import pytest

        from ai_router.cached_llm import BedrockInvalidRequestError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("invalid_request_error", "bad input")]
        )
        with pytest.raises(BedrockInvalidRequestError):
            client._stream_once({"messages": []})

    def test_authentication_yielded_event(self):
        import pytest

        from ai_router.cached_llm import BedrockAuthenticationError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("authentication_error", "bad token")]
        )
        with pytest.raises(BedrockAuthenticationError):
            client._stream_once({"messages": []})

    def test_unknown_error_type_falls_back_to_unknown_stream_error(self):
        """Unbekannter error.type → BedrockUnknownStreamError, NICHT
        BedrockStreamOutOfOrderError (das ist nur für SDK-Anomalien
        OHNE dekodierbare Payload)."""
        import pytest

        from ai_router.cached_llm import BedrockUnknownStreamError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("mystery_unknown_error", "huh")]
        )
        with pytest.raises(BedrockUnknownStreamError) as exc_info:
            client._stream_once({"messages": []})
        assert exc_info.value.bedrock_error_type == "mystery_unknown_error"

    def test_dict_shaped_yielded_error_event(self):
        """Event als raw dict statt Pydantic-Object → decoded korrekt."""
        import pytest

        from ai_router.cached_llm import BedrockOverloadError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_dict("overloaded_error", "Overloaded-dict")]
        )
        with pytest.raises(BedrockOverloadError) as exc_info:
            client._stream_once({"messages": []})
        assert exc_info.value.bedrock_message == "Overloaded-dict"
        assert exc_info.value.request_id == "req_test_dict"


# ---------------------------------------------------------------------------
# APIStatusError — vor First-Event raise
# ---------------------------------------------------------------------------


class TestApiStatusErrorDecoding:
    def test_apistatuserror_with_overloaded_body(self):
        import pytest

        from ai_router.cached_llm import BedrockOverloadError

        client = _make_anthropic_client()
        api_exc = _make_api_status_error(
            status_code=529,
            body={"error": {"type": "overloaded_error", "message": "Overloaded"}},
        )
        client._client.messages.create.side_effect = api_exc

        with pytest.raises(BedrockOverloadError) as exc_info:
            client._stream_once({"messages": []})
        assert exc_info.value.bedrock_error_type == "overloaded_error"

    def test_apistatuserror_empty_body_status_529_falls_back_to_overload(self):
        """SDK kann APIStatusError mit body=None werfen → wir mappen
        Status-Code 529 (Anthropic-overload) → overloaded_error."""
        import pytest

        from ai_router.cached_llm import BedrockOverloadError

        client = _make_anthropic_client()
        api_exc = _make_api_status_error(status_code=529, body=None)
        client._client.messages.create.side_effect = api_exc

        with pytest.raises(BedrockOverloadError):
            client._stream_once({"messages": []})

    def test_apistatuserror_empty_body_status_429_falls_back_to_rate_limit(self):
        import pytest

        from ai_router.cached_llm import BedrockRateLimitError

        client = _make_anthropic_client()
        api_exc = _make_api_status_error(status_code=429, body=None)
        client._client.messages.create.side_effect = api_exc

        with pytest.raises(BedrockRateLimitError):
            client._stream_once({"messages": []})

    def test_apistatuserror_empty_body_status_503_falls_back_to_overload(self):
        import pytest

        from ai_router.cached_llm import BedrockOverloadError

        client = _make_anthropic_client()
        api_exc = _make_api_status_error(status_code=503, body=None)
        client._client.messages.create.side_effect = api_exc

        with pytest.raises(BedrockOverloadError):
            client._stream_once({"messages": []})

    def test_apistatuserror_empty_body_status_400_falls_back_to_invalid_request(self):
        import pytest

        from ai_router.cached_llm import BedrockInvalidRequestError

        client = _make_anthropic_client()
        api_exc = _make_api_status_error(status_code=400, body=None)
        client._client.messages.create.side_effect = api_exc

        with pytest.raises(BedrockInvalidRequestError):
            client._stream_once({"messages": []})

    def test_apistatuserror_empty_body_status_401_falls_back_to_auth(self):
        import pytest

        from ai_router.cached_llm import BedrockAuthenticationError

        client = _make_anthropic_client()
        api_exc = _make_api_status_error(status_code=401, body=None)
        client._client.messages.create.side_effect = api_exc

        with pytest.raises(BedrockAuthenticationError):
            client._stream_once({"messages": []})


# ---------------------------------------------------------------------------
# Content + Usage Aggregation
# ---------------------------------------------------------------------------


class TestStreamContentAndUsage:
    def test_text_chunks_accumulated_correctly(self):
        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [
                _yielded_message_start(input_tokens=100),
                _yielded_content_block_delta_obj("Hello "),
                _yielded_content_block_delta_obj("world"),
                _yielded_content_block_delta_obj("!"),
                _yielded_message_delta(output_tokens=3),
            ]
        )
        result = client._stream_once({"messages": []})
        assert result.content[0].text == "Hello world!"
        assert result.usage.output_tokens == 3
        assert result.usage.input_tokens == 100

    def test_usage_aggregated_from_message_start_and_delta(self):
        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [
                _yielded_message_start(input_tokens=500, cache_read=200, cache_creation=50),
                _yielded_content_block_delta_obj("x"),
                _yielded_message_delta(output_tokens=42),
            ]
        )
        result = client._stream_once({"messages": []})
        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 42
        assert result.usage.cache_read_input_tokens == 200
        assert result.usage.cache_creation_input_tokens == 50

    def test_dict_shaped_content_block_delta_decoded(self):
        """P2.1 — content_block_delta als raw dict; getattr(delta, ...)
        würde None liefern. _delta_type/_delta_text müssen dict-aware sein."""
        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [
                _yielded_message_start(input_tokens=10),
                _yielded_content_block_delta_dict("dict-text"),
                _yielded_message_delta(output_tokens=1),
            ]
        )
        result = client._stream_once({"messages": []})
        assert result.content[0].text == "dict-text"


# ---------------------------------------------------------------------------
# Retry-Schedule
# ---------------------------------------------------------------------------


class TestRetrySchedule:
    def test_overloaded_retried_4_attempts_with_exp_backoff(self):
        """4 Versuche → 3 Sleeps (5,15,45), kein Sleep nach letztem."""
        import pytest

        from ai_router.cached_llm import BedrockOverloadError

        client = _make_anthropic_client()
        # 4× overloaded_error
        client._client.messages.create.side_effect = [
            _IterableStream([_yielded_error_event_obj("overloaded_error", f"OL-{i}")]) for i in range(4)
        ]

        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockOverloadError) as exc_info:
            client._create_or_stream({"messages": []})

        assert client._client.messages.create.call_count == 4
        # Genau 3 Sleeps zwischen den 4 Versuchen
        assert sleep_mock.call_count == 3
        sleep_calls = [c.args[0] for c in sleep_mock.call_args_list]
        assert sleep_calls == [5, 15, 45]
        # Letzte Exception sichtbar
        assert exc_info.value.bedrock_error_type == "overloaded_error"

    def test_overloaded_succeeds_on_third_attempt(self):
        """2× overloaded → 1× success → 2 Sleeps (5,15) + valide Response."""
        client = _make_anthropic_client()
        success_stream = _IterableStream(
            [
                _yielded_message_start(input_tokens=10),
                _yielded_content_block_delta_obj("ok"),
                _yielded_message_delta(output_tokens=1),
            ]
        )
        client._client.messages.create.side_effect = [
            _IterableStream([_yielded_error_event_obj("overloaded_error", "OL")]),
            _IterableStream([_yielded_error_event_obj("overloaded_error", "OL")]),
            success_stream,
        ]

        with patch("time.sleep") as sleep_mock:
            result = client._create_or_stream({"messages": []})

        assert client._client.messages.create.call_count == 3
        sleep_calls = [c.args[0] for c in sleep_mock.call_args_list]
        assert sleep_calls == [5, 15]
        assert result.content[0].text == "ok"

    def test_invalid_request_does_not_retry(self):
        """Non-retryable type → sofortiger raise, sleep NIE gerufen."""
        import pytest

        from ai_router.cached_llm import BedrockInvalidRequestError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("invalid_request_error", "bad input")]
        )

        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockInvalidRequestError):
            client._create_or_stream({"messages": []})

        assert client._client.messages.create.call_count == 1
        sleep_mock.assert_not_called()

    def test_authentication_does_not_retry(self):
        import pytest

        from ai_router.cached_llm import BedrockAuthenticationError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("authentication_error", "bad token")]
        )
        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockAuthenticationError):
            client._create_or_stream({"messages": []})
        assert client._client.messages.create.call_count == 1
        sleep_mock.assert_not_called()

    def test_permission_does_not_retry(self):
        import pytest

        from ai_router.cached_llm import BedrockPermissionError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("permission_error", "denied")]
        )
        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockPermissionError):
            client._create_or_stream({"messages": []})
        sleep_mock.assert_not_called()

    def test_billing_does_not_retry(self):
        import pytest

        from ai_router.cached_llm import BedrockBillingError

        client = _make_anthropic_client()
        client._client.messages.create.return_value = _IterableStream(
            [_yielded_error_event_obj("billing_error", "exceeded")]
        )
        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockBillingError):
            client._create_or_stream({"messages": []})
        sleep_mock.assert_not_called()

    def test_api_error_is_retryable(self):
        """api_error (HTTP 500/502) → wird retried."""
        import pytest

        from ai_router.cached_llm import BedrockApiError

        client = _make_anthropic_client()
        client._client.messages.create.side_effect = [
            _IterableStream([_yielded_error_event_obj("api_error", f"5xx-{i}")]) for i in range(4)
        ]
        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockApiError):
            client._create_or_stream({"messages": []})
        assert client._client.messages.create.call_count == 4
        assert sleep_mock.call_count == 3
