"""Phase B5 — stream() captures token usage from streaming events.

``CachedAnthropicClient.stream()`` verwarf bisher die usage-Daten aus
``message_start``/``message_delta``-Events — ``LLMLog.input_tokens`` blieb
im Chat None. Jetzt akkumuliert stream() die usage in
``client.last_stream_usage`` (additives Attribut, andere Caller unberuehrt).

Beide Event-Shapes werden getestet: Objekt-Attribute (direct Anthropic SDK)
und dicts (Bedrock raw-mode).
"""

from types import SimpleNamespace
from unittest.mock import patch


def _make_client():
    from ai_router.cached_llm import CachedAnthropicClient

    with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
        client = CachedAnthropicClient(model="claude-sonnet-4-6", max_tokens=4096)
    client._is_bedrock = True
    client._is_vertex = False
    client._supports_temp = False
    client._model_alias = "claude-sonnet-4-6"
    return client


def _object_events():
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=17,
                    cache_creation_input_tokens=3,
                    cache_read_input_tokens=5,
                )
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hallo "),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Welt"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=42),
        ),
    ]


def _dict_events():
    return [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 17,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 5,
                }
            },
        },
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hallo "}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Welt"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 42}},
    ]


def _run_stream(client, events):
    client._iter_stream_events = lambda create_kwargs: iter(events)
    return list(client.stream("system", "frage"))


class TestStreamUsageCapture:
    def test_usage_is_none_before_first_stream(self):
        client = _make_client()
        assert client.last_stream_usage is None

    def test_stream_captures_usage_from_object_events(self):
        client = _make_client()
        chunks = _run_stream(client, _object_events())

        assert chunks == ["Hallo ", "Welt"]
        usage = client.last_stream_usage
        assert usage is not None
        assert usage.input_tokens == 17
        assert usage.output_tokens == 42
        assert usage.cache_creation_input_tokens == 3
        assert usage.cache_read_input_tokens == 5

    def test_stream_captures_usage_from_dict_events(self):
        """Bedrock raw-mode liefert dict-Events statt SDK-Objekte."""
        client = _make_client()
        chunks = _run_stream(client, _dict_events())

        assert chunks == ["Hallo ", "Welt"]
        usage = client.last_stream_usage
        assert usage is not None
        assert usage.input_tokens == 17
        assert usage.output_tokens == 42
        assert usage.cache_creation_input_tokens == 3
        assert usage.cache_read_input_tokens == 5

    def test_stream_without_usage_events_leaves_usage_none(self):
        client = _make_client()
        events = [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="nur Text"),
            ),
        ]
        chunks = _run_stream(client, events)

        assert chunks == ["nur Text"]
        assert client.last_stream_usage is None

    def test_stream_resets_usage_from_previous_call(self):
        """Eine zweite stream()-Iteration ohne usage darf keine stale Werte
        des vorherigen Calls liefern."""
        client = _make_client()
        _run_stream(client, _object_events())
        assert client.last_stream_usage is not None

        chunks = _run_stream(
            client,
            [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="zweiter"),
                )
            ],
        )
        assert chunks == ["zweiter"]
        assert client.last_stream_usage is None

    def test_partial_usage_defaults_missing_fields_to_zero(self):
        """Nur message_start mit input_tokens (Stream bricht vor message_delta
        ab) → output_tokens faellt auf 0 zurueck statt zu fehlen."""
        client = _make_client()
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
        ]
        chunks = _run_stream(client, events)

        assert chunks == ["x"]
        usage = client.last_stream_usage
        assert usage.input_tokens == 9
        assert usage.output_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0
