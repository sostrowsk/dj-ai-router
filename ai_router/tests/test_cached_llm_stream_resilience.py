"""S1-Reste — Detection-Helper + SDK-RuntimeError-Fallback (S2-Pfad).

Plan-S2 hat den Legacy-Pfad `messages.stream(...).text_stream` durch
manuelles `messages.create(stream=True)` + `_iter_stream_events` ersetzt.
Die SDK `accumulate_event` RuntimeError aus
anthropic/lib/streaming/_messages.py:454 ist auf dem neuen Pfad nicht
mehr regulär erreichbar — wir nutzen text_stream nicht.

Was bleibt:
1. `_is_sdk_stream_out_of_order_error` — Pattern-Matcher; hilft falls
   irgendein SDK-Update doch wieder mid-iteration ein RuntimeError mit
   diesem Suffix wirft.
2. `_create_or_stream` S1-Fallback: wenn `_stream_once` einen
   RuntimeError mit dem Suffix wirft → wird wie ein retryable api_error
   behandelt; nach _MAX_ATTEMPTS-failed → BedrockStreamOutOfOrderError.

Defensive Tests (kein Live-Pfad, aber Schutz gegen SDK-Regressionen):
"""

from unittest.mock import MagicMock, patch

_SDK_ERROR_MSG = 'Unexpected event order, got error before "message_start"'


def _make_anthropic_client(max_tokens: int = 96000):
    from ai_router.cached_llm import CachedAnthropicClient

    with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
        client = CachedAnthropicClient(model="claude-sonnet-4-6", max_tokens=max_tokens)
    client._client = MagicMock()
    client._is_bedrock = True
    client._is_vertex = False
    client._supports_temp = False
    client._model_alias = "claude-sonnet-4-6"
    return client


class TestSdkStreamOutOfOrderDetection:
    def test_detection_helper_matches_sdk_message(self):
        from ai_router.cached_llm import _is_sdk_stream_out_of_order_error

        exc = RuntimeError(_SDK_ERROR_MSG)
        assert _is_sdk_stream_out_of_order_error(exc) is True

    def test_detection_helper_rejects_other_runtime(self):
        from ai_router.cached_llm import _is_sdk_stream_out_of_order_error

        exc = RuntimeError("something completely unrelated")
        assert _is_sdk_stream_out_of_order_error(exc) is False

    def test_detection_helper_rejects_value_error(self):
        from ai_router.cached_llm import _is_sdk_stream_out_of_order_error

        exc = ValueError(_SDK_ERROR_MSG)
        assert _is_sdk_stream_out_of_order_error(exc) is False


class TestS1FallbackOnRuntimeErrorFromIteration:
    """Falls SDK doch noch einen RuntimeError mid-iteration wirft (z.B.
    nach SDK-Upgrade), behandeln wir das wie retryable api_error."""

    def test_sdk_runtime_error_retried_then_raises_out_of_order(self):
        """4× RuntimeError → 4 Versuche, 3 Sleeps, final
        BedrockStreamOutOfOrderError."""
        import pytest

        from ai_router.cached_llm import BedrockStreamOutOfOrderError

        client = _make_anthropic_client()

        # Simuliere: messages.create gibt ein iterable zurück, dessen __iter__
        # einen RuntimeError mit dem SDK-Pattern wirft.
        def _raising_stream():
            class _S:
                def __iter__(self):
                    raise RuntimeError(_SDK_ERROR_MSG)

                def close(self):
                    pass

            return _S()

        client._client.messages.create.side_effect = [_raising_stream() for _ in range(4)]

        with patch("time.sleep") as sleep_mock, pytest.raises(BedrockStreamOutOfOrderError):
            client._create_or_stream({"messages": []})

        assert client._client.messages.create.call_count == 4
        assert sleep_mock.call_count == 3
        sleep_calls = [c.args[0] for c in sleep_mock.call_args_list]
        assert sleep_calls == [5, 15, 45]

    def test_unrelated_runtime_error_bubbles_without_retry(self):
        import pytest

        client = _make_anthropic_client()

        def _bad_stream():
            class _S:
                def __iter__(self):
                    raise RuntimeError("totally different bug")

                def close(self):
                    pass

            return _S()

        client._client.messages.create.return_value = _bad_stream()

        with patch("time.sleep") as sleep_mock, pytest.raises(RuntimeError) as exc_info:
            client._create_or_stream({"messages": []})

        assert "totally different bug" in str(exc_info.value)
        assert client._client.messages.create.call_count == 1
        sleep_mock.assert_not_called()


class TestNonStreamingPathUnchanged:
    def test_non_streaming_path_skips_retry_logic(self):
        """max_tokens unter Threshold (21000) → messages.create wird
        ohne stream=True gerufen. Retry-Wrapper darf nicht aktiv sein."""
        client = _make_anthropic_client(max_tokens=8000)
        fake_response = MagicMock()
        client._client.messages.create.return_value = fake_response

        result = client._create_or_stream({"messages": [], "max_tokens": 8000})

        assert result is fake_response
        client._client.messages.create.assert_called_once()
        # Non-stream path passes no `stream=True` kwarg.
        assert client._client.messages.create.call_args.kwargs.get("stream") is None
        client._client.messages.stream.assert_not_called()
