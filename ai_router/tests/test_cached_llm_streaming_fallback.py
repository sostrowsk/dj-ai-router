"""F4 — Streaming-Fallback bei high max_tokens.

Anthropic-SDK rejects non-streaming `messages.create()` wenn die
geschätzte Laufzeit > 10 Min. Threshold: `max_tokens * 3600/128000 > 600`
→ max_tokens > 21333. Mit max_tokens=96000 (F2-A) triggert das ALLE
non-stream Pfade in CachedAnthropicClient.

Live-Run 2026-05-09 zeigte: invoke_with_pdf_cache crashed sofort mit
ValueError "Streaming is required for operations that may take longer
than 10 minutes" — Doc-214 + Doc-183 fielen mit 0.4s aus.

Lösung: invoke / invoke_with_cache / invoke_with_pdf_cache /
invoke_raw_cached müssen bei high max_tokens auf `messages.stream()`
umschalten, Chunks sammeln und ein response-like Object liefern.
External-Vertrag bleibt identisch: tuple (CachedInvocationResult,
parsed_output).

Diese Suite verifiziert:
1. invoke_with_pdf_cache ruft messages.stream() (NICHT create()) bei
   max_tokens=96000.
2. Gestreamte Chunks werden konkateniert und als content geliefert.
3. Usage-Token-Counts kommen aus stream.get_final_message().
4. parsed_output via PydanticOutputParser läuft trotzdem (auf dem
   konkatenierten content).
5. Bei max_tokens unter Threshold läuft create() weiter (kein
   unnötiges Streaming-Overhead).
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
    return client


def _stub_stream_events(content_chunks, *, input_tokens=10, output_tokens=42):
    """Plan-S2 — Iterable of RawMessageStreamEvent-shaped objects for the
    new manual stream path (messages.create(stream=True))."""

    def _obj(**kwargs):
        m = MagicMock(spec=list(kwargs.keys()))
        for k, v in kwargs.items():
            setattr(m, k, v)
        return m

    start_usage = _obj(
        input_tokens=input_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        output_tokens=0,
    )
    events = [_obj(type="message_start", message=_obj(usage=start_usage, content=[]))]
    for chunk in content_chunks:
        delta = _obj(type="text_delta", text=chunk)
        events.append(_obj(type="content_block_delta", delta=delta, index=0))
    events.append(_obj(type="message_delta", usage=_obj(output_tokens=output_tokens), delta=_obj()))

    class _Stream:
        def __iter__(self):
            for e in events:
                yield e

        def close(self):
            pass

    return _Stream()


class TestInvokeWithPdfCacheUsesStreamingWhenMaxTokensHigh:
    def test_high_max_tokens_uses_streaming_create_not_blocking_create(self):
        """Doc-214/183-Reproduzent: max_tokens=96000 darf NICHT
        non-streaming messages.create() rufen (würde SDK-ValueError werfen).
        Plan-S2: streaming geht jetzt über messages.create(stream=True)."""
        client = _make_anthropic_client(max_tokens=96000)
        client._client.messages.create.return_value = _stub_stream_events(["{", '"x": 1', "}"])

        client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF-1.4 fake",
            system_prompt="sys",
            extraction_prompt="extract",
        )

        client._client.messages.create.assert_called_once()
        assert client._client.messages.create.call_args.kwargs.get("stream") is True
        client._client.messages.stream.assert_not_called()

    def test_streaming_concatenates_chunks_into_content(self):
        client = _make_anthropic_client(max_tokens=96000)
        chunks = ["chunk-A ", "chunk-B ", "chunk-C"]
        client._client.messages.create.return_value = _stub_stream_events(chunks)

        result, _ = client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF",
            system_prompt="sys",
            extraction_prompt="extract",
        )
        assert result.content == "chunk-A chunk-B chunk-C"

    def test_streaming_propagates_usage_from_final_message(self):
        client = _make_anthropic_client(max_tokens=96000)
        client._client.messages.create.return_value = _stub_stream_events(["{}"], input_tokens=123, output_tokens=456)

        result, _ = client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF",
            system_prompt="sys",
            extraction_prompt="extract",
        )
        assert result.input_tokens == 123
        assert result.output_tokens == 456

    def test_streaming_runs_pydantic_parser_on_concatenated_content(self):
        """parsed_output wird auf dem GANZEN konkatenierten content
        gemacht (nicht pro chunk)."""
        from pydantic import BaseModel

        class _Position(BaseModel):
            position_name: str
            aggregates: dict
            standard_category: str

        class _PositionResult(BaseModel):
            positions: list[_Position]

        client = _make_anthropic_client(max_tokens=96000)
        # Valid _PositionResult in 3 chunks.
        chunks = [
            '{"positions":[',
            '{"position_name":"X","aggregates":{"ebitda":"minus"},',
            '"standard_category":"Materialaufwand"}]}',
        ]
        client._client.messages.create.return_value = _stub_stream_events(chunks)

        _, parsed = client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF",
            system_prompt="sys",
            extraction_prompt="extract",
            output_schema=_PositionResult,
        )
        assert parsed is not None
        assert len(parsed.positions) == 1
        assert parsed.positions[0].position_name == "X"


class TestInvokeWithPdfCacheLowMaxTokens:
    def test_low_max_tokens_still_uses_create_not_stream(self):
        """max_tokens unter Threshold (~21k): kein Streaming-Overhead.
        Backward-Compat: existing low-max_tokens callsites bleiben
        unverändert."""
        client = _make_anthropic_client(max_tokens=8192)
        fake_resp = MagicMock()
        fake_resp.content = [MagicMock(text="{}")]
        fake_resp.usage = MagicMock(
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
        )
        client._client.messages.create.return_value = fake_resp

        client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF",
            system_prompt="sys",
            extraction_prompt="extract",
        )

        client._client.messages.create.assert_called_once()
        client._client.messages.stream.assert_not_called()


class TestInvokeAndInvokeWithCacheAlsoStream:
    """Konsistenz: alle non-stream Methoden, die self.max_tokens nutzen,
    bekommen den Streaming-Fallback."""

    def test_invoke_streams_when_high_max_tokens(self):
        client = _make_anthropic_client(max_tokens=96000)
        client._client.messages.create.return_value = _stub_stream_events(["ok"])
        client.invoke(system_prompt="sys", user_prompt="u")
        client._client.messages.create.assert_called_once()
        assert client._client.messages.create.call_args.kwargs.get("stream") is True
        client._client.messages.stream.assert_not_called()

    def test_invoke_with_cache_streams_when_high_max_tokens(self):
        client = _make_anthropic_client(max_tokens=96000)
        client._client.messages.create.return_value = _stub_stream_events(["ok"])
        client.invoke_with_cache(
            document_content="doc",
            document_name="d.pdf",
            extraction_prompt="extract",
        )
        client._client.messages.create.assert_called_once()
        assert client._client.messages.create.call_args.kwargs.get("stream") is True
        client._client.messages.stream.assert_not_called()

    def test_invoke_raw_cached_streams_when_high_max_tokens(self):
        client = _make_anthropic_client(max_tokens=96000)
        client._client.messages.create.return_value = _stub_stream_events(["ok"])
        client.invoke_raw_cached(system_text="cached", user_prompt="u")
        client._client.messages.create.assert_called_once()
        assert client._client.messages.create.call_args.kwargs.get("stream") is True
        client._client.messages.stream.assert_not_called()
