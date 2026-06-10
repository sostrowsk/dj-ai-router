"""F2-A — Anthropic-Beta-Header pro Provider.

Bedrock validates anthropic-beta-Flags und akzeptiert keine
Anthropic-API-only Tags. Live-Run 2026-05-09: Bedrock 400 "invalid
beta flag" wenn `extended-output-128k-2025-02-19` mitgeschickt wird.
Bedrock supports cache_control nativ (GA April 2025) — kein
Beta-Header notwendig. Bedrock claude-sonnet-4-6 native max_output
≈ 32000.

Diese Suite verifiziert:
1. Helper `_beta_headers_for_provider()` (Provider-State-basiert).
2. Alle 6 self.max_tokens-Callsites nutzen den Helper:
   invoke / stream / invoke_with_cache / invoke_with_pdf_cache /
   stream_pdf / invoke_raw_cached.
3. Vertex bekommt KEINE extra_headers.
4. Bedrock bekommt KEINE extra_headers (cache_control native, keine
   Anthropic-only Beta-Flags).
5. Direct-Anthropic bekommt nur den prompt-caching-Tag.
"""

from unittest.mock import MagicMock, patch

import pytest

_PROMPT_CACHING_TAG = "prompt-caching-2024-07-31"


def _make_anthropic_client(*, is_bedrock: bool, is_vertex: bool):
    """CachedAnthropicClient mit gemocktem _setup_client und manuell
    gesetzten Provider-State-Flags. max_tokens=8000 (unter F4-Threshold
    21000) → messages.create-Pfad bleibt aktiv, Header-Tests können
    `create.call_args` inspizieren. Für den Streaming-Pfad-Test siehe
    test_cached_llm_streaming_fallback.py."""
    from ai_router.cached_llm import CachedAnthropicClient

    with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
        client = CachedAnthropicClient(model="claude-sonnet-4-6", max_tokens=8000)
    client._client = MagicMock()
    client._is_bedrock = is_bedrock
    client._is_vertex = is_vertex
    client._supports_temp = False
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="{}")]
    fake_resp.usage = MagicMock(
        input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )

    # Plan-S2: messages.create now serves BOTH non-streaming (returns
    # response-like) and streaming (returns iterable of events) calls.
    # We dispatch by the `stream` kwarg so the same fixture covers both.
    def _create(**kwargs):
        if kwargs.get("stream"):

            def _events():
                evt = MagicMock(spec=["type", "delta"])
                evt.type = "content_block_delta"
                evt.delta = MagicMock(spec=["type", "text"])
                evt.delta.type = "text_delta"
                evt.delta.text = "chunk"
                yield evt

            class _S:
                def __iter__(self_inner):
                    return _events()

                def close(self_inner):
                    pass

            return _S()
        return fake_resp

    client._client.messages.create.side_effect = _create
    # messages.stream legacy stub stays untouched — Rev 2 callers don't
    # hit it; tests assert it is NOT called.
    return client


# ---------------------------------------------------------------------------
# Helper-Tests (Provider-State-basiert)
# ---------------------------------------------------------------------------


class TestBetaHeadersHelper:
    def test_bedrock_returns_none(self):
        """Bedrock akzeptiert keine Anthropic-API-only Beta-Tags
        (Live-Run 2026-05-09: 400 'invalid beta flag'). cache_control
        ist auf Bedrock seit April 2025 nativ supported."""
        client = _make_anthropic_client(is_bedrock=True, is_vertex=False)
        assert client._beta_headers_for_provider() is None

    def test_direct_anthropic_returns_prompt_caching(self):
        """Direct-Anthropic: prompt-caching-Tag (kein extended-output —
        claude-opus-4 hat native max_output)."""
        client = _make_anthropic_client(is_bedrock=False, is_vertex=False)
        headers = client._beta_headers_for_provider()
        assert headers == {"anthropic-beta": _PROMPT_CACHING_TAG}

    def test_vertex_returns_none(self):
        client = _make_anthropic_client(is_bedrock=False, is_vertex=True)
        assert client._beta_headers_for_provider() is None


# ---------------------------------------------------------------------------
# Per-Callsite-Tests
# ---------------------------------------------------------------------------


class TestBedrockCallsitesSendNoExtraHeaders:
    """Bedrock validates anthropic-beta-Flags und akzeptiert keine
    Anthropic-API-only Tags (Live-Run 400 'invalid beta flag'). Alle 6
    Callsites senden auf Bedrock KEINE extra_headers."""

    @pytest.mark.parametrize(
        "method_name,call,call_kind",
        [
            ("invoke", lambda c: c.invoke(system_prompt="sys", user_prompt="u"), "create"),
            (
                "invoke_with_cache",
                lambda c: c.invoke_with_cache(
                    document_content="doc", document_name="d.pdf", extraction_prompt="extract"
                ),
                "create",
            ),
            (
                "invoke_with_pdf_cache",
                lambda c: c.invoke_with_pdf_cache(
                    pdf_bytes=b"%PDF-1.4 fake", system_prompt="sys", extraction_prompt="extract"
                ),
                "create",
            ),
            ("invoke_raw_cached", lambda c: c.invoke_raw_cached(system_text="cached", user_prompt="u"), "create"),
            ("stream", lambda c: list(c.stream(system_prompt="sys", user_prompt="u")), "create"),
        ],
    )
    def test_bedrock_does_not_set_extra_headers(self, method_name, call, call_kind):
        """Plan-S2: streaming-Pfade gehen jetzt über messages.create(
        stream=True). Beide call_kind-Werte landen auf messages.create."""
        client = _make_anthropic_client(is_bedrock=True, is_vertex=False)
        call(client)
        kwargs = client._client.messages.create.call_args.kwargs
        assert "extra_headers" not in kwargs, (
            f"Bedrock {method_name}: Anthropic-only Beta-Tags würden 400 werfen. " f"kwargs: {list(kwargs)}"
        )

    def test_stream_pdf_does_not_set_extra_headers_on_bedrock(self, tmp_path):
        """Plan-S2: stream_pdf nutzt messages.create(stream=True)."""
        client = _make_anthropic_client(is_bedrock=True, is_vertex=False)
        pdf_file = tmp_path / "fake.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        client.stream_pdf(pdf_path=str(pdf_file), prompt="extract")
        kwargs = client._client.messages.create.call_args.kwargs
        assert "extra_headers" not in kwargs
        assert kwargs.get("stream") is True


class TestDirectAnthropicCallsitesSetPromptCachingTag:
    def test_invoke_with_pdf_cache_sets_prompt_caching_tag(self):
        client = _make_anthropic_client(is_bedrock=False, is_vertex=False)
        client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF-1.4 fake",
            system_prompt="sys",
            extraction_prompt="extract",
        )
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["extra_headers"]["anthropic-beta"] == _PROMPT_CACHING_TAG


class TestVertexCallsitesSetNoExtraHeaders:
    @pytest.mark.parametrize(
        "method_name,call",
        [
            ("invoke", lambda c: c.invoke(system_prompt="sys", user_prompt="u")),
            (
                "invoke_with_pdf_cache",
                lambda c: c.invoke_with_pdf_cache(pdf_bytes=b"%PDF", system_prompt="sys", extraction_prompt="e"),
            ),
            ("invoke_raw_cached", lambda c: c.invoke_raw_cached(system_text="cached", user_prompt="u")),
        ],
    )
    def test_vertex_does_not_set_extra_headers(self, method_name, call):
        client = _make_anthropic_client(is_bedrock=False, is_vertex=True)
        call(client)
        kwargs = client._client.messages.create.call_args.kwargs
        assert (
            "extra_headers" not in kwargs
        ), f"Vertex sollte KEINE extra_headers setzen ({method_name}), kwargs: {list(kwargs)}"
