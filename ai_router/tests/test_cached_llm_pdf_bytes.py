"""Phase 5 — `invoke_with_pdf_cache(pdf_bytes=...)` API.

The Phase-4 slicer produces PDF bytes in-memory. Both client classes
(CachedAnthropicClient, CachedGeminiClient) must accept those bytes
directly without round-tripping through the filesystem — slicing per
extraction call would otherwise force a temp-file write per call.

Contract:
- pdf_bytes XOR pdf_path (exactly one of them must be set).
- Behavior is byte-equivalent: same bytes via either path produce the
  same downstream LLM payload.
- Both client classes share the same signature so the caller
  (extract_document_data_task) is provider-agnostic.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_anthropic_response():
    response = MagicMock()
    response.content = [MagicMock(text="{}")]
    response.content[0].text = "{}"
    response.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def fake_gemini_response():
    response = MagicMock()
    response.text = "{}"
    response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
    return response


# ---------------------------------------------------------------------------
# CachedAnthropicClient
# ---------------------------------------------------------------------------


class TestCachedAnthropicClientPdfBytes:
    def _make_client(self, fake_response):
        """max_tokens=8000 → unter F4-Threshold → non-streaming create-Pfad
        bleibt aktiv (existing pdf_bytes-Tests inspizieren create.call_args)."""
        from ai_router.cached_llm import CachedAnthropicClient

        with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
            client = CachedAnthropicClient(model="claude-sonnet-4-6", max_tokens=8000)
        client._client = MagicMock()
        client._client.messages.create.return_value = fake_response
        client._is_bedrock = False
        return client

    def test_pdf_bytes_argument_accepted(self, fake_anthropic_response):
        client = self._make_client(fake_anthropic_response)
        client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF-1.4 fake bytes",
            system_prompt="sys",
            extraction_prompt="extract",
        )
        assert client._client.messages.create.called

    def test_pdf_bytes_and_pdf_path_both_none_raises(self, fake_anthropic_response):
        client = self._make_client(fake_anthropic_response)
        with pytest.raises(ValueError, match="pdf_path or pdf_bytes"):
            client.invoke_with_pdf_cache(
                system_prompt="sys",
                extraction_prompt="extract",
            )

    def test_pdf_bytes_and_pdf_path_both_set_raises(self, fake_anthropic_response, tmp_path):
        """XOR contract: silently preferring one over the other would
        hide stale-PDF bugs (caller passes a path AND stale slice
        bytes; the wrong context wins)."""
        client = self._make_client(fake_anthropic_response)
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 path")
        with pytest.raises(ValueError, match="not both"):
            client.invoke_with_pdf_cache(
                pdf_path=str(path),
                pdf_bytes=b"%PDF-1.4 bytes",
                system_prompt="sys",
                extraction_prompt="extract",
            )

    def test_pdf_bytes_payload_matches_pdf_path(self, fake_anthropic_response, tmp_path):
        """Same bytes via either entry point must produce identical
        message payload — the LLM should not see any difference."""
        import base64

        pdf_data = b"%PDF-1.4 some content"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(pdf_data)

        client_a = self._make_client(fake_anthropic_response)
        client_a.invoke_with_pdf_cache(
            pdf_path=str(pdf_path),
            system_prompt="sys",
            extraction_prompt="extract",
        )
        kwargs_a = client_a._client.messages.create.call_args.kwargs

        client_b = self._make_client(fake_anthropic_response)
        client_b.invoke_with_pdf_cache(
            pdf_bytes=pdf_data,
            system_prompt="sys",
            extraction_prompt="extract",
        )
        kwargs_b = client_b._client.messages.create.call_args.kwargs

        # Document block (first content item) must contain identical
        # base64-encoded data.
        doc_a = kwargs_a["messages"][0]["content"][0]
        doc_b = kwargs_b["messages"][0]["content"][0]
        expected_b64 = base64.standard_b64encode(pdf_data).decode("utf-8")
        assert doc_a["source"]["data"] == expected_b64
        assert doc_b["source"]["data"] == expected_b64
        assert doc_a == doc_b


# ---------------------------------------------------------------------------
# CachedGeminiClient
# ---------------------------------------------------------------------------


class TestCachedGeminiClientPdfBytes:
    def _make_client(self, fake_response):
        from ai_router.cached_llm import CachedGeminiClient

        with patch.object(CachedGeminiClient, "_setup_client", lambda self: None):
            client = CachedGeminiClient(model="gemini-2.5-flash")
        client._client = MagicMock()
        client._client.models.generate_content.return_value = fake_response
        return client

    def test_pdf_bytes_argument_accepted(self, fake_gemini_response):
        client = self._make_client(fake_gemini_response)
        client.invoke_with_pdf_cache(
            pdf_bytes=b"%PDF-1.4 fake bytes",
            system_prompt="sys",
            extraction_prompt="extract",
        )
        assert client._client.models.generate_content.called

    def test_pdf_bytes_and_pdf_path_both_none_raises(self, fake_gemini_response):
        client = self._make_client(fake_gemini_response)
        with pytest.raises(ValueError, match="pdf_path or pdf_bytes"):
            client.invoke_with_pdf_cache(
                system_prompt="sys",
                extraction_prompt="extract",
            )

    def test_pdf_bytes_and_pdf_path_both_set_raises(self, fake_gemini_response, tmp_path):
        """XOR contract symmetric across both providers."""
        client = self._make_client(fake_gemini_response)
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 path")
        with pytest.raises(ValueError, match="not both"):
            client.invoke_with_pdf_cache(
                pdf_path=str(path),
                pdf_bytes=b"%PDF-1.4 bytes",
                system_prompt="sys",
                extraction_prompt="extract",
            )

    def test_pdf_bytes_payload_matches_pdf_path(self, fake_gemini_response, tmp_path):
        pdf_data = b"%PDF-1.4 gemini content"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(pdf_data)

        client_a = self._make_client(fake_gemini_response)
        client_a.invoke_with_pdf_cache(
            pdf_path=str(pdf_path),
            system_prompt="sys",
            extraction_prompt="extract",
        )
        contents_a = client_a._client.models.generate_content.call_args.kwargs["contents"]

        client_b = self._make_client(fake_gemini_response)
        client_b.invoke_with_pdf_cache(
            pdf_bytes=pdf_data,
            system_prompt="sys",
            extraction_prompt="extract",
        )
        contents_b = client_b._client.models.generate_content.call_args.kwargs["contents"]

        # Both calls produced the same Part.from_bytes payload.
        # types.Part.from_bytes wraps in an inline_data dict; we
        # verify the raw bytes via the part's inline_data.data.
        part_a = contents_a[0]
        part_b = contents_b[0]
        assert getattr(part_a, "inline_data", None) is not None
        assert part_a.inline_data.data == pdf_data
        assert part_b.inline_data.data == pdf_data
