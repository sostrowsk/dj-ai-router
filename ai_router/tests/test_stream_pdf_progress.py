"""Tests fuer Progress-Logging in CachedAnthropicClient.stream_pdf.

Bei grossen PDF-Extractions (mehrere Minuten) soll alle ~5000 Zeichen
ein Log-Eintrag erscheinen — sonst wirkt es als haenge der Prozess.
"""

import logging
from unittest.mock import MagicMock

from ai_router.cached_llm import CachedAnthropicClient


def _obj(**kwargs):
    m = MagicMock(spec=list(kwargs.keys()))
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


class _FakeStream:
    """Plan-S2 — stream_pdf jetzt manual event iteration. Liefert
    iterable of RawMessageStreamEvent-shaped objects."""

    def __init__(self, chunks):
        usage_start = _obj(input_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=0)
        events = [_obj(type="message_start", message=_obj(usage=usage_start, content=[]))]
        for c in chunks:
            delta = _obj(type="text_delta", text=c)
            events.append(_obj(type="content_block_delta", delta=delta, index=0))
        events.append(_obj(type="message_delta", usage=_obj(output_tokens=50), delta=_obj()))
        self._events = events

    def __iter__(self):
        for e in self._events:
            yield e

    def close(self):
        pass


def _make_client():
    """Bypass __init__ (braucht Anthropic-Creds) — setze Attribute direkt."""
    client = CachedAnthropicClient.__new__(CachedAnthropicClient)
    client.model = "claude-sonnet-4-6"
    client._client = MagicMock()
    client._is_bedrock = True
    return client


class TestStreamPdfLogsProgress:
    def test_logs_every_5000_chars(self, tmp_path, caplog):
        client = _make_client()
        # 12000 chars in kleinen Chunks → erwartet: logs bei 5000, 10000
        chunks = ["x" * 100] * 120
        client._client.messages.create = MagicMock(return_value=_FakeStream(chunks))

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        with caplog.at_level(logging.INFO, logger="ai_router.cached_llm"):
            client.stream_pdf(str(pdf_path), prompt="extract")

        progress_logs = [r for r in caplog.records if "chars received" in r.getMessage()]
        assert len(progress_logs) >= 2, (
            f"Expected >=2 progress logs at 5k/10k chars, got {len(progress_logs)}: "
            f"{[r.getMessage() for r in progress_logs]}"
        )

    def test_no_logs_for_tiny_output(self, tmp_path, caplog):
        """< 5000 chars → kein Progress-Log noetig."""
        client = _make_client()
        client._client.messages.create = MagicMock(return_value=_FakeStream(["short"]))

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        with caplog.at_level(logging.INFO, logger="ai_router.cached_llm"):
            client.stream_pdf(str(pdf_path), prompt="extract")

        progress_logs = [r for r in caplog.records if "chars received" in r.getMessage()]
        assert len(progress_logs) == 0

    def test_returns_full_concatenated_content(self, tmp_path):
        """Progress-Logging darf Content-Assembly nicht beeinflussen."""
        client = _make_client()
        chunks = ["Hello ", "world ", "from ", "Claude"]
        client._client.messages.create = MagicMock(return_value=_FakeStream(chunks))

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")

        result = client.stream_pdf(str(pdf_path), prompt="extract")
        assert result.content == "Hello world from Claude"
