"""F2-A — max_tokens Setting (runtime-read).

Doc 183 (277 Positionen) trunkated bei char 31969 mit
max_tokens=16384. User-Direktive: max_output 96000. Bedrock claude-
sonnet-4-6 supports 96k via Extended-Output-Beta-Header (siehe
test_cached_llm_extended_output_header.py).

Diese Suite verifiziert:
1. CachedAnthropicClient default max_tokens=96000 aus Setting.
2. override_settings funktioniert (runtime-read im __init__).
3. Caller-Kwarg überstimmt Setting.
4. CachedGeminiClient bleibt bei eigenem Setting (NICHT symmetrisch
   — Gemini-Modelle haben modell-abhängige Caps).
"""

from unittest.mock import patch

from django.test import override_settings


class TestAnthropicMaxTokens:
    def _make_client(self):
        from ai_router.cached_llm import CachedAnthropicClient

        with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
            return CachedAnthropicClient(model="claude-sonnet-4-6")

    def _make_client_with_kwarg(self, **kwargs):
        from ai_router.cached_llm import CachedAnthropicClient

        with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
            return CachedAnthropicClient(model="claude-sonnet-4-6", **kwargs)

    def test_anthropic_default_max_tokens_uses_setting(self):
        """Default-Setting (Bedrock-safe 32000) wird gelesen."""
        with override_settings(LLM_MAX_OUTPUT_TOKENS=32000):
            client = self._make_client()
        assert client.max_tokens == 32000

    def test_anthropic_max_tokens_overridable_via_settings(self):
        """Verifiziert runtime-read im __init__ — nicht Import-Default-
        Evaluation."""
        with override_settings(LLM_MAX_OUTPUT_TOKENS=24000):
            client = self._make_client()
        assert client.max_tokens == 24000

    def test_anthropic_max_tokens_kwarg_wins(self):
        """Caller-Kwarg überstimmt Setting."""
        with override_settings(LLM_MAX_OUTPUT_TOKENS=96000):
            client = self._make_client_with_kwarg(max_tokens=8000)
        assert client.max_tokens == 8000


class TestGeminiMaxTokens:
    def _make_client(self):
        from ai_router.cached_llm import CachedGeminiClient

        with patch.object(CachedGeminiClient, "_setup_client", lambda self: None):
            return CachedGeminiClient(model="gemini-3.1-pro-preview")

    def test_gemini_default_max_tokens_uses_own_setting(self):
        """Gemini bekommt eigenes Setting, NICHT symmetrisch zu
        Anthropic 96000 (Gemini-Modelle haben modell-abhängige Caps)."""
        with override_settings(GEMINI_MAX_OUTPUT_TOKENS=16384, LLM_MAX_OUTPUT_TOKENS=96000):
            client = self._make_client()
        assert client.max_tokens == 16384, (
            "Gemini darf NICHT auf 96000 hochgesetzt werden — eigene Setting "
            "GEMINI_MAX_OUTPUT_TOKENS wegen modell-abhängiger Caps."
        )

    def test_gemini_max_tokens_overridable_via_own_setting(self):
        with override_settings(GEMINI_MAX_OUTPUT_TOKENS=32000):
            client = self._make_client()
        assert client.max_tokens == 32000


class TestMaxTokensPassedToCreateCall:
    def test_max_tokens_passed_to_messages_create(self):
        """`max_tokens` muss beim Anthropic-Call ankommen. Mit
        max_tokens=8000 läuft der non-streaming-Pfad → create.call_args."""
        from unittest.mock import MagicMock

        from ai_router.cached_llm import CachedAnthropicClient

        # max_tokens=8000 → unter F4-Threshold (21000) → create-Pfad.
        with patch.object(CachedAnthropicClient, "_setup_client", lambda self: None):
            client = CachedAnthropicClient(model="claude-sonnet-4-6", max_tokens=8000)
        client._client = MagicMock()
        fake_resp = MagicMock()
        fake_resp.content = [MagicMock(text="{}")]
        fake_resp.usage = MagicMock(
            input_tokens=1, output_tokens=1, cache_creation_input_tokens=0, cache_read_input_tokens=0
        )
        client._client.messages.create.return_value = fake_resp
        client._is_bedrock = True
        client._is_vertex = False
        client._supports_temp = False

        with override_settings(GCP_PROJECT_ID=""):
            client.invoke_with_pdf_cache(
                pdf_bytes=b"%PDF-1.4 fake",
                system_prompt="sys",
                extraction_prompt="extract",
            )

        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 8000
