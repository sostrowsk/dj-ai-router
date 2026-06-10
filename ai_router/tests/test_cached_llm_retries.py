"""F3 — Bedrock 5xx-Resilience.

Anthropic-SDK-Default ist 2 Retries (3 Attempts). Bedrock eu-central-1
zeigt regelmäßig kurze 5xx-Spitzen, die mit 3 Attempts nicht abgedeckt
sind (siehe Doc 181 Live-Run 2026-05-09: 3× 503 → komplette Extraction
gescheitert). Setting `ANTHROPIC_SDK_MAX_RETRIES` (Default 8) erhöht
SDK-Retries client-side.

Diese Tests verifizieren:
1. AnthropicBedrock wird mit max_retries=settings.ANTHROPIC_SDK_MAX_RETRIES
   konstruiert.
2. AnthropicVertex symmetrisch.
3. override_settings funktioniert (runtime-read, nicht Default-Argument-
   Evaluation beim Import).
"""

from unittest.mock import patch

from django.test import override_settings


class TestBedrockClientMaxRetries:
    def test_bedrock_client_constructed_with_max_retries_from_settings(self):
        """Bedrock-Pfad: kein GCP_PROJECT_ID, _is_bedrock=True. Constructor
        muss max_retries=8 (default Setting) bekommen."""
        from ai_router.cached_llm import CachedAnthropicClient

        with patch("anthropic.AnthropicBedrock") as mock_ctor, override_settings(
            GCP_PROJECT_ID="", ANTHROPIC_SDK_MAX_RETRIES=8
        ):
            CachedAnthropicClient(model="claude-sonnet-4-6")

        assert mock_ctor.called
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs.get("max_retries") == 8, f"AnthropicBedrock muss max_retries=8 sehen, hat: {kwargs}"

    def test_max_retries_overridable_via_settings(self):
        """override_settings im Test → runtime-read (nicht Import-Default-
        Evaluation)."""
        from ai_router.cached_llm import CachedAnthropicClient

        with patch("anthropic.AnthropicBedrock") as mock_ctor, override_settings(
            GCP_PROJECT_ID="", ANTHROPIC_SDK_MAX_RETRIES=5
        ):
            CachedAnthropicClient(model="claude-sonnet-4-6")
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs.get("max_retries") == 5


class TestVertexClientMaxRetries:
    def test_vertex_client_constructed_with_max_retries_from_settings(self):
        """Vertex-Pfad: GCP_PROJECT_ID gesetzt → AnthropicVertex statt
        AnthropicBedrock. Symmetrisch max_retries=8."""
        from ai_router.cached_llm import CachedAnthropicClient

        with patch("anthropic.AnthropicVertex") as mock_ctor, override_settings(
            GCP_PROJECT_ID="some-project",
            GCP_REGION="europe-west4",
            ANTHROPIC_SDK_MAX_RETRIES=8,
        ):
            CachedAnthropicClient(model="claude-sonnet-4-6")

        assert mock_ctor.called
        kwargs = mock_ctor.call_args.kwargs
        assert kwargs.get("max_retries") == 8
