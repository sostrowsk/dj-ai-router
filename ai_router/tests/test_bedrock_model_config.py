"""BEDROCK_MODEL_CONFIG — Modell-Registry (model_id + supports_temp).

Opus 4.7/4.8 deprecaten den `temperature`-Parameter auf Bedrock (sonst
ValidationException); daher supports_temp=False — der Client poppt temperature
aus den create_kwargs.
"""

from ai_router.bedrock_client import BEDROCK_MODEL_CONFIG


def test_opus_4_8_registered_as_temp_unsupported():
    cfg = BEDROCK_MODEL_CONFIG.get("claude-opus-4-8")
    assert cfg is not None, "claude-opus-4-8 fehlt in BEDROCK_MODEL_CONFIG"
    assert cfg["model_id"] == "eu.anthropic.claude-opus-4-8"
    assert cfg["supports_temp"] is False
