"""Smoke tests for the W3-L3 moves: cached_llm + encoders live in ai_router.

`ai_agents/utils/cached_llm.py` is a deprecated shim that re-exports the
identical objects and emits a DeprecationWarning on import. The former
`scribe/encoders.py` shim was removed with the dj-rag-db extraction (W6) —
scribe imports `ai_router.encoders` directly.
"""

import importlib
import sys
import warnings


def _import_fresh_with_warnings(module_name):
    sys.modules.pop(module_name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(module_name)
    return caught


class TestCachedLlmMove:
    def test_cached_llm_importable_from_ai_router(self):
        module = importlib.import_module("ai_router.cached_llm")
        assert hasattr(module, "CachedAnthropicClient")
        assert hasattr(module, "CachedGeminiClient")
        assert hasattr(module, "CachedInvocationResult")
        assert hasattr(module, "get_cached_client")
        assert hasattr(module, "get_document_cache_key")

    def test_legacy_shim_reexports_identical_objects(self):
        import ai_agents.utils.cached_llm as legacy
        import ai_router.cached_llm as new

        assert legacy.CachedAnthropicClient is new.CachedAnthropicClient
        assert legacy.CachedGeminiClient is new.CachedGeminiClient
        assert legacy.CachedInvocationResult is new.CachedInvocationResult
        assert legacy.get_cached_client is new.get_cached_client
        assert legacy.ForcedToolUseError is new.ForcedToolUseError
        assert legacy.BedrockOverloadError is new.BedrockOverloadError

    def test_legacy_shim_emits_deprecation_warning(self):
        caught = _import_fresh_with_warnings("ai_agents.utils.cached_llm")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_ai_agents_utils_package_import_does_not_warn(self):
        """ai_agents.utils re-exports from ai_router directly — no shim warning."""
        sys.modules.pop("ai_agents.utils.cached_llm", None)
        caught = _import_fresh_with_warnings("ai_agents.utils")
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


class TestEncodersMove:
    def test_encoders_importable_from_ai_router(self):
        module = importlib.import_module("ai_router.encoders")
        assert hasattr(module, "BaseEncoder")
        assert hasattr(module, "AzureOpenAIEncoder")


class TestLLMLogProjectFK:
    def test_project_fk_targets_project_model_by_default(self):
        """AI_ROUTER_PROJECT_MODEL default keeps the FK on project.Project."""
        from ai_router.models import LLMLog

        field = LLMLog._meta.get_field("project")
        assert field.remote_field.model._meta.label == "project.Project"
