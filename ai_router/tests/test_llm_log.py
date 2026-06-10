"""
Regression tests for LLMLog after move from ai_agents to ai_router.

Ensures the model is importable, writable, and queryable from its new location.
"""

import pytest

from ai_router.models import LLMLog


@pytest.mark.django_db
class TestLLMLogModel:
    """Verify LLMLog works correctly after migration to ai_router."""

    def test_import_from_ai_router(self):
        """LLMLog is importable from ai_router.models."""
        assert LLMLog is not None
        assert LLMLog._meta.app_label == "ai_router"

    def test_create_and_query(self):
        """LLMLog can be created and queried."""
        log = LLMLog.objects.create(
            agent_name="test_agent",
            model="test-model",
            system_prompt="system",
            user_prompt="user",
            status=LLMLog.Status.SUCCESS,
            output="response",
            duration_ms=100,
            input_tokens=50,
            output_tokens=25,
        )
        assert log.pk is not None
        assert LLMLog.objects.filter(pk=log.pk).exists()
        assert log.is_success
        assert not log.is_error

    def test_status_choices(self):
        """Status choices are accessible."""
        assert LLMLog.Status.PENDING == "pending"
        assert LLMLog.Status.SUCCESS == "success"
        assert LLMLog.Status.ERROR == "error"

    def test_table_name(self):
        """Table is ai_router_llmlog (not ai_agents_llmlog)."""
        assert LLMLog._meta.db_table == "ai_router_llmlog"

    def test_not_importable_from_ai_agents(self):
        """LLMLog is no longer exported from ai_agents.models."""
        from ai_agents import models as ai_agents_models

        assert not hasattr(ai_agents_models, "LLMLog")
