"""Regression test for the dead ``LLMLog.retry_count`` field.

``retry_count`` was declared on the model and surfaced in the admin, but never
written anywhere in production code. Dead fields confuse the admin (always shows
0) and waste a column. This pins that the field is gone.
"""

from ai_router.models import LLMLog


def test_llm_log_has_no_dead_retry_count_field():
    field_names = {f.name for f in LLMLog._meta.get_fields()}
    assert "retry_count" not in field_names
