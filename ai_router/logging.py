"""
Central LLM call logging utility.

Provides context managers for logging every LLM invocation to the database.
"""

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

from asgiref.sync import sync_to_async

from ai_router.models import LLMLog

logger = logging.getLogger(__name__)


def _sanitize_int_fields(log):
    """Ensure token fields are int or None before DB save."""
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        val = getattr(log, field, None)
        if val is not None and not isinstance(val, int):
            setattr(log, field, None)


@contextmanager
def llm_log(agent_name, model, system_prompt="", user_prompt="", project=None, user=None):
    """Context manager for logging LLM calls.

    Creates a PENDING LLMLog entry, yields it, and finalizes on exit.
    The caller sets output, tokens, etc. on the yielded log object.
    Degrades gracefully if DB is unavailable (e.g. in unit tests).
    """
    try:
        log = LLMLog.objects.create(
            agent_name=agent_name,
            model=model,
            system_prompt=(system_prompt or "")[:10000],
            user_prompt=(user_prompt or "")[:10000],
            project=project,
            user=user,
            status=LLMLog.Status.PENDING,
        )
    except Exception:
        # DB not available (unit tests, migrations) — yield a dummy
        yield SimpleNamespace(
            output="",
            input_tokens=None,
            output_tokens=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        )
        return

    log._start_time = time.time()
    try:
        yield log
    except Exception as e:
        log.status = LLMLog.Status.ERROR
        log.error = str(e)[:10000]
        log.duration_ms = int((time.time() - log._start_time) * 1000)
        _sanitize_int_fields(log)
        # Save output + tokens even on error (for debugging)
        log.save(
            update_fields=[
                "status",
                "error",
                "duration_ms",
                "output",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "updated_at",
            ]
        )
        raise
    else:
        log.status = LLMLog.Status.SUCCESS
        log.duration_ms = int((time.time() - log._start_time) * 1000)
        _sanitize_int_fields(log)
        log.save(
            update_fields=[
                "status",
                "duration_ms",
                "output",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "updated_at",
            ]
        )


@asynccontextmanager
async def allm_log(agent_name, model, system_prompt="", user_prompt="", project=None, user=None):
    """Async version of llm_log for use in async consumers.

    Degrades gracefully if DB is unavailable — yields a dummy object
    so the actual LLM call proceeds even when logging fails.
    """
    try:
        log = await sync_to_async(LLMLog.objects.create)(
            agent_name=agent_name,
            model=model,
            system_prompt=(system_prompt or "")[:10000],
            user_prompt=(user_prompt or "")[:10000],
            project=project,
            user=user,
            status=LLMLog.Status.PENDING,
        )
    except Exception:
        logger.warning("allm_log: DB unavailable, degrading to dummy log")
        yield SimpleNamespace(
            output="",
            input_tokens=None,
            output_tokens=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        )
        return

    log._start_time = time.time()
    try:
        yield log
    except Exception as e:
        log.status = LLMLog.Status.ERROR
        log.error = str(e)[:10000]
        log.duration_ms = int((time.time() - log._start_time) * 1000)
        _sanitize_int_fields(log)
        save = sync_to_async(log.save)
        await save(
            update_fields=[
                "status",
                "error",
                "duration_ms",
                "output",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "updated_at",
            ]
        )
        raise
    else:
        log.status = LLMLog.Status.SUCCESS
        log.duration_ms = int((time.time() - log._start_time) * 1000)
        _sanitize_int_fields(log)
        save = sync_to_async(log.save)
        await save(
            update_fields=[
                "status",
                "duration_ms",
                "output",
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "updated_at",
            ]
        )
