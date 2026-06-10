"""
Cached LLM Client Utilities.

Provides cached LLM clients for document extraction:
- CachedGeminiClient: Uses google-genai SDK (primary)
- CachedAnthropicClient: Uses Anthropic SDK (legacy/fallback)
- get_cached_client(): Factory that picks the right client based on model engine

Usage:
    client = get_cached_client(model="gemini-3.1-pro-preview")
    result, parsed = client.invoke_with_cache(
        document_content=large_document,
        extraction_prompt="Extract GuV data..."
    )
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from django.conf import settings
from pydantic import BaseModel

from ai_router.parsers import PydanticOutputParser

logger = logging.getLogger(__name__)


# Plan-S1 — SDK-Streaming-Resilience.
# Anthropic SDK (anthropic/lib/streaming/_messages.py:454) raises a
# RuntimeError("Unexpected event order, got <type> before \"message_start\"")
# when the Bedrock SSE stream sends an `error` (or any other) event before
# `message_start` — typical for transient Bedrock-side throttle / internal
# errors that arrive mid-stream after HTTP-200. We pattern-match on the
# suffix to stay robust across SDK versions and event types, and to NEVER
# blanket-catch RuntimeError (which would mask real bugs).
_SDK_STREAM_OUT_OF_ORDER_SUFFIX = 'before "message_start"'


def _is_sdk_stream_out_of_order_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and _SDK_STREAM_OUT_OF_ORDER_SUFFIX in str(exc)


class BedrockStreamOutOfOrderError(RuntimeError):
    """Plan-S1 — SDK-level Out-of-Order RuntimeError without a decodable
    Bedrock payload. Kept as a fallback for SDK anomalies that the new
    manual stream iterator (S2) cannot decode (e.g. truncated streams).
    """


# Plan-S2 — Typed Bedrock-Error-Hierarchie mit decoded SSE-Payload.
# Sobald _stream_once manuell iteriert, sehen wir das `error`-Event mit
# `error.type` / `error.message` / `request_id`. Wir mappen es auf typed
# Subclasses, damit Caller `BedrockOverloadError` von
# `BedrockInvalidRequestError` unterscheiden können — auch ohne den
# Body-String zu parsen.
class BedrockStreamErrorBase(RuntimeError):
    """Base für decoded Bedrock-Stream-Errors. Subclasses RuntimeError
    für Backward-Compat zu existing `except Exception/RuntimeError`-Pfaden
    im extraction-pipeline."""

    def __init__(
        self,
        error_type: str,
        message: str,
        model: str,
        request_id: Optional[str] = None,
    ):
        self.bedrock_error_type = error_type
        self.bedrock_message = message
        self.model = model
        self.request_id = request_id
        super().__init__(f"Bedrock {error_type} ({model}): {message}")


class BedrockOverloadError(BedrockStreamErrorBase):
    """Server-side capacity throttle. Retryable."""


class BedrockRateLimitError(BedrockStreamErrorBase):
    """Account-/key-level rate limit. Retryable."""


class BedrockApiError(BedrockStreamErrorBase):
    """Generic 5xx / internal error. Retryable."""


class BedrockTimeoutError(BedrockStreamErrorBase):
    """Server-side timeout. Retryable."""


class BedrockInvalidRequestError(BedrockStreamErrorBase):
    """Bad payload / prompt / params. NOT retryable."""


class BedrockAuthenticationError(BedrockStreamErrorBase):
    """Bad credentials. NOT retryable."""


class BedrockPermissionError(BedrockStreamErrorBase):
    """IAM / model-access denied. NOT retryable."""


class BedrockNotFoundError(BedrockStreamErrorBase):
    """Model / resource not found. NOT retryable."""


class BedrockBillingError(BedrockStreamErrorBase):
    """Quota / payment issue. NOT retryable."""


class BedrockUnknownStreamError(BedrockStreamErrorBase):
    """Decoded event with an error_type we don't have a typed class for.
    Kept distinct from BedrockStreamOutOfOrderError (SDK anomaly without
    decoded payload) so callers can still see the raw error_type/message."""


class ForcedToolUseError(RuntimeError):
    """Plan-Tool-Use — raised when a forced-tool_choice response does
    not contain the expected tool_use block.

    Mögliche Ursachen: max_tokens-Truncation mid-tool_use, Modell-Edge-
    Case (selten, aber bei forced choice trotzdem möglich). Carrier des
    raw response text damit Caller audit kann."""

    def __init__(self, message: str, raw_content: Optional[str] = None):
        super().__init__(message)
        self.raw_content = raw_content or ""


_ERROR_TYPE_TO_EXCEPTION: Dict[str, type] = {
    "overloaded_error": BedrockOverloadError,
    "rate_limit_error": BedrockRateLimitError,
    "api_error": BedrockApiError,
    "timeout_error": BedrockTimeoutError,
    "invalid_request_error": BedrockInvalidRequestError,
    "authentication_error": BedrockAuthenticationError,
    "permission_error": BedrockPermissionError,
    "not_found_error": BedrockNotFoundError,
    "billing_error": BedrockBillingError,
}

_RETRYABLE_ERROR_TYPES = frozenset({"overloaded_error", "rate_limit_error", "api_error", "timeout_error"})

# HTTP status → error_type fallback for APIStatusError without a usable
# `body["error"]` payload (some SDK paths surface only the status code).
_HTTP_STATUS_TO_ERROR_TYPE: Dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
    529: "overloaded_error",  # Anthropic-specific overload status
}


def _error_type_from_status_code(status_code: Optional[int]) -> str:
    if status_code is None:
        return "unknown_error"
    mapped = _HTTP_STATUS_TO_ERROR_TYPE.get(status_code)
    if mapped:
        return mapped
    if 500 <= status_code < 600:
        return "api_error"
    return "unknown_error"


# Retry-Schedule für Plan-S3: 4 Versuche, 3 Sleeps zwischen den
# Versuchen. Bedrock-Overload entspannt sich typisch in 30-60s.
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = (5, 15, 45)


# Plan-S2 / P2.1 — dict|object Event-Helpers.
# Anthropic-SDK kann Events als Pydantic-Objects ODER plain dicts liefern
# (je nach SDK-Version, raw vs. parsed mode). Diese Helper akzeptieren
# beides; getattr() allein würde dict-Pfade silent dropen.
def _event_field(event, key: str):
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _event_type(event) -> Optional[str]:
    return _event_field(event, "type")


def _event_request_id(event) -> Optional[str]:
    return _event_field(event, "request_id")


def _event_message(event):
    return _event_field(event, "message")


def _event_delta(event):
    return _event_field(event, "delta")


def _event_usage(event):
    return _event_field(event, "usage")


def _event_error_payload(event) -> Dict[str, Any]:
    err = _event_field(event, "error")
    if err is None:
        return {}
    if isinstance(err, dict):
        return err
    if hasattr(err, "model_dump"):
        try:
            return err.model_dump()
        except Exception:  # noqa: BLE001
            pass
    return {
        "type": getattr(err, "type", "unknown_error"),
        "message": getattr(err, "message", ""),
    }


def _delta_type(delta) -> Optional[str]:
    return _event_field(delta, "type") if delta is not None else None


def _delta_text(delta) -> str:
    if delta is None:
        return ""
    val = _event_field(delta, "text")
    return val if isinstance(val, str) else ""


def _usage_field(usage, key: str, default: int = 0) -> int:
    if usage is None:
        return default
    val = _event_field(usage, key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


@dataclass
class CachedInvocationResult:
    """Result of a cached LLM invocation."""

    content: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    duration_ms: int
    model: str

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens (fresh + cached)."""
        return self.input_tokens

    @property
    def effective_input_tokens(self) -> int:
        """Effective input tokens (accounting for cache discount)."""
        # Cache read tokens are ~10% cost, cache creation is full cost
        return (
            (self.input_tokens - self.cache_read_input_tokens - self.cache_creation_input_tokens)
            + self.cache_creation_input_tokens
            + int(self.cache_read_input_tokens * 0.1)
        )

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of tokens served from cache."""
        if self.input_tokens == 0:
            return 0.0
        return self.cache_read_input_tokens / self.input_tokens


class CachedAnthropicClient:
    """
    Anthropic client with prompt caching support.

    Uses the raw Anthropic client for proper cache_control handling,
    as LangChain's ChatAnthropic doesn't fully support cache_control yet.
    """

    # Plan-F2-A — Anthropic-Beta-Tags als Klassen-Konstanten.
    # Bedrock Live-Run 2026-05-09: `extended-output-128k-2025-02-19`
    # → 400 "invalid beta flag". Bedrock akzeptiert keine Anthropic-API-
    # only Beta-Flags. Bedrock supports `cache_control` nativ + bietet
    # max_output bis zur model-spezifischen Cap (claude-sonnet-4-6: 32k).
    # Nur direct-Anthropic-Pfad bekommt prompt-caching-Beta.
    _PROMPT_CACHING_BETA_TAG = "prompt-caching-2024-07-31"

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or settings.DEFAULT_MODEL_BEDROCK
        self.temperature = temperature
        # Plan-F2-A: settings runtime-read im Body, NICHT als Default-Arg
        # (Default-Args werden beim Import evaluiert → override_settings
        # greift nicht).
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_OUTPUT_TOKENS
        self._client = None
        self._is_bedrock = False
        self._is_vertex = False
        self._supports_temp = True
        # Phase B5 — usage des letzten stream()-Calls (SimpleNamespace mit
        # input_tokens/output_tokens/cache_*_input_tokens) oder None, wenn
        # der Stream keine usage-Events lieferte. Additiv: andere Caller
        # (invoke, _stream_once) bleiben unberuehrt.
        self.last_stream_usage = None
        self._setup_client()

    def _setup_client(self):
        """Initialize the Anthropic client (Vertex AI or Bedrock fallback)."""
        gcp_project = getattr(settings, "GCP_PROJECT_ID", "")
        gcp_region = getattr(settings, "GCP_REGION", "europe-west4")

        # Plan-F3: Anthropic-SDK-Retries client-side hochsetzen.
        # Settings runtime-read — NICHT als Default-Argument, sonst greift
        # override_settings nicht (Default-Args werden beim Import evaluiert).
        max_retries = settings.ANTHROPIC_SDK_MAX_RETRIES

        if gcp_project:
            from anthropic import AnthropicVertex

            self._is_vertex = True
            self._client = AnthropicVertex(
                project_id=gcp_project,
                region=gcp_region,
                max_retries=max_retries,
            )
            logger.info(f"CachedAnthropicClient using Vertex AI ({gcp_region})")
        else:
            from anthropic import AnthropicBedrock

            from ai_router.bedrock_client import BEDROCK_MODEL_CONFIG

            bedrock_config = BEDROCK_MODEL_CONFIG.get(self.model)
            self._model_alias = self.model  # Keep alias for logging (e.g. "claude-sonnet-4-6")
            if bedrock_config:
                self.model = bedrock_config["model_id"]
                self._supports_temp = bedrock_config.get("supports_temp", True)

            self._is_bedrock = True
            self._client = AnthropicBedrock(
                aws_region=getattr(settings, "AWS_BEDROCK_REGION", "eu-central-1"),
                aws_access_key=getattr(settings, "AWS_BEDROCK_ACCESS_KEY_ID", "") or None,
                aws_secret_key=getattr(settings, "AWS_BEDROCK_SECRET_ACCESS_KEY", "") or None,
                max_retries=max_retries,
            )
            logger.info("CachedAnthropicClient initialized with Bedrock")

    @property
    def log_model(self) -> str:
        """Model name for logging — returns alias (e.g. 'claude-sonnet-4-6') not Bedrock ID."""
        return getattr(self, "_model_alias", self.model)

    # Plan-F4 — Anthropic-SDK-Schwelle für non-streaming.
    # _calculate_nonstreaming_timeout (anthropic._base_client:730) raised
    # ValueError wenn `max_tokens * 3600/128_000 > 600` → max_tokens > ~21333.
    # Mit max_tokens=96000 (F2-A) crashes JEDER non-stream Aufruf. Wir
    # schalten ab Threshold auf messages.stream() um.
    _NONSTREAMING_MAX_TOKENS_THRESHOLD = 21000

    def _should_use_streaming(self) -> bool:
        return bool(self.max_tokens and self.max_tokens > self._NONSTREAMING_MAX_TOKENS_THRESHOLD)

    def _create_or_stream(self, create_kwargs: Dict[str, Any]):
        """Plan-F4 — schaltet auf streaming-create um wenn max_tokens
        die SDK-non-stream-Schwelle übersteigt.

        Plan-S2 — `_stream_once` iteriert `messages.create(stream=True)`
        raw und dekodiert `error`-Events / APIStatusError in typed
        BedrockStream*Error mit decoded payload.

        Plan-S3 — retried hier auf `overloaded_error`, `rate_limit_error`,
        `api_error`, `timeout_error` über max _MAX_ATTEMPTS Versuche mit
        Backoff _BACKOFF_SECONDS. Non-retryable types raisen sofort.
        SDK-Anomalien ohne decodable payload (BedrockStreamOutOfOrderError)
        werden gleich behandelt wie api_error (retryable).

        Liefert ein response-like Object mit `.content[].text` und
        `.usage` — Caller-Kontrakt identisch zu messages.create().
        """
        if not self._should_use_streaming():
            return self._client.messages.create(**create_kwargs)

        # Plan-C1 — beide Retry-Pfade (typed BedrockStream*Error + S1-
        # Fallback für SDK-RuntimeError ohne decodable payload) teilen
        # denselben Backoff-Schedule und denselben Log-Tag. Eine
        # last_exception-Variable, ein post-loop raise.
        last_exception: Optional[Exception] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._stream_once(create_kwargs)
            except BedrockStreamErrorBase as exc:
                if exc.bedrock_error_type not in _RETRYABLE_ERROR_TYPES:
                    logger.error(
                        "Bedrock %s on attempt %d (model=%s, request_id=%s): %s",
                        exc.bedrock_error_type,
                        attempt,
                        self.log_model,
                        exc.request_id,
                        exc.bedrock_message,
                    )
                    raise
                last_exception = exc
                self._log_stream_retry(attempt, exc)
            except RuntimeError as exc:
                if not _is_sdk_stream_out_of_order_error(exc):
                    raise
                wrapped = BedrockStreamOutOfOrderError(f"Bedrock stream out-of-order (model={self.log_model}): {exc}")
                wrapped.__cause__ = exc
                last_exception = wrapped
                self._log_stream_retry(attempt, wrapped)

            if attempt < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_SECONDS[attempt - 1])

        assert last_exception is not None
        raise last_exception

    def _log_stream_retry(self, attempt: int, exc: Exception) -> None:
        """Plan-C1 — unified WARN-log für stream-retry-Pfad. Distinkter
        Tag pro Exception-Class damit grep ['rl-trace-typed'] /
        ['rl-trace-ooo'] funktioniert; suffix 'retrying in …s' / 'giving up'
        je nach attempt-position."""
        will_retry = attempt < _MAX_ATTEMPTS
        backoff = _BACKOFF_SECONDS[attempt - 1] if will_retry else 0
        suffix = f"retrying in {backoff}s" if will_retry else "giving up"

        if isinstance(exc, BedrockStreamErrorBase):
            logger.warning(
                "Bedrock %s on attempt %d/%d (model=%s, request_id=%s): %s — %s",
                exc.bedrock_error_type,
                attempt,
                _MAX_ATTEMPTS,
                self.log_model,
                exc.request_id,
                exc.bedrock_message,
                suffix,
            )
        else:
            # BedrockStreamOutOfOrderError — message tells the story
            cause = exc.__cause__ or exc
            logger.warning(
                "Stream-Out-of-Order on attempt %d/%d (model=%s): %s — %s",
                attempt,
                _MAX_ATTEMPTS,
                self.log_model,
                cause,
                suffix,
            )

    def _raise_typed_from_error_payload(
        self,
        err: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> None:
        err_type = err.get("type") if isinstance(err, dict) else None
        err_msg = err.get("message") if isinstance(err, dict) else None
        if not err_type:
            err_type = "unknown_error"
        if not err_msg:
            err_msg = "(no message)"
        exc_class = _ERROR_TYPE_TO_EXCEPTION.get(err_type, BedrockUnknownStreamError)
        raise exc_class(err_type, err_msg, self.log_model, request_id)

    def _raise_typed_from_api_status_error(self, api_exc) -> None:
        """Decode anthropic.APIStatusError into a typed BedrockStream*Error.
        Prefers `body['error']`; falls back to status_code mapping."""
        body = getattr(api_exc, "body", None)
        err_payload: Dict[str, Any] = {}
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                err_payload = inner
            elif inner is not None and hasattr(inner, "model_dump"):
                try:
                    err_payload = inner.model_dump()
                except Exception:  # noqa: BLE001
                    err_payload = {}
        request_id = getattr(api_exc, "request_id", None)

        if not err_payload.get("type"):
            status_code = getattr(api_exc, "status_code", None)
            err_payload = {
                "type": _error_type_from_status_code(status_code),
                "message": err_payload.get("message") or getattr(api_exc, "message", "") or f"HTTP {status_code}",
            }
        self._raise_typed_from_error_payload(err_payload, request_id=request_id)

    def _iter_stream_events(self, create_kwargs: Dict[str, Any]):
        """Plan-S2 — yield raw RawMessageStreamEvent from messages.create(
        stream=True). Decodes `error`-events and APIStatusError into
        typed BedrockStream*Error; everything else passes through.
        """
        from anthropic import APIStatusError

        try:
            raw_stream = self._client.messages.create(stream=True, **create_kwargs)
        except APIStatusError as api_exc:
            self._raise_typed_from_api_status_error(api_exc)
            return  # pragma: no cover — _raise_typed_* always raises

        close = getattr(raw_stream, "close", None)
        try:
            for event in raw_stream:
                if _event_type(event) == "error":
                    self._raise_typed_from_error_payload(
                        _event_error_payload(event),
                        request_id=_event_request_id(event),
                    )
                yield event
        except APIStatusError as api_exc:
            self._raise_typed_from_api_status_error(api_exc)
        finally:
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 — defensive on cleanup
                    pass

    def _stream_once(self, create_kwargs: Dict[str, Any]):
        """Plan-S2 — single-attempt streaming via _iter_stream_events.
        Accumulates text chunks + usage from message_start/content_block_delta/
        message_delta; tolerates dict|object event shapes."""
        from types import SimpleNamespace

        text_chunks: List[str] = []
        input_tokens = output_tokens = cache_creation = cache_read = 0

        for event in self._iter_stream_events(create_kwargs):
            evt_type = _event_type(event)
            if evt_type == "message_start":
                msg = _event_message(event)
                if msg is not None:
                    usage = _event_field(msg, "usage")
                    if usage is not None:
                        input_tokens = _usage_field(usage, "input_tokens", default=input_tokens)
                        cache_creation = _usage_field(usage, "cache_creation_input_tokens", default=cache_creation)
                        cache_read = _usage_field(usage, "cache_read_input_tokens", default=cache_read)
            elif evt_type == "content_block_delta":
                delta = _event_delta(event)
                if _delta_type(delta) == "text_delta":
                    text_chunks.append(_delta_text(delta))
            elif evt_type == "message_delta":
                usage = _event_usage(event)
                if usage is not None:
                    output_tokens = _usage_field(usage, "output_tokens", default=output_tokens)

        usage_obj = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        )
        content_text = "".join(text_chunks)
        return SimpleNamespace(
            content=[SimpleNamespace(text=content_text)],
            usage=usage_obj,
        )

    def _beta_headers_for_provider(self) -> Optional[Dict[str, str]]:
        """Plan-F2-A — Extra-Headers für anthropic-beta, provider-aware.

        Liest INSTANCE-STATE (`self._is_vertex`, `self._is_bedrock`),
        nicht live `settings.GCP_PROJECT_ID` — Setup ist zur
        Construction-Time finalisiert; spätere `override_settings`-
        oder Config-Reloads dürfen den Helper nicht inkonsistent zum
        bereits aufgebauten `_client` machen.

        - Vertex: keine extra_headers (nicht supported).
        - Bedrock: KEINE extra_headers — Bedrock validates anthropic-
          beta-Flags und akzeptiert keine Anthropic-API-only Tags.
          cache_control wird nativ supported (GA April 2025).
          Live-Run 2026-05-09: extended-output-128k-Tag → 400 "invalid
          beta flag" → ganze Extraction crashed.
        - Direct Anthropic: prompt-caching-Beta-Tag.
        """
        # getattr-fallback: legacy Test-Stubs die `_setup_client` mocken
        # ohne `_is_vertex` zu setzen — defaulting auf False = direct-
        # Anthropic-Pfad (sicherer als Crash).
        if getattr(self, "_is_vertex", False):
            return None
        if self._is_bedrock:
            return None
        return {"anthropic-beta": self._PROMPT_CACHING_BETA_TAG}

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Tuple["CachedInvocationResult", Optional[BaseModel]]:
        """
        Simple text-only invoke — drop-in replacement for LangChain chain.invoke().

        Args:
            system_prompt: System instructions
            user_prompt: User message
            output_schema: Optional Pydantic schema for structured output

        Returns:
            Tuple of (CachedInvocationResult, parsed_output or None)
        """
        start_time = time.time()

        system_content = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        user_text = user_prompt
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            user_text = f"{user_prompt}\n\n## Output Format:\n{format_instructions}"

        try:
            is_vertex = getattr(settings, "GCP_PROJECT_ID", "")
            create_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_content,
                "messages": [{"role": "user", "content": user_text}],
            }
            if not self._supports_temp:
                create_kwargs.pop("temperature", None)
            beta = self._beta_headers_for_provider()
            if beta:
                create_kwargs["extra_headers"] = beta
            if is_vertex:
                for block in create_kwargs["system"]:
                    block.pop("cache_control", None)

            response = self._create_or_stream(create_kwargs)

            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0)

            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                duration_ms=duration_ms,
                model=self.model,
            )

            parsed_output = None
            if output_schema and content:
                try:
                    parser = PydanticOutputParser(pydantic_object=output_schema)
                    parsed_output = parser.parse(content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse output: {parse_error}")

            return result, parsed_output

        except Exception as e:
            logger.error(f"Invoke failed: {e}")
            raise

    def stream(
        self,
        system_prompt: str,
        user_prompt: str = "",
        messages: list[dict] | None = None,
    ):
        """
        Streaming text invoke — replacement for LangChain astream().

        Yields text chunks as they arrive.
        Pass ``messages`` (list of role/content dicts) for multi-turn chat.
        """
        system_content = [
            {
                "type": "text",
                "text": system_prompt,
            }
        ]

        is_vertex = getattr(settings, "GCP_PROJECT_ID", "")
        create_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_content,
            "messages": messages or [{"role": "user", "content": user_prompt}],
        }
        if not self._supports_temp:
            create_kwargs.pop("temperature", None)
        beta = self._beta_headers_for_provider()
        if beta:
            create_kwargs["extra_headers"] = beta
        if is_vertex:
            for block in create_kwargs["system"]:
                block.pop("cache_control", None)

        # Plan-S2 — route through _iter_stream_events so Bedrock
        # error-events / APIStatusError surface as typed
        # BedrockStream*Error instead of being swallowed by the SDK's
        # accumulate_event RuntimeError.
        # Phase B5 — usage aus message_start/message_delta in
        # self.last_stream_usage capturen (inkrementell, damit auch bei
        # mid-stream-Abbruch die bereits gesehenen Werte verfuegbar sind).
        from types import SimpleNamespace

        self.last_stream_usage = None
        for event in self._iter_stream_events(create_kwargs):
            evt_type = _event_type(event)
            if evt_type == "message_start":
                msg = _event_message(event)
                usage = _event_field(msg, "usage") if msg is not None else None
                if usage is not None:
                    acc = self.last_stream_usage or SimpleNamespace(
                        input_tokens=0,
                        output_tokens=0,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                    )
                    acc.input_tokens = _usage_field(usage, "input_tokens", default=acc.input_tokens)
                    acc.cache_creation_input_tokens = _usage_field(
                        usage, "cache_creation_input_tokens", default=acc.cache_creation_input_tokens
                    )
                    acc.cache_read_input_tokens = _usage_field(
                        usage, "cache_read_input_tokens", default=acc.cache_read_input_tokens
                    )
                    self.last_stream_usage = acc
            elif evt_type == "content_block_delta":
                delta = _event_delta(event)
                if _delta_type(delta) == "text_delta":
                    yield _delta_text(delta)
            elif evt_type == "message_delta":
                usage = _event_usage(event)
                if usage is not None:
                    acc = self.last_stream_usage or SimpleNamespace(
                        input_tokens=0,
                        output_tokens=0,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                    )
                    acc.output_tokens = _usage_field(usage, "output_tokens", default=acc.output_tokens)
                    self.last_stream_usage = acc

    def _build_system_message_with_cache(
        self,
        document_content: str,
        document_name: str,
        document_tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build system message content with cache control marker.

        The document content is marked for caching so subsequent calls
        with different extraction prompts can reuse the cached content.
        """
        system_text = f"""Du bist ein Experte für Finanzanalyse und Dokumentenextraktion.
Du analysierst Geschäftsdokumente und extrahierst strukturierte Daten.

## Dokument zur Analyse

**Dokumentname:** {document_name}
**Dokumenttyp:** {document_tag or 'Unbekannt'}

---

## Dokumentinhalt:

{document_content}

---

Analysiere das obige Dokument sorgfältig und extrahiere die angeforderten Daten.
Beachte dabei:
- Erfasse Werte GENAU so, wie sie im Dokument stehen
- Bei deutschen Zahlenformaten: 1.234,56 = 1234.56
- Klammern bedeuten negative Werte: (1.234) = -1234
- TEUR bedeutet Tausend Euro
- Wenn Daten nicht vorhanden sind, setze null/None
"""
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def invoke_with_cache(
        self,
        document_content: str,
        document_name: str,
        extraction_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        document_tag: Optional[str] = None,
    ) -> Tuple[CachedInvocationResult, Optional[BaseModel]]:
        """
        Invoke the LLM with prompt caching enabled.

        Args:
            document_content: The document content to analyze (will be cached)
            document_name: Name of the document
            extraction_prompt: The specific extraction prompt
            output_schema: Optional Pydantic schema for structured output
            document_tag: Optional document tag/category

        Returns:
            Tuple of (CachedInvocationResult, parsed_output or None)
        """
        start_time = time.time()

        # Build system message with cache control
        system_content = self._build_system_message_with_cache(
            document_content=document_content,
            document_name=document_name,
            document_tag=document_tag,
        )

        # Add format instructions if schema provided
        user_prompt = extraction_prompt
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            user_prompt = f"{extraction_prompt}\n\n## Output Format:\n{format_instructions}"

        try:
            # Prompt caching: Bedrock supports cache_control natively (GA April 2025),
            # direct Anthropic API needs beta header, Vertex AI not supported
            is_vertex = getattr(settings, "GCP_PROJECT_ID", "")
            create_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_content,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            if not self._supports_temp:
                create_kwargs.pop("temperature", None)
            beta = self._beta_headers_for_provider()
            if beta:
                create_kwargs["extra_headers"] = beta
            if is_vertex:
                # Vertex AI: cache_control nicht unterstützt
                for block in create_kwargs["system"]:
                    block.pop("cache_control", None)
            # Bedrock: cache_control wird nativ unterstützt (GA seit April 2025)

            response = self._create_or_stream(create_kwargs)

            # Extract content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            # Extract usage info
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0)

            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                duration_ms=duration_ms,
                model=self.model,
            )

            # Parse output if schema provided
            parsed_output = None
            if output_schema and content:
                try:
                    parser = PydanticOutputParser(pydantic_object=output_schema)
                    parsed_output = parser.parse(content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse output: {parse_error}")

            # Log cache statistics
            if cache_read > 0:
                logger.info(
                    f"Cache hit: {cache_read} tokens read from cache " f"({result.cache_hit_ratio:.1%} hit ratio)"
                )
            elif cache_creation > 0:
                logger.info(f"Cache miss: {cache_creation} tokens added to cache")

            return result, parsed_output

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Cached invocation failed: {e}")
            raise

    def invoke_with_pdf_cache(
        self,
        pdf_path: Optional[str] = None,
        system_prompt: str = "",
        extraction_prompt: str = "",
        output_schema: Optional[Type[BaseModel]] = None,
        *,
        pdf_bytes: Optional[bytes] = None,
    ) -> Tuple[CachedInvocationResult, Optional[BaseModel]]:
        """
        Invoke the LLM with an original PDF document and prompt caching.

        The system prompt (extraction rules) is cached. The PDF is sent as a
        document block in the user message with cache_control so that subsequent
        calls with the same PDF (e.g. GuV then Bilanz) get a cache hit.

        Args:
            pdf_path: Path to the PDF file (mutually exclusive with pdf_bytes)
            pdf_bytes: PDF data already in memory (mutually exclusive with
                pdf_path). Used by the Phase-4 slicer which produces bytes
                in-memory; round-tripping through the filesystem per call
                would be wasteful when the slice already lives in RAM.
            system_prompt: Extraction rules (e.g. GuV or Bilanz system prompt)
            extraction_prompt: User prompt with format instructions
            output_schema: Optional Pydantic schema for structured output

        Returns:
            Tuple of (CachedInvocationResult, parsed_output or None)
        """
        import base64
        from pathlib import Path

        if pdf_path is not None and pdf_bytes is not None:
            raise ValueError("pass pdf_path or pdf_bytes, not both")
        if pdf_bytes is None:
            if pdf_path is None:
                raise ValueError("either pdf_path or pdf_bytes is required")
            pdf_bytes = Path(pdf_path).read_bytes()

        start_time = time.time()

        # Read and encode PDF
        pdf_data = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        # PDF goes FIRST in user message with cache_control — this is the shared
        # prefix across OCR, GuV, and Bilanz calls. The extraction-specific prompt
        # (system_prompt + extraction_prompt) goes AFTER the PDF so the PDF cache
        # is reused across all extraction types for the same document.
        #
        # Cache structure:
        #   [PDF document block — CACHED] + [extraction rules + prompt — varies]
        #
        # This means: OCR caches the PDF, then GuV/Bilanz get cache hits on the PDF.

        # Build the full extraction prompt: system rules + user prompt + format instructions
        full_prompt = f"{system_prompt}\n\n---\n\n{extraction_prompt}"
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            full_prompt += f"\n\n## Output Format:\n{format_instructions}"

        user_content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_data,
                },
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": full_prompt},
        ]

        try:
            is_vertex = getattr(settings, "GCP_PROJECT_ID", "")
            create_kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": user_content}],
            }
            if not self._supports_temp:
                create_kwargs.pop("temperature", None)
            beta = self._beta_headers_for_provider()
            if beta:
                create_kwargs["extra_headers"] = beta
            if is_vertex:
                for block in user_content:
                    block.pop("cache_control", None)

            response = self._create_or_stream(create_kwargs)

            # Extract content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            # Extract usage info
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0)

            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                duration_ms=duration_ms,
                model=self.model,
            )

            # Parse output if schema provided
            parsed_output = None
            if output_schema and content:
                try:
                    parser = PydanticOutputParser(pydantic_object=output_schema)
                    parsed_output = parser.parse(content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse PDF extraction output: {parse_error}")

            # Log cache statistics
            if cache_read > 0:
                logger.info(f"PDF cache hit: {cache_read} tokens from cache ({result.cache_hit_ratio:.1%} hit ratio)")
            elif cache_creation > 0:
                logger.info(f"PDF cache miss: {cache_creation} tokens added to cache")

            return result, parsed_output

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"PDF cached invocation failed after {duration_ms}ms: {e}")
            raise

    def stream_pdf(
        self,
        pdf_path: str,
        prompt: str,
        max_tokens: int = 32000,
    ) -> CachedInvocationResult:
        """
        Stream extraction from a PDF document. Used for OCR markdown extraction
        where streaming avoids SDK timeouts on large documents.

        Returns CachedInvocationResult with the full concatenated content.
        """
        import base64
        from pathlib import Path

        start_time = time.time()
        pdf_data = base64.standard_b64encode(Path(pdf_path).read_bytes()).decode("utf-8")

        result_parts = []
        total_chars = 0
        next_log = 5000
        # Plan-F2-A: Beta-Header für Bedrock auch im Streaming-OCR-Pfad.
        stream_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data},
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        beta = self._beta_headers_for_provider()
        if beta:
            stream_kwargs["extra_headers"] = beta
        # Plan-S2 — route through _iter_stream_events so Bedrock
        # error-events / APIStatusError surface as typed
        # BedrockStream*Error. Usage is aggregated from message_start +
        # message_delta events (no SDK get_final_message dependency).
        input_tokens = output_tokens = cache_creation = cache_read = 0
        for event in self._iter_stream_events(stream_kwargs):
            evt_type = _event_type(event)
            if evt_type == "message_start":
                msg = _event_message(event)
                if msg is not None:
                    usage = _event_field(msg, "usage")
                    if usage is not None:
                        input_tokens = _usage_field(usage, "input_tokens", default=input_tokens)
                        cache_creation = _usage_field(usage, "cache_creation_input_tokens", default=cache_creation)
                        cache_read = _usage_field(usage, "cache_read_input_tokens", default=cache_read)
            elif evt_type == "content_block_delta":
                delta = _event_delta(event)
                if _delta_type(delta) == "text_delta":
                    text = _delta_text(delta)
                    result_parts.append(text)
                    total_chars += len(text)
                    if total_chars >= next_log:
                        logger.info("PDF extraction streaming: %d chars received", total_chars)
                        next_log += 5000
            elif evt_type == "message_delta":
                usage = _event_usage(event)
                if usage is not None:
                    output_tokens = _usage_field(usage, "output_tokens", default=output_tokens)

        content = "".join(result_parts)
        duration_ms = int((time.time() - start_time) * 1000)

        return CachedInvocationResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            duration_ms=duration_ms,
            model=self.model,
        )

    def invoke_image(
        self,
        image_data_base64: str,
        media_type: str,
        prompt: str,
        max_tokens: int = 4096,
    ) -> CachedInvocationResult:
        """
        Invoke with an image (base64 PNG/JPEG). Used for vision OCR.

        Returns CachedInvocationResult with the extracted text.
        """
        start_time = time.time()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": image_data_base64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        content = response.content[0].text if response.content else ""
        usage = response.usage
        duration_ms = int((time.time() - start_time) * 1000)

        return CachedInvocationResult(
            content=content,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
            duration_ms=duration_ms,
            model=self.model,
        )

    def invoke_with_pdf_tool(
        self,
        *,
        pdf_bytes: bytes,
        system_prompt: str,
        user_prompt: str,
        tool_name: str,
        tool_description: str,
        tool_input_schema: Dict[str, Any],
        max_tokens: int = 4096,
    ) -> Tuple[CachedInvocationResult, Dict[str, Any]]:
        """Plan-Tool-Use — strukturell erzwungener JSON-Channel via
        Anthropic forced `tool_choice`.

        Im Unterschied zu invoke_with_pdf_cache:
        - Setzt `tools=[{name, description, input_schema}]` + `tool_choice=
          {"type": "tool", "name": tool_name}` — Modell MUSS einen
          tool_use-Block ausgeben statt freien Text.
        - Per-call `max_tokens` (Default 4096) statt self.max_tokens —
          Review-Output ist klein, kein 96k-Budget nötig. Bleibt unter
          dem _NONSTREAMING_MAX_TOKENS_THRESHOLD damit non-streaming
          messages.create direkt gerufen wird (kein _create_or_stream-
          Detour, kein streaming-Pfad nötig).

        Returns: (CachedInvocationResult, tool_input dict).
          CachedInvocationResult.content ist der JSON-serialisierte
          tool_input (truncated auf 4000 chars) für Audit-Zwecke.

        Raises:
          ForcedToolUseError: response ohne tool_use-Block mit
            passendem tool_name (z.B. wenn max_tokens mid-tool_use
            truncated).
        """
        import base64

        start_time = time.time()
        pdf_data = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        user_content = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_data,
                },
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": full_prompt},
        ]

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user_content}],
            "tools": [
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": tool_input_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        if not self._supports_temp:
            create_kwargs.pop("temperature", None)
        beta = self._beta_headers_for_provider()
        if beta:
            create_kwargs["extra_headers"] = beta
        if getattr(self, "_is_vertex", False):
            for block in user_content:
                block.pop("cache_control", None)

        # Non-streaming direkt — keine _create_or_stream-Schicht.
        # max_tokens=4096 ist unter dem SDK-non-stream-Threshold, plus
        # tool_use-Events vom Stream-Iterator werden im aktuellen Code
        # nicht handled.
        response = self._client.messages.create(**create_kwargs)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0

        tool_input: Optional[Dict[str, Any]] = None
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                raw_input = getattr(block, "input", None)
                if isinstance(raw_input, dict):
                    tool_input = raw_input
                break

        import json as _json

        content_repr = _json.dumps(tool_input, default=str)[:4000] if tool_input is not None else ""
        duration_ms = int((time.time() - start_time) * 1000)

        result = CachedInvocationResult(
            content=content_repr,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            duration_ms=duration_ms,
            model=self.model,
        )

        if tool_input is None:
            stop_reason = getattr(response, "stop_reason", "?")
            raise ForcedToolUseError(
                f"Response has no tool_use block for {tool_name!r} "
                f"(stop_reason={stop_reason}) despite forced tool_choice",
                raw_content=str(getattr(response, "content", ""))[:2000],
            )

        return result, tool_input

    def invoke_raw_cached(
        self,
        system_text: str,
        user_prompt: str,
    ) -> CachedInvocationResult:
        """Invoke with a custom cached system prompt (no document template).

        Use this when you need caching but have your own system prompt structure,
        e.g. for info-memo chapter generation where the document content is part
        of a custom system prompt.
        """
        start_time = time.time()
        system_content = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        is_vertex = getattr(settings, "GCP_PROJECT_ID", "")
        create_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_content,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if not self._supports_temp:
            create_kwargs.pop("temperature", None)
        beta = self._beta_headers_for_provider()
        if beta:
            create_kwargs["extra_headers"] = beta
        if is_vertex:
            for block in create_kwargs["system"]:
                block.pop("cache_control", None)

        try:
            response = self._create_or_stream(create_kwargs)

            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
            cache_read = getattr(usage, "cache_read_input_tokens", 0)
            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                duration_ms=duration_ms,
                model=self.model,
            )

            if cache_read > 0:
                logger.info(
                    f"Cache hit: {cache_read} tokens read from cache " f"({result.cache_hit_ratio:.1%} hit ratio)"
                )
            elif cache_creation > 0:
                logger.info(f"Cache miss: {cache_creation} tokens added to cache")

            return result

        except Exception as e:
            logger.error(f"Raw cached invocation failed: {e}")
            raise


class CachedGeminiClient:
    """
    Gemini client via google-genai SDK with same interface as CachedAnthropicClient.

    Uses system_instruction for document content (analogous to Anthropic's cached system prompt).
    No prompt caching — Gemini pricing makes it unnecessary.
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or settings.DEFAULT_MODEL_VERTEX
        self.temperature = temperature
        # Plan-F2-A: Gemini bekommt eigenes Setting (NICHT symmetrisch
        # zu Anthropic 96000 — Gemini-Modelle haben modell-abhängige Caps).
        self.max_tokens = max_tokens if max_tokens is not None else settings.GEMINI_MAX_OUTPUT_TOKENS
        self._client = None
        self._setup_client()

    def _setup_client(self):
        from ai_router.vertex_client import _get_gemini_client

        self._client = _get_gemini_client()
        if settings.GCP_GEMINI_API_KEY:
            logger.info("CachedGeminiClient using Gemini Developer API (API key)")
        else:
            logger.info(
                f"CachedGeminiClient using Vertex AI project={settings.GCP_GEMINI_PROJECT_ID}, "
                f"region={settings.GCP_GEMINI_REGION}"
            )

    @property
    def log_model(self) -> str:
        """Model name for logging — Gemini doesn't need alias resolution."""
        return self.model

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Tuple["CachedInvocationResult", Optional[BaseModel]]:
        """Simple text-only invoke — drop-in replacement for LangChain chain.invoke()."""
        start_time = time.time()

        user_text = user_prompt
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            user_text = f"{user_prompt}\n\n## Output Format:\n{format_instructions}"

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_text,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "system_instruction": system_prompt,
                },
            )

            content = response.text or ""

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                duration_ms=duration_ms,
                model=self.model,
            )

            parsed_output = None
            if output_schema and content:
                try:
                    parser = PydanticOutputParser(pydantic_object=output_schema)
                    parsed_output = parser.parse(content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse Gemini output: {parse_error}")

            return result, parsed_output

        except Exception as e:
            logger.error(f"Gemini invoke failed: {e}")
            raise

    def stream(
        self,
        system_prompt: str,
        user_prompt: str = "",
        messages: list[dict] | None = None,
    ):
        """Streaming text invoke — yields text chunks.
        Pass ``messages`` (list of role/content dicts) for multi-turn chat.
        """
        if messages:
            contents = [
                {"role": "model" if m["role"] == "assistant" else m["role"], "parts": [{"text": m["content"]}]}
                for m in messages
            ]
        else:
            contents = user_prompt
        response = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "system_instruction": system_prompt,
            },
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _build_system_instruction(
        self,
        document_content: str,
        document_name: str,
        document_tag: Optional[str] = None,
    ) -> str:
        return f"""Du bist ein Experte für Finanzanalyse und Dokumentenextraktion.
Du analysierst Geschäftsdokumente und extrahierst strukturierte Daten.

## Dokument zur Analyse

**Dokumentname:** {document_name}
**Dokumenttyp:** {document_tag or 'Unbekannt'}

---

## Dokumentinhalt:

{document_content}

---

Analysiere das obige Dokument sorgfältig und extrahiere die angeforderten Daten.
Beachte dabei:
- Erfasse Werte GENAU so, wie sie im Dokument stehen
- Bei deutschen Zahlenformaten: 1.234,56 = 1234.56
- Klammern bedeuten negative Werte: (1.234) = -1234
- TEUR bedeutet Tausend Euro
- Wenn Daten nicht vorhanden sind, setze null/None
"""

    def invoke_with_cache(
        self,
        document_content: str,
        document_name: str,
        extraction_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        document_tag: Optional[str] = None,
    ) -> Tuple["CachedInvocationResult", Optional[BaseModel]]:
        start_time = time.time()

        system_instruction = self._build_system_instruction(
            document_content=document_content,
            document_name=document_name,
            document_tag=document_tag,
        )

        user_prompt = extraction_prompt
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            user_prompt = f"{extraction_prompt}\n\n## Output Format:\n{format_instructions}"

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "system_instruction": system_instruction,
                },
            )

            content = response.text or ""

            # Extract usage from response
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                duration_ms=duration_ms,
                model=self.model,
            )

            parsed_output = None
            if output_schema and content:
                try:
                    parser = PydanticOutputParser(pydantic_object=output_schema)
                    parsed_output = parser.parse(content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse output: {parse_error}")

            return result, parsed_output

        except Exception as e:
            logger.error(f"Gemini invocation failed: {e}")
            raise

    def invoke_with_pdf_cache(
        self,
        pdf_path: Optional[str] = None,
        system_prompt: str = "",
        extraction_prompt: str = "",
        output_schema: Optional[Type[BaseModel]] = None,
        *,
        pdf_bytes: Optional[bytes] = None,
    ) -> Tuple["CachedInvocationResult", Optional[BaseModel]]:
        """
        Invoke Gemini with an original PDF document.

        Signature mirrors CachedAnthropicClient: provider-agnostic
        callers (extract_document_data_task) can pass `pdf_bytes`
        from the Phase-4 slicer without filesystem round-trips.

        Uses google-genai's native file upload for PDF processing.
        System prompt goes into system_instruction, PDF + extraction prompt into contents.
        """
        from pathlib import Path

        if pdf_path is not None and pdf_bytes is not None:
            raise ValueError("pass pdf_path or pdf_bytes, not both")
        if pdf_bytes is None:
            if pdf_path is None:
                raise ValueError("either pdf_path or pdf_bytes is required")
            pdf_bytes = Path(pdf_path).read_bytes()

        start_time = time.time()

        # Build user prompt with format instructions
        user_prompt = extraction_prompt
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            format_instructions = parser.get_format_instructions()
            user_prompt = f"{extraction_prompt}\n\n## Output Format:\n{format_instructions}"

        try:
            from google.genai import types

            # Create PDF part for Gemini
            pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

            response = self._client.models.generate_content(
                model=self.model,
                contents=[pdf_part, user_prompt],
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "system_instruction": system_prompt,
                },
            )

            content = response.text or ""

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            duration_ms = int((time.time() - start_time) * 1000)

            result = CachedInvocationResult(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                duration_ms=duration_ms,
                model=self.model,
            )

            parsed_output = None
            if output_schema and content:
                try:
                    parser = PydanticOutputParser(pydantic_object=output_schema)
                    parsed_output = parser.parse(content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse Gemini PDF output: {parse_error}")

            return result, parsed_output

        except Exception as e:
            logger.error(f"Gemini PDF invocation failed: {e}")
            raise


def get_cached_client(model: str = None) -> CachedAnthropicClient | CachedGeminiClient:
    """Factory: returns the right cached client based on model engine."""
    from ai_router.vertex_client import VERTEX_MODEL_CONFIG

    model = model or settings.DEFAULT_MODEL_BEDROCK
    config = VERTEX_MODEL_CONFIG.get(model)

    if config and config["engine"] == "gemini":
        return CachedGeminiClient(model=model)
    return CachedAnthropicClient(model=model)


def get_document_cache_key(document_content: str) -> str:
    """
    Generate a cache key for document content.

    Used to verify cache hits when debugging.
    """
    return hashlib.sha256(document_content.encode()).hexdigest()[:16]
