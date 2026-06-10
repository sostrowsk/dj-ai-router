import functools
import logging
import random
import re
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_RETRY_AFTER = 30

# Bedrock/Anthropic is the PRIMARY provider, so throttling does not always arrive
# as an ``openai.RateLimitError``. It can be an ``anthropic.RateLimitError``, a
# botocore ``ThrottlingException`` (str: "Too many requests"), or a generic
# wrapper whose message/status hints at a 429. Match all of these.
_THROTTLE_MESSAGE_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "throttl",  # ThrottlingException / throttled / throttling
    "429",
    "slow down",
    "quota",
)


def _retry_after_from_headers(error: Exception) -> Optional[int]:
    """Read the Retry-After header from an SDK error response, if present.

    Supports both the httpx-based response on anthropic/openai errors
    (``error.response.headers``) and the botocore ClientError response dict
    (``error.response["ResponseMetadata"]["HTTPHeaders"]``).
    """
    response = getattr(error, "response", None)
    if response is None:
        return None

    # anthropic / openai: httpx.Response with a .headers mapping.
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is not None:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                pass

    # botocore ClientError: response is a dict.
    if isinstance(response, dict):
        http_headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        value = http_headers.get("retry-after") or http_headers.get("Retry-After")
        if value is not None:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                pass

    return None


def parse_retry_after(error: Exception) -> int:
    """Determine how long to wait before retrying a throttled request.

    Precedence: the ``Retry-After`` HTTP header (honored for Bedrock/Anthropic
    and OpenAI), then known message formats, then a conservative default.
    """
    header_value = _retry_after_from_headers(error)
    if header_value is not None:
        return header_value

    if hasattr(error, "message"):
        message = str(error.message)
    else:
        message = str(error)
    match = re.search(r"try again in (\d+) seconds", message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"retry-after: (\d+)", message, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return DEFAULT_RETRY_AFTER


def _is_throttling_error(error: Exception) -> bool:
    """True if ``error`` represents a retryable rate-limit/throttling condition.

    Covers OpenAI and Anthropic ``RateLimitError`` natively (both subclass their
    SDK's ``APIStatusError``), botocore ``ThrottlingException``, HTTP 429, and any
    error whose message carries a known throttling marker.
    """
    # Native SDK rate-limit exceptions (imported lazily so the util has no hard
    # dependency on either SDK being installed).
    try:
        from openai import RateLimitError as OpenAIRateLimitError

        if isinstance(error, OpenAIRateLimitError):
            return True
    except ImportError:  # pragma: no cover - openai always installed here
        pass
    try:
        from anthropic import RateLimitError as AnthropicRateLimitError

        if isinstance(error, AnthropicRateLimitError):
            return True
    except ImportError:  # pragma: no cover - anthropic always installed here
        pass

    # HTTP status 429 on an SDK error (httpx response or botocore response dict).
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
        if isinstance(code, str) and "throttl" in code.lower():
            return True
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 429:
            return True

    message = str(error).lower()
    return any(marker in message for marker in _THROTTLE_MESSAGE_MARKERS)


def exponential_backoff_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 120.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _is_throttling_error(e):
                        raise
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    retry_after = parse_retry_after(e)
                    delay = min(max(retry_after, base_delay * (backoff_factor**attempt)), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        f"Rate limit hit in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
            if last_exception:
                raise last_exception

        return wrapper

    return decorator
