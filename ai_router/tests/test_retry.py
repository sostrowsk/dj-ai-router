"""Regression tests for ai_router.utils.retry.

Bedrock/Anthropic is the PRIMARY provider, but the original decorator only
caught ``openai.RateLimitError`` natively and ``parse_retry_after`` only read
the OpenAI message format. Bedrock ``ThrottlingException`` / Anthropic
``RateLimitError`` fell through (no retry) and the ``Retry-After`` HTTP header
was never honored. These tests pin the broadened behavior.
"""

from unittest import mock

import anthropic
import botocore.exceptions
import httpx
import openai

from ai_router.utils.retry import exponential_backoff_retry, parse_retry_after


def _httpx_response(status_code: int, headers=None) -> httpx.Response:
    request = httpx.Request("POST", "https://bedrock.example.com")
    return httpx.Response(status_code, headers=headers or {}, request=request)


def _anthropic_rate_limit(retry_after: str | None = None) -> anthropic.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = _httpx_response(429, headers=headers)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _bedrock_throttling(retry_after: str | None = None) -> botocore.exceptions.ClientError:
    http_headers = {"retry-after": retry_after} if retry_after is not None else {}
    return botocore.exceptions.ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Too many requests, please wait before trying again.",
            },
            "ResponseMetadata": {"HTTPHeaders": http_headers, "HTTPStatusCode": 429},
        },
        operation_name="InvokeModel",
    )


class TestRetryRetriesBedrockAndAnthropic:
    """The decorator must retry the PRIMARY provider's throttling errors."""

    def test_retries_anthropic_rate_limit_error(self):
        calls = {"n": 0}

        @exponential_backoff_retry(max_retries=2, base_delay=0, jitter=False)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise _anthropic_rate_limit()
            return "ok"

        with mock.patch("ai_router.utils.retry.time.sleep"):
            assert flaky() == "ok"
        assert calls["n"] == 2

    def test_retries_bedrock_throttling_exception(self):
        calls = {"n": 0}

        @exponential_backoff_retry(max_retries=2, base_delay=0, jitter=False)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise _bedrock_throttling()
            return "ok"

        with mock.patch("ai_router.utils.retry.time.sleep"):
            assert flaky() == "ok"
        assert calls["n"] == 2

    def test_still_retries_openai_rate_limit_error(self):
        calls = {"n": 0}
        response = _httpx_response(429)
        err = openai.RateLimitError("rate limit reached", response=response, body=None)

        @exponential_backoff_retry(max_retries=2, base_delay=0, jitter=False)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise err
            return "ok"

        with mock.patch("ai_router.utils.retry.time.sleep"):
            assert flaky() == "ok"
        assert calls["n"] == 2

    def test_non_throttling_error_is_not_retried(self):
        calls = {"n": 0}

        @exponential_backoff_retry(max_retries=3, base_delay=0, jitter=False)
        def boom():
            calls["n"] += 1
            raise ValueError("totally unrelated failure")

        with mock.patch("ai_router.utils.retry.time.sleep"):
            try:
                boom()
            except ValueError:
                pass
            else:
                raise AssertionError("ValueError should propagate without retry")
        assert calls["n"] == 1


class TestParseRetryAfterHonorsHeader:
    """parse_retry_after must read the Retry-After HTTP header, not just text."""

    def test_reads_anthropic_retry_after_header(self):
        err = _anthropic_rate_limit(retry_after="42")
        assert parse_retry_after(err) == 42

    def test_reads_bedrock_retry_after_header(self):
        err = _bedrock_throttling(retry_after="17")
        assert parse_retry_after(err) == 17

    def test_header_takes_precedence_over_default(self):
        # No parseable text, but a header is present -> use the header.
        err = _anthropic_rate_limit(retry_after="7")
        assert parse_retry_after(err) == 7

    def test_falls_back_to_message_text(self):
        err = Exception("Please try again in 5 seconds.")
        assert parse_retry_after(err) == 5

    def test_default_when_nothing_parseable(self):
        err = Exception("opaque failure")
        assert parse_retry_after(err) == 30
