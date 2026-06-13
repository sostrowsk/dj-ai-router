"""Local-model client (vllm-mlx / OpenAI-compatible) — routing + invoke/stream.

CachedLocalClient talks to a local `vllm-mlx serve` endpoint via the OpenAI
SDK. Tests mock the SDK client (no real network) and patch _setup_client so
no live server is needed. Reasoning models surface a separate
`reasoning_content` delta that must NOT leak into the yielded text.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from django.test import override_settings


def _make_client(model="qwen3.6-35b-a3b", max_tokens=256, **kw):
    from ai_router.local_client import CachedLocalClient

    with patch.object(CachedLocalClient, "_setup_client", lambda self: None):
        client = CachedLocalClient(model=model, max_tokens=max_tokens, **kw)
    client._client = MagicMock()
    return client


def _completion(content, prompt_tokens=11, completion_tokens=7):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _content_chunk(text=None, reasoning=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, reasoning_content=reasoning))],
        usage=None,
    )


def _usage_chunk(prompt_tokens, completion_tokens):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class TestResolveLocalConfig:
    def test_known_alias_resolves(self):
        from ai_router.local_client import resolve_local_config

        config = resolve_local_config("qwen3.6-35b-a3b")
        assert config["model_id"] == "unsloth/Qwen3.6-35B-A3B-MLX-8bit"
        assert config["base_url"] == "http://localhost:8001/v1"
        assert config["reasoning"] is True

    def test_second_known_alias_resolves(self):
        from ai_router.local_client import resolve_local_config

        config = resolve_local_config("gemma-4-12b-it")
        assert config["model_id"] == "mlx-community/gemma-4-12B-it-8bit"
        assert config["base_url"] == "http://localhost:8002/v1"

    def test_unknown_model_returns_none(self):
        from ai_router.local_client import resolve_local_config

        assert resolve_local_config("claude-sonnet-4-6") is None
        assert resolve_local_config(None) is None

    @override_settings(AI_ROUTER_LOCAL_MODELS={"qwen3.6-35b-a3b": {"base_url": "http://gpu-box:9000/v1"}})
    def test_settings_override_merges_by_alias(self):
        from ai_router.local_client import resolve_local_config

        config = resolve_local_config("qwen3.6-35b-a3b")
        # overridden field
        assert config["base_url"] == "http://gpu-box:9000/v1"
        # built-in fields preserved
        assert config["model_id"] == "unsloth/Qwen3.6-35B-A3B-MLX-8bit"

    @override_settings(
        AI_ROUTER_LOCAL_MODELS={"my-local": {"model_id": "org/My-Model", "base_url": "http://localhost:8003/v1"}}
    )
    def test_settings_can_add_new_alias(self):
        from ai_router.local_client import resolve_local_config

        config = resolve_local_config("my-local")
        assert config["model_id"] == "org/My-Model"


class TestRouting:
    def test_get_cached_client_routes_local_alias(self):
        from ai_router.cached_llm import get_cached_client
        from ai_router.local_client import CachedLocalClient

        with patch.object(CachedLocalClient, "_setup_client", lambda self: None):
            client = get_cached_client("gemma-4-12b-it")
        assert isinstance(client, CachedLocalClient)
        assert client.model == "mlx-community/gemma-4-12B-it-8bit"


class TestInvoke:
    def test_invoke_returns_content_and_token_usage(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _completion("Antwort", 11, 7)

        result, parsed = client.invoke("system", "frage")

        assert result.content == "Antwort"
        assert result.input_tokens == 11
        assert result.output_tokens == 7
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert parsed is None

    def test_invoke_sends_system_and_user_messages(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _completion("ok")

        client.invoke("be brief", "hallo")

        kwargs = client._client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "unsloth/Qwen3.6-35B-A3B-MLX-8bit"
        assert kwargs["max_tokens"] == 256
        assert kwargs["stream"] is False
        assert kwargs["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hallo"},
        ]

    def test_invoke_appends_format_instructions_for_schema(self):
        from pydantic import BaseModel

        class Out(BaseModel):
            value: int

        client = _make_client()
        client._client.chat.completions.create.return_value = _completion('{"value": 5}')

        result, parsed = client.invoke("sys", "extract", output_schema=Out)

        user_msg = client._client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
        assert "Output Format" in user_msg
        assert parsed is not None and parsed.value == 5

    def test_supports_temp_false_omits_temperature(self):
        client = _make_client()
        client._supports_temp = False
        client._client.chat.completions.create.return_value = _completion("ok")

        client.invoke("sys", "user")

        assert "temperature" not in client._client.chat.completions.create.call_args.kwargs


class TestStream:
    def test_stream_yields_content_and_captures_usage(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = iter(
            [_content_chunk("Hallo "), _content_chunk("Welt"), _usage_chunk(13, 9)]
        )

        chunks = list(client.stream("system", "frage"))

        assert chunks == ["Hallo ", "Welt"]
        assert client.last_stream_usage.input_tokens == 13
        assert client.last_stream_usage.output_tokens == 9
        assert client.last_stream_usage.cache_creation_input_tokens == 0

    def test_stream_drops_reasoning_content(self):
        """Qwen3 reasoning deltas (reasoning_content, content=None) must not
        leak into the user-visible stream."""
        client = _make_client()
        client._client.chat.completions.create.return_value = iter(
            [
                _content_chunk(reasoning="Der Nutzer fragt..."),
                _content_chunk("finale "),
                _content_chunk("Antwort"),
            ]
        )

        chunks = list(client.stream("system", "frage"))

        assert chunks == ["finale ", "Antwort"]

    def test_stream_requests_usage_in_options(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = iter([_content_chunk("x")])

        list(client.stream("system", "frage"))

        kwargs = client._client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}


def _tc(call_id, name, arguments_raw):
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments_raw)
    )


def _tool_completion(tool_calls, content=None, finish_reason="tool_calls", p=10, c=4):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c),
    )


WEATHER_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


class TestForcedToolCall:
    def test_invoke_with_tool_returns_parsed_arguments(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion(
            [_tc("call_1", "get_weather", '{"city": "Berlin"}')]
        )

        result, tool_input = client.invoke_with_tool(
            "sys", "weather?", tool_name="get_weather", tool_description="d", tool_input_schema=WEATHER_SCHEMA
        )

        assert tool_input == {"city": "Berlin"}
        assert result.input_tokens == 10 and result.output_tokens == 4
        assert '"city": "Berlin"' in result.content

    def test_invoke_with_tool_sends_forced_tool_choice(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion(
            [_tc("c1", "get_weather", "{}")]
        )

        client.invoke_with_tool(
            "sys", "u", tool_name="get_weather", tool_description="d", tool_input_schema=WEATHER_SCHEMA
        )

        kwargs = client._client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tools"][0]["function"]["parameters"] == WEATHER_SCHEMA

    def test_invoke_with_tool_raises_when_no_tool_call(self):
        from ai_router.cached_llm import ForcedToolUseError

        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion(
            [], content="ich kann das nicht", finish_reason="stop"
        )

        with pytest.raises(ForcedToolUseError):
            client.invoke_with_tool(
                "sys", "u", tool_name="get_weather", tool_description="d", tool_input_schema=WEATHER_SCHEMA
            )

    def test_invoke_with_tool_malformed_arguments_yields_empty_dict(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion(
            [_tc("c1", "get_weather", "{not json")]
        )

        result, tool_input = client.invoke_with_tool(
            "sys", "u", tool_name="get_weather", tool_description="d", tool_input_schema=WEATHER_SCHEMA
        )

        assert tool_input == {}


class TestAutoToolCalls:
    def test_invoke_tools_returns_parsed_calls_and_assistant_message(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion(
            [_tc("call_42", "get_weather", '{"city": "Hamburg"}')]
        )

        out = client.invoke_tools(
            "sys",
            "weather in Hamburg?",
            tools=[{"name": "get_weather", "description": "d", "parameters": WEATHER_SCHEMA}],
        )

        assert out.finish_reason == "tool_calls"
        assert len(out.tool_calls) == 1
        call = out.tool_calls[0]
        assert call.id == "call_42"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Hamburg"}
        # assistant_message is ready to append for continuation
        assert out.assistant_message["role"] == "assistant"
        assert out.assistant_message["tool_calls"][0]["function"]["arguments"] == '{"city": "Hamburg"}'

    def test_invoke_tools_defaults_to_auto_choice_and_normalizes_shorthand(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion([], content="hi", finish_reason="stop")

        client.invoke_tools("sys", "hi", tools=[{"name": "t", "description": "d", "parameters": {}}])

        kwargs = client._client.chat.completions.create.call_args.kwargs
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["tools"][0] == {"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}

    def test_invoke_tools_passes_full_tool_dict_through(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion([], content="x")
        full = {"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}

        client.invoke_tools("sys", "u", tools=[full], tool_choice="required")

        kwargs = client._client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"][0] is full
        assert kwargs["tool_choice"] == "required"

    def test_invoke_tools_no_calls_returns_text_only(self):
        client = _make_client()
        client._client.chat.completions.create.return_value = _tool_completion(
            [], content="just text", finish_reason="stop"
        )

        out = client.invoke_tools("sys", "u", tools=[{"name": "t", "description": "d", "parameters": {}}])

        assert out.tool_calls == []
        assert out.content == "just text"
        assert "tool_calls" not in out.assistant_message


class TestToolHelpers:
    def test_function_tool_shape(self):
        from ai_router.local_client import function_tool

        assert function_tool("n", "desc", {"type": "object"}) == {
            "type": "function",
            "function": {"name": "n", "description": "desc", "parameters": {"type": "object"}},
        }

    def test_tool_result_message_shape(self):
        from ai_router.local_client import tool_result_message

        assert tool_result_message("call_7", "22°C") == {
            "role": "tool",
            "tool_call_id": "call_7",
            "content": "22°C",
        }


class TestUnsupportedPaths:
    def test_pdf_cache_raises_not_implemented(self):
        client = _make_client()
        with pytest.raises(NotImplementedError):
            client.invoke_with_pdf_cache(pdf_bytes=b"%PDF-", system_prompt="s", extraction_prompt="e")

    def test_pdf_tool_raises_not_implemented(self):
        client = _make_client()
        with pytest.raises(NotImplementedError):
            client.invoke_with_pdf_tool(
                pdf_bytes=b"%PDF-",
                system_prompt="s",
                user_prompt="u",
                tool_name="t",
                tool_description="d",
                tool_input_schema={},
            )
