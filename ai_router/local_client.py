"""
Local-model client for OpenAI-compatible servers (vllm-mlx / vLLM / llama.cpp).

Talks to a local `vllm-mlx serve ...` endpoint through the OpenAI SDK
(`/v1/chat/completions`). Same text-path surface as CachedAnthropicClient /
CachedGeminiClient (invoke / stream / invoke_with_cache / invoke_raw_cached)
so get_cached_client() can route to it transparently.

Local models have no prompt-cache billing — the cache_* token fields are
always 0. Reasoning models (e.g. Qwen3 with `--reasoning-parser qwen3`)
surface their thoughts in a separate `reasoning_content` field; we keep the
final answer (`content`) and drop the reasoning from the yielded text.

Example servers this targets:
    vllm-mlx serve unsloth/Qwen3.6-35B-A3B-MLX-8bit --port 8001 ...
    vllm-mlx serve mlx-community/gemma-4-12B-it-8bit --port 8002 ...
"""

import json
import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Type

from django.conf import settings
from pydantic import BaseModel

from ai_router.cached_llm import CachedInvocationResult, ForcedToolUseError
from ai_router.parsers import PydanticOutputParser

logger = logging.getLogger(__name__)

# Local OpenAI-compatible models served by vllm-mlx. Each entry maps a short
# alias to the served model id + its base_url. Hosts override/extend this via
# settings.AI_ROUTER_LOCAL_MODELS (merged by alias at construction time) so
# ports/URLs aren't baked into the package.
LOCAL_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "qwen3.6-35b-a3b": {
        "engine": "openai",
        "model_id": "unsloth/Qwen3.6-35B-A3B-MLX-8bit",
        "base_url": "http://localhost:8001/v1",
        "supports_temp": True,
        "reasoning": True,
    },
    "gemma-4-12b-it": {
        "engine": "openai",
        "model_id": "mlx-community/gemma-4-12B-it-8bit",
        "base_url": "http://localhost:8002/v1",
        "supports_temp": True,
        "reasoning": False,
    },
}

_DEFAULT_LOCAL_MAX_OUTPUT_TOKENS = 4096


@dataclass
class ToolCall:
    """A single tool call requested by the model (OpenAI function-calling)."""

    id: str
    name: str
    arguments: Dict[str, Any]
    arguments_raw: str


@dataclass
class ToolCallResult:
    """Result of a tools-enabled invocation (auto/required tool_choice).

    ``assistant_message`` is the raw assistant turn (with tool_calls) ready to
    append back to ``messages`` in an agentic loop — append it, then append a
    ``tool_result_message(...)`` per tool call, then call again.
    """

    content: str
    tool_calls: List[ToolCall]
    finish_reason: Optional[str]
    usage: CachedInvocationResult
    assistant_message: Dict[str, Any] = field(default_factory=dict)


def function_tool(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Build an OpenAI ``tools`` entry from a JSON-Schema parameter object."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def tool_result_message(tool_call_id: str, content: str) -> Dict[str, Any]:
    """Build the ``role=tool`` message that returns a tool result to the model."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _normalize_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either a full OpenAI tool dict (has ``type``) or the shorthand
    ``{name, description, parameters}`` and return the OpenAI shape."""
    if "type" in tool and "function" in tool:
        return tool
    return function_tool(
        tool["name"],
        tool.get("description", ""),
        tool.get("parameters") or tool.get("input_schema") or {},
    )


def _parse_tool_calls(message) -> List[ToolCall]:
    raw_calls = getattr(message, "tool_calls", None) or []
    parsed: List[ToolCall] = []
    for tc in raw_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        args_raw = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            decoded = json.loads(args_raw) if args_raw else {}
        except (json.JSONDecodeError, TypeError):
            decoded = {}
        parsed.append(
            ToolCall(
                id=getattr(tc, "id", "") or "",
                name=name or "",
                arguments=decoded if isinstance(decoded, dict) else {},
                arguments_raw=args_raw or "",
            )
        )
    return parsed


def _assistant_message_dict(message, tool_calls: List[ToolCall]) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "assistant", "content": getattr(message, "content", None) or ""}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments_raw}}
            for tc in tool_calls
        ]
    return msg


def resolve_local_config(model: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the merged local-model config for ``model``, or None if it is
    not a known local alias. settings.AI_ROUTER_LOCAL_MODELS (optional dict
    keyed by alias) overrides/extends the built-in LOCAL_MODEL_CONFIG."""
    if not model:
        return None
    overrides = getattr(settings, "AI_ROUTER_LOCAL_MODELS", None) or {}
    if model in overrides:
        merged = dict(LOCAL_MODEL_CONFIG.get(model, {}))
        merged.update(overrides[model])
        return merged
    if model in LOCAL_MODEL_CONFIG:
        return dict(LOCAL_MODEL_CONFIG[model])
    return None


class CachedLocalClient:
    """OpenAI-compatible client for local models served by vllm-mlx."""

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ):
        config = resolve_local_config(model) or {}
        self._alias = model
        self.model = config.get("model_id", model)
        self.base_url = config.get("base_url") or getattr(
            settings, "AI_ROUTER_LOCAL_BASE_URL", "http://localhost:8001/v1"
        )
        self.temperature = temperature
        self._supports_temp = config.get("supports_temp", True)
        self._reasoning = config.get("reasoning", False)
        # Settings runtime-read im Body (nicht als Default-Arg) damit
        # override_settings greift, analog zu CachedAnthropicClient.
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(settings, "AI_ROUTER_LOCAL_MAX_OUTPUT_TOKENS", _DEFAULT_LOCAL_MAX_OUTPUT_TOKENS)
        )
        # Phase B5 parity — usage des letzten stream()-Calls oder None.
        self.last_stream_usage = None
        self._client = None
        self._setup_client()

    def _setup_client(self):
        from openai import OpenAI

        # Local servers usually don't check the key, but the SDK requires a
        # non-empty value. Host can override via AI_ROUTER_LOCAL_API_KEY.
        api_key = getattr(settings, "AI_ROUTER_LOCAL_API_KEY", "") or "local"
        self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        logger.info("CachedLocalClient initialized (model=%s, base_url=%s)", self.model, self.base_url)

    @property
    def log_model(self) -> str:
        """Model name for logging — returns alias (e.g. 'qwen3.6-35b-a3b')."""
        return self._alias or self.model

    def _create_kwargs(self, messages: list, *, stream: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self._supports_temp:
            kwargs["temperature"] = self.temperature
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    @staticmethod
    def _messages(system_prompt: str, user_text: str) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _content_of(response) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        return getattr(message, "content", None) or ""

    @staticmethod
    def _usage_of(response) -> SimpleNamespace:
        usage = getattr(response, "usage", None)
        return SimpleNamespace(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def _complete(
        self,
        system_prompt: str,
        user_text: str,
        output_schema: Optional[Type[BaseModel]],
    ) -> Tuple[CachedInvocationResult, Optional[BaseModel]]:
        start_time = time.time()
        if output_schema:
            parser = PydanticOutputParser(pydantic_object=output_schema)
            user_text = f"{user_text}\n\n## Output Format:\n{parser.get_format_instructions()}"
        try:
            response = self._client.chat.completions.create(
                **self._create_kwargs(self._messages(system_prompt, user_text), stream=False)
            )
            content = self._content_of(response)
            usage = self._usage_of(response)
            result = CachedInvocationResult(
                content=content,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                duration_ms=int((time.time() - start_time) * 1000),
                model=self.model,
            )
            parsed_output = None
            if output_schema and content:
                try:
                    parsed_output = PydanticOutputParser(pydantic_object=output_schema).parse(content)
                except Exception as parse_error:  # noqa: BLE001
                    logger.warning(f"Failed to parse local-model output: {parse_error}")
            return result, parsed_output
        except Exception as e:
            logger.error(f"Local invoke failed (model={self.log_model}): {e}")
            raise

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Tuple[CachedInvocationResult, Optional[BaseModel]]:
        """Simple text-only invoke — drop-in replacement for chain.invoke()."""
        return self._complete(system_prompt, user_prompt, output_schema)

    def stream(
        self,
        system_prompt: str,
        user_prompt: str = "",
        messages: list[dict] | None = None,
    ):
        """Streaming text invoke — yields content chunks (reasoning dropped).

        Pass ``messages`` (role/content dicts) for multi-turn chat. Captures
        token usage from the final usage-only chunk into last_stream_usage.
        """
        chat_messages = messages or self._messages(system_prompt, user_prompt)
        self.last_stream_usage = None
        response = self._client.chat.completions.create(
            **self._create_kwargs(chat_messages, stream=True)
        )
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    yield text
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_stream_usage = SimpleNamespace(
                    input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                )

    def invoke_with_cache(
        self,
        document_content: str,
        document_name: str,
        extraction_prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        document_tag: Optional[str] = None,
    ) -> Tuple[CachedInvocationResult, Optional[BaseModel]]:
        """Document extraction — drop-in parity with the Anthropic/Gemini
        clients. Local models have no prompt cache; the document goes into the
        system prompt and the extraction prompt into the user message."""
        system_prompt = _document_system_prompt(document_content, document_name, document_tag)
        return self._complete(system_prompt, extraction_prompt, output_schema)

    def invoke_raw_cached(
        self,
        system_text: str,
        user_prompt: str,
    ) -> CachedInvocationResult:
        """Invoke with a custom system prompt (no document template, no cache)."""
        result, _ = self._complete(system_text, user_prompt, None)
        return result

    def invoke_with_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        tool_input_schema: Dict[str, Any],
        max_tokens: Optional[int] = None,
        messages: list[dict] | None = None,
    ) -> Tuple[CachedInvocationResult, Dict[str, Any]]:
        """Forced single-tool call → structured dict.

        OpenAI/vLLM analogue of CachedAnthropicClient.invoke_with_pdf_tool:
        sets ``tool_choice={"type":"function","function":{"name":tool_name}}``
        so the model MUST emit a tool call for ``tool_name``. Returns
        (CachedInvocationResult, tool_input dict); ``result.content`` is the
        JSON-serialized tool_input (truncated to 4000 chars) for audit.

        Raises ForcedToolUseError if the response carries no matching tool
        call (e.g. truncated mid-arguments)."""
        start_time = time.time()
        chat_messages = messages or self._messages(system_prompt, user_prompt)
        create_kwargs = self._create_kwargs(chat_messages, stream=False)
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        create_kwargs["tools"] = [function_tool(tool_name, tool_description, tool_input_schema)]
        create_kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_name}}

        response = self._client.chat.completions.create(**create_kwargs)
        choice = self._first_choice(response)
        message = getattr(choice, "message", None) if choice is not None else None
        tool_calls = _parse_tool_calls(message) if message is not None else []
        usage = self._usage_of(response)

        tool_input: Optional[Dict[str, Any]] = None
        for tc in tool_calls:
            if tc.name == tool_name:
                tool_input = tc.arguments
                break

        content_repr = json.dumps(tool_input, default=str)[:4000] if tool_input is not None else ""
        result = CachedInvocationResult(
            content=content_repr,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            duration_ms=int((time.time() - start_time) * 1000),
            model=self.model,
        )
        if tool_input is None:
            finish = getattr(choice, "finish_reason", "?") if choice is not None else "?"
            raise ForcedToolUseError(
                f"Local response has no tool call for {tool_name!r} "
                f"(finish_reason={finish}) despite forced tool_choice",
                raw_content=(getattr(message, "content", None) or "")[:2000] if message is not None else "",
            )
        return result, tool_input

    def invoke_tools(
        self,
        system_prompt: str,
        user_prompt: str = "",
        *,
        tools: list,
        tool_choice: Any = "auto",
        messages: list[dict] | None = None,
        max_tokens: Optional[int] = None,
    ) -> ToolCallResult:
        """Multi-tool invocation for agentic loops.

        ``tools`` entries may be full OpenAI tool dicts or the shorthand
        ``{name, description, parameters}``. ``tool_choice`` is "auto"
        (default), "none", "required", or a forced ``{"type":"function",...}``.

        Returns a ToolCallResult with the assistant text, parsed tool_calls
        and the assistant_message to append before feeding tool results back
        via tool_result_message()."""
        start_time = time.time()
        chat_messages = messages or self._messages(system_prompt, user_prompt)
        create_kwargs = self._create_kwargs(chat_messages, stream=False)
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        create_kwargs["tools"] = [_normalize_tool(t) for t in tools]
        create_kwargs["tool_choice"] = tool_choice

        response = self._client.chat.completions.create(**create_kwargs)
        choice = self._first_choice(response)
        message = getattr(choice, "message", None) if choice is not None else None
        tool_calls = _parse_tool_calls(message) if message is not None else []
        usage = self._usage_of(response)
        content = (getattr(message, "content", None) or "") if message is not None else ""

        result = CachedInvocationResult(
            content=content,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            duration_ms=int((time.time() - start_time) * 1000),
            model=self.model,
        )
        return ToolCallResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
            usage=result,
            assistant_message=_assistant_message_dict(message, tool_calls) if message is not None else {},
        )

    @staticmethod
    def _first_choice(response):
        choices = getattr(response, "choices", None) or []
        return choices[0] if choices else None

    def invoke_with_pdf_cache(self, *args, **kwargs):
        raise NotImplementedError(
            "CachedLocalClient is text-only — local vllm-mlx servers do not accept "
            "PDF document blocks. Use a Bedrock/Vertex/Gemini model for PDF extraction."
        )

    def invoke_with_pdf_tool(self, *args, **kwargs):
        raise NotImplementedError(
            "CachedLocalClient does not support invoke_with_pdf_tool (PDF + forced "
            "tool_choice). Use a Bedrock/Vertex model for that path."
        )


def _document_system_prompt(
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
