# dj-ai-router

Unified multi-provider LLM client factory for Django projects: Anthropic
Claude via AWS Bedrock and Google Vertex, Azure OpenAI (GPT) and Google
Gemini (google-genai). Provides prompt caching (`CachedAnthropicClient`,
`CachedGeminiClient`), streaming with resilience/fallback, 429/throttling
retry handling, safe-markdown rendering and per-call DB logging (`LLMLog`)
with admin.

Python package name: **`ai_router`** (the repo name `dj-ai-router` is only
the distribution name — app label, import path, DB tables and migrations
stay `ai_router`).

## Installation

Installed by the host project as a uv git dependency (single lock
authority lives in the host):

```toml
[project]
dependencies = [
    "dj-ai-router",
]

[tool.uv.sources]
dj-ai-router = { git = "ssh://git@github.com/sostrowsk/dj-ai-router.git", branch = "main" }
```

```python
INSTALLED_APPS = [
    ...
    "ai_router",
]
```

No URLs, no templates, no static files — the app ships models, admin,
clients and a management command (`clearllmlog`).

## Usage

```python
from ai_router import get_llm_client

client = get_llm_client()  # DEFAULT_MODEL_BEDROCK
result, parsed = client.invoke(system_prompt, user_prompt, output_schema=MySchema)

for chunk in client.stream(system_prompt, user_prompt):
    ...

result, parsed = client.invoke_with_pdf_cache(pdf_path, system_prompt, user_prompt)
```

Model routing: `get_cached_client(model)` picks `CachedLocalClient` for
models in `LOCAL_MODEL_CONFIG` (local vllm-mlx), then `CachedGeminiClient`
for models in `VERTEX_MODEL_CONFIG` with `engine == "gemini"`, otherwise
`CachedAnthropicClient` (Bedrock, or Vertex when `GCP_PROJECT_ID` is set).
Per-provider config dicts: `ai_router.bedrock_client.BEDROCK_MODEL_CONFIG`,
`ai_router.azure_client.AZURE_MODEL_CONFIG`,
`ai_router.vertex_client.VERTEX_MODEL_CONFIG`,
`ai_router.local_client.LOCAL_MODEL_CONFIG`.

### Local models (vllm-mlx / OpenAI-compatible)

`CachedLocalClient` talks to a local OpenAI-compatible server over the
`openai` SDK (`/v1/chat/completions`). Built-in aliases in
`LOCAL_MODEL_CONFIG`:

| Alias | Served model | Default base_url |
| --- | --- | --- |
| `qwen3.6-35b-a3b` | `unsloth/Qwen3.6-35B-A3B-MLX-8bit` | `http://localhost:8001/v1` |
| `gemma-4-12b-it` | `mlx-community/gemma-4-12B-it-8bit` | `http://localhost:8002/v1` |

```bash
vllm-mlx serve unsloth/Qwen3.6-35B-A3B-MLX-8bit --port 8001 \
  --continuous-batching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen --max-request-tokens 131072
vllm-mlx serve mlx-community/gemma-4-12B-it-8bit --port 8002 --max-request-tokens 131072
```

```python
client = get_llm_client("qwen3.6-35b-a3b")
result, parsed = client.invoke(system_prompt, user_prompt, output_schema=MySchema)
for chunk in client.stream(system_prompt, user_prompt):
    ...
```

Text-only: `invoke` / `stream` / `invoke_with_cache` / `invoke_raw_cached`
work; `invoke_with_pdf_cache` / `invoke_with_pdf_tool` raise
`NotImplementedError` (local servers take no PDF document blocks). No prompt
cache — `cache_*` token fields are always 0. Reasoning models (Qwen3) expose
their thoughts in `reasoning_content`; the client returns only the final
`content` and drops reasoning from the streamed text.

#### Tool calling

For servers started with `--enable-auto-tool-choice --tool-call-parser ...`,
`CachedLocalClient` exposes OpenAI-style function calling:

```python
from ai_router.local_client import function_tool, tool_result_message

client = get_llm_client("qwen3.6-35b-a3b")

# 1. Forced single tool → structured dict (analogue of invoke_with_pdf_tool).
#    Raises ForcedToolUseError if the model emits no matching tool call.
result, args = client.invoke_with_tool(
    system_prompt, user_prompt,
    tool_name="extract_invoice",
    tool_description="Extract invoice fields",
    tool_input_schema={"type": "object", "properties": {...}, "required": [...]},
)

# 2. Multi-tool agentic loop (tool_choice "auto"|"required"|"none"|forced).
tools = [function_tool("get_weather", "Current weather", WEATHER_SCHEMA)]
messages = [{"role": "user", "content": "weather in Hamburg?"}]
out = client.invoke_tools(system_prompt, messages=messages, tools=tools)
for call in out.tool_calls:          # ToolCall(id, name, arguments, arguments_raw)
    answer = run(call.name, call.arguments)
    messages += [out.assistant_message, tool_result_message(call.id, answer)]
out = client.invoke_tools(system_prompt, messages=messages, tools=tools)  # continue
```

`tools` entries may be full OpenAI dicts or the shorthand
`{name, description, parameters}`. `invoke_tools` returns a `ToolCallResult`
(`content`, `tool_calls`, `finish_reason`, `usage`, `assistant_message`).
Tool calling is non-streaming. The Anthropic client's forced-tool path is
`invoke_with_pdf_tool` (PDF + forced `tool_choice`).

## Settings catalog

### Required (no defaults — `django.conf.settings` attribute access)

| Setting | Used for |
| --- | --- |
| `AUTH_USER_MODEL` | `LLMLog.user` FK (swappable) |
| `BASE_DIR` | resolving a relative `GCP_GEMINI_CREDENTIALS` path |
| `DEFAULT_MODEL_BEDROCK` | default model for `get_llm_client()` / `CachedAnthropicClient` |
| `DEFAULT_MODEL_VERTEX` | default model for `CachedGeminiClient` |
| `ANTHROPIC_SDK_MAX_RETRIES` | `max_retries` passed to the Anthropic SDK clients |
| `LLM_MAX_OUTPUT_TOKENS` | max output tokens for Anthropic calls |
| `GEMINI_MAX_OUTPUT_TOKENS` | max output tokens for Gemini calls |
| `AZURE_LEASING_API_KEY` | Azure OpenAI embeddings/encoder |
| `AZURE_EMBEDDINGS_BASE_URL` | Azure OpenAI embeddings/encoder |
| `AZURE_EMBEDDINGS_API_VERSION` | Azure OpenAI embeddings/encoder |
| `GCP_GEMINI_PROJECT_ID` | google-genai Vertex client |
| `GCP_GEMINI_REGION` | google-genai Vertex client |
| `GCP_GEMINI_API_KEY` | google-genai API-key auth (takes precedence) |
| `GCP_GEMINI_CREDENTIALS` | path to a service-account JSON (optional fallback, may be empty) |

### Optional (read via `getattr` with stable defaults)

| Setting | Default | Used for |
| --- | --- | --- |
| `AI_ROUTER_PROJECT_MODEL` | `"project.Project"` | `LLMLog.project` FK target (see migration note below) |
| `GCP_PROJECT_ID` | `""` | non-empty switches `CachedAnthropicClient` to `AnthropicVertex` |
| `GCP_REGION` | `"europe-west4"` | AnthropicVertex region |
| `AWS_BEDROCK_REGION` | `"eu-central-1"` | AnthropicBedrock region |
| `AWS_BEDROCK_ACCESS_KEY_ID` | `""` | empty → ambient AWS credentials |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | `""` | empty → ambient AWS credentials |
| `AI_ROUTER_LOCAL_MODELS` | `{}` | dict keyed by alias; overrides/extends `LOCAL_MODEL_CONFIG` (e.g. change `base_url`/port, add a new local model) |
| `AI_ROUTER_LOCAL_BASE_URL` | `"http://localhost:8001/v1"` | fallback base_url when an alias has none |
| `AI_ROUTER_LOCAL_API_KEY` | `"local"` | bearer key for the local server (most ignore it; the SDK just needs non-empty) |
| `AI_ROUTER_LOCAL_MAX_OUTPUT_TOKENS` | `4096` | default `max_tokens` for `CachedLocalClient` |

### Referenced indirectly via `AZURE_MODEL_CONFIG`

Entries in `AZURE_MODEL_CONFIG` name host settings by string
(`api_key` / `api_version` / `endpoint`), e.g. `AZURE_SHOOBRIDGE_API_KEY`,
`AZURE_CHAT_API_VERSION`, `AZURE_SHOOBRIDGE_BASE_URL`. Hosts that route GPT
models must define the settings named in the config entries they use.

## Migrations / foreign hosts (`MIGRATION_MODULES` note)

`ai_router/migrations/0001_initial.py` is pinned to the original host:
it depends on `("project", "0068_project_language")` and creates the
`LLMLog.project` FK against `project.Project` (state-only,
`SeparateDatabaseAndState`).

- For the original host (leasing) the shipped migrations are byte-stable —
  nothing to do.
- A foreign host **without** a `project` app (or with
  `AI_ROUTER_PROJECT_MODEL` overridden) cannot apply the shipped
  migrations. Point Django at host-owned migrations instead:

  ```python
  MIGRATION_MODULES = {"ai_router": "myhost.migrations_ai_router"}
  ```

  and generate them with `makemigrations ai_router`. Note that
  `AI_ROUTER_PROJECT_MODEL` is evaluated at module import time
  (module-level settings FK, like django-taggit) — `override_settings`
  has no effect on it.

## Peer requirements

None — dj-ai-router has no dependencies on other extracted dj-* packages.
(It is itself a peer of dj-rag-db, dj-ai-chat and dj-data-room.)

## Host contract

- Configured Django project providing the settings above.
- DB: the `LLMLog` table (`ai_router_llmlog`); admin is registered
  automatically when `django.contrib.admin` is installed. The admin
  `project_link` builds the change-URL from `obj.project._meta`, so a
  custom `AI_ROUTER_PROJECT_MODEL` works as long as the model is
  registered in the same admin site.
- Management command `clearllmlog` for log retention (host schedules it).

## Tests

Tests live in `ai_router/tests/` and run from the host via:

```bash
pytest --pyargs ai_router.tests
```

## Development workflow

- Local override in the host: add an editable path source in the host
  `pyproject.toml` (or `uv pip install -e ../dj-ai-router` into the host
  venv; note a later `uv sync` in the host resets to the locked git ref):

  ```toml
  [tool.uv.sources]
  dj-ai-router = { path = "../dj-ai-router", editable = true }
  ```

- Release: commit + push to `main`, then in the host
  `uv lock --upgrade-package dj-ai-router` (followed by `uv sync`).
