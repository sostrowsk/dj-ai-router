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

Installed by the host project as a Poetry git dependency (single lock
authority lives in the host):

```toml
[tool.poetry.dependencies]
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

Model routing: `get_cached_client(model)` picks `CachedGeminiClient` for
models in `VERTEX_MODEL_CONFIG` with `engine == "gemini"`, otherwise
`CachedAnthropicClient` (Bedrock, or Vertex when `GCP_PROJECT_ID` is set).
Per-provider config dicts: `ai_router.bedrock_client.BEDROCK_MODEL_CONFIG`,
`ai_router.azure_client.AZURE_MODEL_CONFIG`,
`ai_router.vertex_client.VERTEX_MODEL_CONFIG`.

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

- Local override in the host:
  `poetry run pip install -e ../dj-ai-router`
  (note: a later `poetry install` in the host resets to the locked git ref).
- Release: commit + push to `main`, then in the host
  `poetry update dj-ai-router`.
