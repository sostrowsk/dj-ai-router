# dj-ai-router

Django app package `ai_router` (app label, import path, DB tabellen bleiben
`ai_router`). Host-Projekte pinnen dieses Repo als uv-git-Dependency
(`[tool.uv.sources]`) auf `main` — jeder Push auf main ist sofort
releasebar. Build-Backend: hatchling (PEP 621).

## Provider-Clients

`get_cached_client(model)` (in `cached_llm.py`) routet anhand von Config-
Dicts auf den passenden Client:

- **Bedrock/Vertex** (`CachedAnthropicClient`) — Anthropic Claude. Forced-
  tool-Pfad: `invoke_with_pdf_tool` (PDF + `tool_choice`).
- **Vertex/Gemini** (`CachedGeminiClient`) — `VERTEX_MODEL_CONFIG`,
  `engine == "gemini"`.
- **Lokal** (`CachedLocalClient`, `local_client.py`) — OpenAI-kompatible
  vllm-mlx-Server via `openai`-SDK. `LOCAL_MODEL_CONFIG` (per
  `settings.AI_ROUTER_LOCAL_MODELS` override-/erweiterbar). Text-only
  (`invoke`/`stream`/`invoke_with_cache`/`invoke_raw_cached`); PDF-Pfade
  raisen `NotImplementedError`. Tool-Calling (OpenAI function-calling):
  `invoke_with_tool` (forced single tool → dict) und `invoke_tools`
  (auto/required, agentic loop). Kein Prompt-Cache → `cache_*` immer 0;
  Reasoning-Modelle (Qwen3) liefern `reasoning_content` separat, nur
  `content` wird zurueckgegeben.

## TDD-Regeln (Pflicht)

- **Test zuerst, RED bestaetigen, dann implementieren, GREEN bestaetigen.**
- Bugfix = Regressionstest, der den Bug reproduziert und VOR dem Fix failt.
- Reine Moves: Import-Smoke-Tests (`tests/test_imports.py`).
- Tests laufen aus dem Host-Projekt: `pytest --pyargs ai_router.tests`
  (das Package hat keine eigene Settings-/pytest-Infrastruktur).
- LLM-/Netzwerk-Calls IMMER mocken — kein Test darf echte Provider-APIs
  (Bedrock/Vertex/Azure/Gemini) treffen.

## Architektur-Regeln

- Keine Imports aus Host-Apps (users, project, leasing, ai_agents, scribe,
  data_room, ...). FK-Targets nur via `settings.AUTH_USER_MODEL` bzw.
  getattr-Settings mit stabilen Defaults (`AI_ROUTER_PROJECT_MODEL`).
- **Migrations-Byte-Stabilitaet:** Aenderungen duerfen keine neuen
  Migrationen im Host erzeugen (`makemigrations --check --dry-run` muss im
  Host clean bleiben). Modul-Level-Settings-FKs nicht "dynamisieren".
- Settings-Katalog im README aktuell halten, wenn neue Settings dazukommen.
- Keine Peer-git-Deps in pyproject (nur der Host pinnt dj-* Packages).
