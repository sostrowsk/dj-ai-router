# dj-ai-router

Django app package `ai_router` (app label, import path, DB tabellen bleiben
`ai_router`). Host-Projekte pinnen dieses Repo als Poetry-git-Dependency auf
`main` — jeder Push auf main ist sofort releasebar.

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
