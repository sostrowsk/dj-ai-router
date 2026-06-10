"""Move LLMLog from ai_agents to ai_router (state-only, no DB changes)."""

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("project", "0068_project_language"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name="LLMLog",
                    fields=[
                        (
                            "id",
                            models.BigAutoField(
                                auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                            ),
                        ),
                        ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                        ("updated_at", models.DateTimeField(auto_now=True)),
                        (
                            "agent_name",
                            models.CharField(
                                db_index=True, help_text="Name des Agents (z.B. 'guv_extractor')", max_length=100
                            ),
                        ),
                        (
                            "model",
                            models.CharField(db_index=True, help_text="LLM Model (z.B. 'gpt-5')", max_length=100),
                        ),
                        ("system_prompt", models.TextField(help_text="System-Prompt mit Rolle und Regeln")),
                        ("user_prompt", models.TextField(help_text="User-Prompt mit Aufgabe und Dokumenten")),
                        ("output", models.TextField(blank=True, default="", help_text="Antwort des LLM")),
                        (
                            "status",
                            models.CharField(
                                choices=[("pending", "Pending"), ("success", "Success"), ("error", "Error")],
                                db_index=True,
                                default="pending",
                                max_length=20,
                            ),
                        ),
                        ("error", models.TextField(blank=True, default="", help_text="Fehlermeldung bei Status=error")),
                        (
                            "duration_ms",
                            models.IntegerField(blank=True, help_text="Dauer des Aufrufs in Millisekunden", null=True),
                        ),
                        (
                            "input_tokens",
                            models.IntegerField(blank=True, help_text="Anzahl der Input-Tokens", null=True),
                        ),
                        (
                            "output_tokens",
                            models.IntegerField(blank=True, help_text="Anzahl der Output-Tokens", null=True),
                        ),
                        ("retry_count", models.IntegerField(default=0, help_text="Anzahl der Wiederholungsversuche")),
                        (
                            "project",
                            models.ForeignKey(
                                blank=True,
                                null=True,
                                on_delete=django.db.models.deletion.SET_NULL,
                                related_name="llm_logs",
                                to="project.project",
                            ),
                        ),
                        (
                            "user",
                            models.ForeignKey(
                                blank=True,
                                null=True,
                                on_delete=django.db.models.deletion.SET_NULL,
                                related_name="llm_logs",
                                to=settings.AUTH_USER_MODEL,
                            ),
                        ),
                    ],
                    options={
                        "verbose_name": "LLM Log",
                        "verbose_name_plural": "LLM Logs",
                        "ordering": ["-created_at"],
                        "db_table": "ai_agents_llmlog",
                        "indexes": [
                            models.Index(fields=["agent_name", "status"], name="ai_agents_l_agent_n_9be473_idx"),
                            models.Index(fields=["created_at", "status"], name="ai_agents_l_created_05b6c3_idx"),
                        ],
                    },
                ),
            ],
            database_operations=[],
        ),
    ]
