"""
LLM Log Model - Protokolliert jeden LLM-Aufruf.

Speichert SystemPrompt, UserPrompt, Output und ggf. Fehlermeldungen
fuer Debugging, Auditing und Analyse.
"""

from django.conf import settings
from django.db import models


class LLMLog(models.Model):
    """
    Protokolliert jeden Aufruf an ein LLM.

    Speichert:
    - system_prompt: Der System-Prompt (Rolle, Regeln)
    - user_prompt: Der User-Prompt (Aufgabe, Dokumente)
    - output: Die Antwort des LLM
    - error: Fehlermeldung falls der Aufruf fehlgeschlagen ist
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SUCCESS = "success", "Success"
        ERROR = "error", "Error"

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Context
    agent_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Name des Agents (z.B. 'guv_extractor')",
    )
    model = models.CharField(
        max_length=100,
        db_index=True,
        help_text="LLM Model (z.B. 'gpt-5')",
    )
    project = models.ForeignKey(
        getattr(settings, "AI_ROUTER_PROJECT_MODEL", "project.Project"),
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="llm_logs",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="llm_logs",
    )

    # Prompts
    system_prompt = models.TextField(
        help_text="System-Prompt mit Rolle und Regeln",
    )
    user_prompt = models.TextField(
        help_text="User-Prompt mit Aufgabe und Dokumenten",
    )

    # Response
    output = models.TextField(
        blank=True,
        default="",
        help_text="Antwort des LLM",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    error = models.TextField(
        blank=True,
        default="",
        help_text="Fehlermeldung bei Status=error",
    )

    # Metadata
    duration_ms = models.IntegerField(
        null=True,
        blank=True,
        help_text="Dauer des Aufrufs in Millisekunden",
    )
    input_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text="Anzahl der Input-Tokens",
    )
    output_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text="Anzahl der Output-Tokens",
    )
    cache_read_input_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text="Tokens aus dem Prompt-Cache gelesen",
    )
    cache_creation_input_tokens = models.IntegerField(
        null=True,
        blank=True,
        help_text="Tokens neu in den Prompt-Cache geschrieben",
    )

    class Meta:
        verbose_name = "LLM Log"
        verbose_name_plural = "LLM Logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["agent_name", "status"]),
            models.Index(fields=["created_at", "status"]),
        ]

    def __str__(self):
        return f"{self.agent_name} ({self.model}) - {self.status} - {self.created_at:%Y-%m-%d %H:%M}"

    @property
    def is_success(self):
        return self.status == self.Status.SUCCESS

    @property
    def is_error(self):
        return self.status == self.Status.ERROR

    @property
    def prompt_preview(self):
        """Kurze Vorschau des User-Prompts."""
        return self.user_prompt[:200] + "..." if len(self.user_prompt) > 200 else self.user_prompt

    @property
    def output_preview(self):
        """Kurze Vorschau des Outputs."""
        return self.output[:200] + "..." if len(self.output) > 200 else self.output
