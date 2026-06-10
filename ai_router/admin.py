from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html

from ai_router.models import LLMLog


class LLMLogAdmin(admin.ModelAdmin):
    """Admin for LLM Log entries."""

    list_display = [
        "id",
        "created_at",
        "agent_name",
        "model",
        "status_badge",
        "duration_display",
        "tokens_display",
        "project_link",
        "user",
    ]
    list_filter = [
        "status",
        "agent_name",
        "model",
        "created_at",
    ]
    search_fields = [
        "agent_name",
        "model",
        "system_prompt",
        "user_prompt",
        "output",
        "error",
        "project__name",
        "user__username",
    ]
    readonly_fields = [
        "created_at",
        "updated_at",
        "agent_name",
        "model",
        "project",
        "user",
        "system_prompt",
        "user_prompt",
        "output",
        "status",
        "error",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ]
    ordering = ["-created_at"]
    date_hierarchy = "created_at"

    fieldsets = [
        (
            "Uebersicht",
            {
                "fields": [
                    "created_at",
                    "updated_at",
                    "agent_name",
                    "model",
                    "status",
                    "duration_ms",
                ],
            },
        ),
        (
            "Kontext",
            {
                "fields": [
                    "project",
                    "user",
                ],
            },
        ),
        (
            "System Prompt",
            {
                "fields": ["system_prompt"],
                "classes": ["collapse"],
            },
        ),
        (
            "User Prompt",
            {
                "fields": ["user_prompt"],
                "classes": ["collapse"],
            },
        ),
        (
            "Output",
            {
                "fields": ["output"],
                "classes": ["collapse"],
            },
        ),
        (
            "Fehler",
            {
                "fields": ["error"],
                "classes": ["collapse"],
            },
        ),
        (
            "Token-Statistik",
            {
                "fields": [
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                ],
                "classes": ["collapse"],
            },
        ),
    ]

    def status_badge(self, obj):
        """Display status as colored badge."""
        colors = {
            LLMLog.Status.PENDING: "#ffc107",  # yellow
            LLMLog.Status.SUCCESS: "#28a745",  # green
            LLMLog.Status.ERROR: "#dc3545",  # red
        }
        color = colors.get(obj.status, "#6c757d")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; '
            'border-radius: 3px; font-size: 11px;">{}</span>',
            color,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"
    status_badge.admin_order_field = "status"

    def duration_display(self, obj):
        """Display duration in human-readable format."""
        if obj.duration_ms is None:
            return "-"
        if obj.duration_ms < 1000:
            return f"{obj.duration_ms}ms"
        return f"{obj.duration_ms / 1000:.1f}s"

    duration_display.short_description = "Dauer"
    duration_display.admin_order_field = "duration_ms"

    def tokens_display(self, obj):
        """Display tokens as IN / CACHED / OUT."""
        if obj.input_tokens is None and obj.output_tokens is None:
            return "-"
        input_t = f"{obj.input_tokens or 0:,}"
        output_t = f"{obj.output_tokens or 0:,}"
        cached = (obj.cache_read_input_tokens or 0) + (obj.cache_creation_input_tokens or 0)
        return format_html(
            '<span style="color: #17a2b8;">{}</span> / '
            '<span style="color: #ffc107;">{}</span> / '
            '<span style="color: #28a745;">{}</span>',
            input_t,
            f"{cached:,}",
            output_t,
        )

    tokens_display.short_description = "Tokens (IN/CACHED/OUT)"

    def project_link(self, obj):
        """Display project as link using reverse() for proper URL generation."""
        if obj.project:
            meta = obj.project._meta
            url = reverse(f"admin:{meta.app_label}_{meta.model_name}_change", args=[obj.project.id])
            return format_html(
                '<a href="{}">{}</a>',
                url,
                obj.project.name[:30],
            )
        return "-"

    project_link.short_description = "Project"

    def has_add_permission(self, request):
        """Disable adding logs manually."""
        return False

    def has_change_permission(self, request, obj=None):
        """Disable editing logs."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Allow deleting old logs."""
        return True
