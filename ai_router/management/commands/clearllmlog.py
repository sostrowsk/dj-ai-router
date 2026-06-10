from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils.timezone import now

from ai_router.models import LLMLog


class Command(BaseCommand):
    help = "Delete LLM log entries older than the specified number of days (default: 30)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Delete logs older than this many days (default: 30).",
        )

    def handle(self, *args, **options):
        days = options["days"]
        cutoff_date = now() - timedelta(days=days)

        old_logs = LLMLog.objects.filter(created_at__lt=cutoff_date)
        count = old_logs.count()

        if count > 0:
            deleted_count, _ = old_logs.delete()
            self.stdout.write(f"Deleted {deleted_count} LLM log entries older than {days} days.")
        else:
            self.stdout.write(f"No LLM log entries older than {days} days found.")
