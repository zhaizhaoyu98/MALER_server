from __future__ import absolute_import

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Audit deployment-sensitive MALER/Django settings without exposing secrets."

    def add_arguments(self, parser):
        parser.add_argument("--strict", action="store_true", help="Fail if any high-risk check fails")

    def handle(self, *args, **options):
        signing_key = str(getattr(settings, "MALER_MODEL_SIGNING_KEY", "") or "")
        secret_key = str(getattr(settings, "SECRET_KEY", "") or "")
        hosts = list(getattr(settings, "ALLOWED_HOSTS", []))
        checks = [
            ("DEBUG disabled", not bool(settings.DEBUG), "high"),
            ("non-default Django secret", bool(secret_key) and secret_key != "change-me-in-production", "high"),
            ("model signing key at least 32 characters", len(signing_key) >= 32, "high"),
            ("explicit hosts without wildcard", bool(hosts) and "*" not in hosts, "high"),
            ("HTTPS redirect enabled", bool(getattr(settings, "SECURE_SSL_REDIRECT", False)), "high"),
            ("secure session cookie", bool(getattr(settings, "SESSION_COOKIE_SECURE", False)), "high"),
            ("secure CSRF cookie", bool(getattr(settings, "CSRF_COOKIE_SECURE", False)), "high"),
            ("HSTS enabled", int(getattr(settings, "SECURE_HSTS_SECONDS", 0)) > 0, "medium"),
            ("legacy user pickle disabled", not bool(getattr(settings, "MALER_ALLOW_LEGACY_MODEL_UPLOAD", True)), "high"),
            ("positive cache retention", float(getattr(settings, "MALER_CACHE_RETENTION_HOURS", 0)) > 0, "medium"),
        ]
        failed_high = []
        for name, passed, severity in checks:
            status = "PASS" if passed else "FAIL"
            self.stdout.write("%s\t%s\t%s" % (status, severity, name))
            if not passed and severity == "high":
                failed_high.append(name)
        self.stdout.write("Summary: %d/%d checks passed." % (
            sum(1 for _, passed, _ in checks if passed), len(checks)))
        if options["strict"] and failed_high:
            raise CommandError("High-risk deployment checks failed: %s" % "; ".join(failed_high))
