"""Reproducible configuration-level security audit for the legacy deployment stack.

This is deliberately not a penetration test or a live-host TLS audit.  It checks
that a simulated production process activates the controls supplied by MALER.
"""

from __future__ import print_function

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "validation" / "results" / "production_security_audit.json"
sys.path.insert(0, str(ROOT))


def configure_environment():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ML_WebServer.settings")
    os.environ.setdefault("DJANGO_DEBUG", "False")
    os.environ.setdefault("DJANGO_SECRET_KEY", "Q7m!2Zx#9Lp$4Vr%8Tk&1Ns*6Hw@3Bc+5Df=0Gj?7Ky^2Pa!9Ru#4Xe%8Wq&1Mv*6Ct@3")
    os.environ.setdefault("DJANGO_ALLOWED_HOSTS", "maler.example.org")
    os.environ.setdefault("DJANGO_SECURE_SSL", "True")
    os.environ.setdefault("DJANGO_HSTS_SECONDS", "31536000")
    os.environ.setdefault("MALER_MODEL_SIGNING_KEY", "K3p!8Vn#1Rs$6Yq%4Wm&9Cx*2Ht@7Bd+5Lf=0Gj?3Zu^8Ae!1Ti#6Oo%4Uk&9Iy*2Er@7")


def main():
    configure_environment()
    import django

    django.setup()
    from django.conf import settings
    from django.core.exceptions import SuspiciousOperation
    from mlserver.security import validated_project_id

    traversal_rejected = False
    try:
        validated_project_id("../outside")
    except SuspiciousOperation:
        traversal_rejected = True

    checks = [
        ("debug_disabled", settings.DEBUG is False, "DEBUG=False"),
        ("strong_nondefault_secret", len(settings.SECRET_KEY) >= 50 and len(set(settings.SECRET_KEY)) >= 5 and settings.SECRET_KEY != "change-me-in-production", "Secret is non-default, >=50 characters, and diverse"),
        ("restricted_allowed_hosts", bool(settings.ALLOWED_HOSTS) and "*" not in settings.ALLOWED_HOSTS, "ALLOWED_HOSTS is explicit"),
        ("secure_cookies", settings.SESSION_COOKIE_SECURE and settings.CSRF_COOKIE_SECURE, "Session and CSRF cookies require HTTPS"),
        ("https_redirect", settings.SECURE_SSL_REDIRECT is True, "HTTP-to-HTTPS redirect enabled"),
        ("hsts", settings.SECURE_HSTS_SECONDS >= 31536000 and settings.SECURE_HSTS_INCLUDE_SUBDOMAINS and settings.SECURE_HSTS_PRELOAD, "One-year HSTS with subdomains and preload"),
        ("proxy_ssl_header", settings.SECURE_PROXY_SSL_HEADER == ("HTTP_X_FORWARDED_PROTO", "https"), "Reverse-proxy HTTPS header configured"),
        ("model_signing_key", len(settings.MALER_MODEL_SIGNING_KEY) >= 50, "Independent strong model-signing key present"),
        ("bounded_uploads", 0 < settings.FILE_UPLOAD_MAX_MEMORY_SIZE <= settings.MALER_MAX_DATA_UPLOAD_BYTES and settings.MALER_MAX_MODEL_UPLOAD_BYTES > 0, "Positive bounded data/model limits and smaller memory threshold"),
        ("project_path_control", "mlserver.middleware.ProjectIdentifierMiddleware" in settings.MIDDLEWARE and traversal_rejected, "Strict project-ID middleware registered and traversal token rejected"),
    ]
    rows = [{"check": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in checks]
    passed = sum(1 for row in rows if row["passed"])
    payload = {
        "scope": "simulated production configuration and application-control audit",
        "not_in_scope": ["live Aliyun host", "TLS certificate/chain", "network access control", "penetration testing", "backup restore"],
        "passed_checks": passed,
        "total_checks": len(rows),
        "all_passed": passed == len(rows),
        "checks": rows,
    }
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
