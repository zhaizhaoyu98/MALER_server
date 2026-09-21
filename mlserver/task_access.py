"""Capability-token access control for uploaded analysis projects."""

from __future__ import absolute_import

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta

from django.conf import settings
from django.core.exceptions import PermissionDenied

from .security import validated_project_id


ACCESS_FILE = "access.json"


def project_directory(projectid):
    projectid = validated_project_id(projectid)
    cache_root = os.path.realpath(os.path.join(settings.STATIC_ROOT, "cache"))
    target = os.path.realpath(os.path.join(cache_root, projectid))
    if os.path.commonpath([cache_root, target]) != cache_root:
        raise PermissionDenied("Invalid project path.")
    return target


def _digest(token):
    key = settings.SECRET_KEY.encode("utf-8")
    return hmac.new(key, token.encode("utf-8"), hashlib.sha256).hexdigest()


def issue_task_token(projectid):
    directory = project_directory(projectid)
    os.makedirs(directory, exist_ok=True)
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    lifetime = int(getattr(settings, "MALER_CACHE_RETENTION_HOURS", 24))
    payload = {
        "token_digest": _digest(token),
        "created_utc": now.replace(microsecond=0).isoformat() + "Z",
        "expires_utc": (now + timedelta(hours=lifetime)).replace(microsecond=0).isoformat() + "Z",
    }
    temporary = os.path.join(directory, ACCESS_FILE + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, os.path.join(directory, ACCESS_FILE))
    return token


def request_task_token(request):
    return (
        request.POST.get("access_token")
        or request.GET.get("token")
        or request.META.get("HTTP_X_MALER_TASK_TOKEN")
        or ""
    )


def require_task_access(request, projectid):
    directory = project_directory(projectid)
    access_path = os.path.join(directory, ACCESS_FILE)
    try:
        with open(access_path, "r", encoding="utf-8") as handle:
            access = json.load(handle)
    except (OSError, ValueError):
        raise PermissionDenied("This project has no valid access record.")
    token = request_task_token(request)
    if not token or not hmac.compare_digest(_digest(token), access.get("token_digest", "")):
        raise PermissionDenied("A valid project access token is required.")
    expires = datetime.strptime(access["expires_utc"], "%Y-%m-%dT%H:%M:%SZ")
    if datetime.utcnow() > expires:
        raise PermissionDenied("This project access token has expired.")
    return directory, token
