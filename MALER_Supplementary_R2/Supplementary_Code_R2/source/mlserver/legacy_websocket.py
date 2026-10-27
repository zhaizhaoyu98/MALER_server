"""Explicitly disable retired websocket calculation handlers.

The legacy modules remain importable for read-only helper functions used by
preview, but their old calculation routes must never be executed.
"""

from functools import wraps

from django.core.exceptions import PermissionDenied


def accept_websocket(function):
    @wraps(function)
    def disabled(*args, **kwargs):
        raise PermissionDenied("Legacy websocket analysis is disabled.")
    return disabled
