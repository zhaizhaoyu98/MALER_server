from __future__ import absolute_import

import re

from django.core.exceptions import SuspiciousOperation


PROJECT_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


def validated_project_id(value):
    """Return a cache-safe project identifier or reject the request."""
    if not value or not PROJECT_ID_PATTERN.match(str(value)):
        raise SuspiciousOperation('Invalid project identifier.')
    return str(value)
