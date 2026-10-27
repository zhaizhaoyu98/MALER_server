from __future__ import absolute_import

import shutil

from django.conf import settings
from django.http import HttpResponse

from .security import validated_project_id


class ProjectIdentifierMiddleware(object):
    """Reject unsafe route identifiers before any cache path is constructed."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_view(self, request, view_func, view_args, view_kwargs):
        for key, value in view_kwargs.items():
            if key.startswith('projectid'):
                validated_project_id(value)
        return None


class MalerPrivacyHeadersMiddleware(object):
    """Keep capability URLs and uploaded-data pages out of caches/referrers."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if request.path.startswith('/maler/'):
            # Django 2.1 checks the HTTPS Referer. Send only the origin so
            # legitimate forms work without exposing task paths or tokens.
            response['Referrer-Policy'] = 'strict-origin'
            response['Cache-Control'] = 'no-store'
            response['X-Content-Type-Options'] = 'nosniff'
        return response


class MalerUploadCapacityMiddleware(object):
    """Reject uploads before multipart parsing when private storage is full."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        minimum = settings.MALER_MIN_FREE_TASK_BYTES
        if (minimum > 0 and settings.MALER_PUBLIC_UPLOADS_ENABLED
                and request.method == 'POST'
                and request.path in ('/maler/preview', '/maler/predict_preview')):
            try:
                incoming = int(request.META.get('CONTENT_LENGTH') or 0)
                if incoming < 0:
                    raise ValueError('Negative Content-Length')
            except (TypeError, ValueError):
                return HttpResponse('Invalid upload length.', status=400)
            try:
                free = shutil.disk_usage(settings.MALER_TASK_ROOT).free
            except OSError:
                return HttpResponse('Task storage is temporarily unavailable. Please retry later.', status=503)
            if free < minimum + incoming:
                return HttpResponse('Task storage is busy. Please retry later or use a smaller file.', status=507)
        return self.get_response(request)
