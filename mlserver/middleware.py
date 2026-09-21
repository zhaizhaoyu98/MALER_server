from __future__ import absolute_import

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
