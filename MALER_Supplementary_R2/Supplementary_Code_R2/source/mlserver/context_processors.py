from django.conf import settings


def service_policy(request):
    return {
        "public_uploads_enabled": bool(settings.MALER_PUBLIC_UPLOADS_ENABLED),
        "task_retention_hours": settings.MALER_CACHE_RETENTION_HOURS,
        "data_upload_limit_mb": settings.MALER_MAX_DATA_UPLOAD_BYTES // (1024 * 1024),
        "model_upload_limit_mb": settings.MALER_MAX_MODEL_UPLOAD_BYTES // (1024 * 1024),
    }
