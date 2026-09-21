from __future__ import absolute_import

import json
import os
import shutil

from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.conf import settings
from django.core.mail import send_mail
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods, require_POST

from mlserver.task_access import project_directory, require_task_access
from mlserver.validated_analysis import run_validated_analysis


def _load_result(directory):
    path = os.path.join(directory, "validated_result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_rows(result):
    rows = []
    if not result:
        return rows
    for model in result.get("models", []):
        summary = model.get("nested_cv", {}).get("summary", {})
        for metric, values in summary.items():
            if isinstance(values, dict) and "mean" in values:
                rows.append({"model": model.get("key"), "source": "nested CV", "metric": metric,
                             "value": "%.4f ± %.4f" % (values["mean"], values.get("std", 0.0))})
        for metric, value in (model.get("heldout_test") or {}).items():
            if isinstance(value, (int, float)):
                rows.append({"model": model.get("key"), "source": "held-out test", "metric": metric,
                             "value": "%.4f" % value})
    return rows


@require_http_methods(["GET", "POST"])
def result(request, projectid):
    directory, token = require_task_access(request, projectid)
    analysis_result = _load_result(directory)
    error = None
    notification = None
    if request.method == "POST" and analysis_result is None:
        try:
            analysis_result = run_validated_analysis(directory, projectid, {
                "feature_select_method": request.POST.get("feature_select_method"),
                "fsm": request.POST.get("fsm"),
                "feature_norm": request.POST.get("feature_norm"),
                "model_md5": request.POST.get("model_md5"),
            })
            recipient = request.POST.get("to_mail", "").strip()
            if recipient and settings.EMAIL_HOST_USER:
                result_url = request.build_absolute_uri(request.path + "?token=" + token)
                try:
                    send_mail(
                        "MALER analysis completed",
                        "Your validated MALER result is available at:\n%s\n\nKeep this private link secure." % result_url,
                        settings.EMAIL_HOST_USER,
                        [recipient],
                        fail_silently=False,
                    )
                    notification = "Completion email sent."
                except Exception:
                    notification = "Analysis completed, but the optional email notification could not be sent."
        except Exception as exc:
            error = str(exc)
    return render(request, "validated_result.html", {
        "projectid": projectid,
        "access_token": token,
        "result": analysis_result,
        "metric_rows": _metric_rows(analysis_result),
        "error": error,
        "notification": notification,
    }, status=422 if error else 200)


def previous_result(request, projectid_paramd5):
    projectid = request.GET.get("projectid") or projectid_paramd5
    return result(request, projectid)


def result_json(request, projectid):
    directory, _ = require_task_access(request, projectid)
    result_path = os.path.join(directory, "validated_result.json")
    if not os.path.exists(result_path):
        raise Http404("The analysis result is not available yet.")
    with open(result_path, "r", encoding="utf-8") as handle:
        return JsonResponse(json.load(handle), json_dumps_params={"indent": 2})


def model_bundle(request, projectid, filename):
    directory, _ = require_task_access(request, projectid)
    if not filename.endswith(".maler") or os.path.basename(filename) != filename:
        raise Http404("Invalid model bundle name.")
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        raise Http404("Model bundle not found.")
    return FileResponse(open(path, "rb"), as_attachment=True, filename=filename)


def prediction_artifact(request, projectid, filename):
    directory, _ = require_task_access(request, projectid)
    if (not filename.endswith("_predictions.csv") or os.path.basename(filename) != filename
            or not all(character.isalnum() or character in "._-" for character in filename)):
        raise Http404("Invalid prediction artifact name.")
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        raise Http404("Prediction artifact not found.")
    return FileResponse(open(path, "rb"), as_attachment=True, filename=filename)


@require_POST
def delete_project(request, projectid):
    directory, _ = require_task_access(request, projectid)
    shutil.rmtree(directory)
    return HttpResponse("Project data and generated artifacts were permanently deleted.",
                        content_type="text/plain; charset=utf-8")


def legacy_disabled(request, projectid):
    return HttpResponse(
        "Legacy websocket analysis is disabled. Use the validated result page for this project.",
        status=410, content_type="text/plain; charset=utf-8")
