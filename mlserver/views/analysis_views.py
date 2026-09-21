from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import os, json
from ML_WebServer.settings import STATIC_ROOT
from mlserver.model_registry import FEATURE_REDUCTION, MODEL_REGISTRY, MODEL_REGISTRY_VERSION

def get_analysis_page(request):
    # generate random numbers
    if request.method == 'GET':
        return render(request,'analysis.html')
    elif request.method == 'POST':
        return render(request, 'analysis.html')


def method_registry(request):
    """Machine-readable canonical estimator, default and search-space record."""
    return JsonResponse({
        'registry_version': MODEL_REGISTRY_VERSION,
        'models': MODEL_REGISTRY,
        'feature_reduction': FEATURE_REDUCTION,
    }, json_dumps_params={'indent': 2})


# def get_vue_analysis_page(request):
#     if request.method == 'GET':
#         return render(request,'vue_analysis.html')
#     elif request.method == 'POST':
#         return render(request, 'vue_analysis.html')



#
# def analysis_perpara(request):
#     return render(request, 'analysis_perpara.html')









