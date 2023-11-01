from django.shortcuts import render,HttpResponse
import os, json
from ML_WebServer.settings import STATIC_ROOT

def get_analysis_page(request):
    # generate random numbers
    if request.method == 'GET':
        return render(request,'analysis.html')
    elif request.method == 'POST':
        return render(request, 'analysis.html')


# def get_vue_analysis_page(request):
#     if request.method == 'GET':
#         return render(request,'vue_analysis.html')
#     elif request.method == 'POST':
#         return render(request, 'vue_analysis.html')



#
# def analysis_perpara(request):
#     return render(request, 'analysis_perpara.html')









