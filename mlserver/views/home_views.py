from django.shortcuts import render
from ML_WebServer.settings import STATIC_ROOT
# Create your views here.
from django.http import HttpResponse
def hello_world(request):
    return  HttpResponse('Hello Jack Ma!')

def home(request):
    return render(request, 'home.html')