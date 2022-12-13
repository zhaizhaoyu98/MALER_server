from django.shortcuts import render
from django.http import HttpResponse


def get_help_page(request):
    return render(request, 'help.html')

