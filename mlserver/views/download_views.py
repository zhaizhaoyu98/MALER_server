from django.views.static import serve
from django.conf import settings
import os
from django.http import FileResponse

def download_sample_data(request, fname):
    # print(request.path)
    print(fname)
    filepath = os.getcwd()
    filepath = filepath + '/mlserver/static/cache/example/' + fname
    file = open(filepath.replace('\\', '/'), 'rb')
    res = FileResponse(file)
    return res

