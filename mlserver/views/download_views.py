from django.views.static import serve
from django.conf import settings
import os
from django.http import FileResponse, Http404, StreamingHttpResponse

from ML_WebServer.settings import STATIC_ROOT

def download_sample_data(request, fname):
    # print(request.path)
    print(fname)
    filepath = os.getcwd()
    filepath = filepath + '/mlserver/static/cache/example/' + fname
    file = open(filepath.replace('\\', '/'), 'rb')
    res = FileResponse(file)
    return res


def download_model(request, projectid_model):
    projectid = projectid_model.split('_')[0]
    model = projectid_model.split('_')[1].replace(' ','_')
    print(projectid,model)
    file_path = (STATIC_ROOT + '/cache/' + projectid + '/' + model + '.pkl')
    try:
        response = StreamingHttpResponse(open(file_path, 'rb'))
        response['content_type'] = "application/octet-stream"
        response['Content-Disposition'] = 'attachment; filename=' + projectid + '_' + os.path.basename(file_path)
        return response
    except Exception:
        raise Http404
