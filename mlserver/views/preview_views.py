from django.shortcuts import render

import os, shutil, pickle

from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5
from mlserver.views.classification_cp_result_views import select_class_model, md5_convert
from mlserver.views.regression_cp_result_views import select_reg_model
from mlserver.views.survival_cp_result_views import select_sur_model
def preview_result(request):
    projectid = request.POST.get('projectid')
    feature_select_method = request.POST.get('feature_select_method')
    file_upload_type = request.POST.get('file_upload_type')
    select_model = request.POST.get('select_model')
    strategy = request.POST.get('strategy')
    model_md5 = None
    print(projectid,file_upload_type, strategy)
    if file_upload_type == 'example_data':
        if select_model == 'model_bclass':
            prefix, example_name = 'BC', 'binary_classification_example.csv'
            if strategy == 'C':
                model, model_name = select_class_model(request)
        elif select_model == 'model_mclass':
            prefix, example_name = 'MC', 'multiclass_classification_example.csv'
            if strategy == 'C':
                model, model_name = select_class_model(request)
        elif select_model == 'model_reg':
            prefix, example_name = 'R', 'regression_example.csv'
            if strategy == 'C':
                model, model_name = select_reg_model(request)
        else:
            prefix, example_name = 'S', 'survival_example.csv'
            if strategy == 'C':
                model, model_name = select_sur_model(request)

        prefix = prefix + strategy
        upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/', example_name))
        if strategy == 'O':
            projectid = prefix + '-' + upload_file_md5[:6] + '-' + feature_select_method
        else:
            token = request.POST.get('random_token')
            projectid = prefix + '-' + upload_file_md5[:6] + '-' + token


        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
            os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
            shutil.copy(STATIC_ROOT + '/cache/example/' + example_name, os.path.join(STATIC_ROOT, 'cache', projectid))
            os.rename(os.path.join(STATIC_ROOT, 'cache', projectid, example_name),  \
                      os.path.join(STATIC_ROOT, 'cache', projectid, 'data.csv'))

        print(strategy)
        if strategy == 'C':
            if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'model_pickle.pkl')):
                model_set = {}
            else:
                with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
                    model_set = pickle.load(f)
            submodel = {
                'model': model,
                'model_name': model_name
            }
            model_md5 = md5_convert(str(model.get_params()) + str(submodel) + feature_select_method)
            model_set[model_md5] = submodel
            print(model_set)
            with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'wb') as f:
                pickle.dump(model_set, f)

    else:
        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
            # file load
            upload_file = request.FILES.get('upload_file')
            if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
                os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
            f = open(os.path.join(STATIC_ROOT, 'cache', projectid, upload_file.name), 'wb')
            for line in upload_file.chunks():
                f.write(line)
            f.close()
            os.rename(os.path.join(STATIC_ROOT, 'cache', projectid, upload_file.name), \
                      os.path.join(STATIC_ROOT, 'cache', projectid, 'data.csv'))

        if strategy == 'C':
            if select_model == 'model_bclass':
                model, model_name = select_class_model(request)
            elif select_model == 'model_mclass':
                model, model_name = select_class_model(request)
            elif select_model == 'model_reg':
                model, model_name = select_reg_model(request)
            else:
                model, model_name = select_sur_model(request)

            if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'model_pickle.pkl')):
                model_set = {}
            else:
                with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
                    model_set = pickle.load(f)
            submodel = {
                'model': model,
                'model_name': model_name
            }
            model_md5 = md5_convert(str(model.get_params()) + str(submodel) + feature_select_method)
            model_set[model_md5] = submodel
            with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'wb') as f:
                pickle.dump(model_set, f)

    if projectid[0] == 'B' or projectid[0] == 'M':
        form_action_p = 'classification'
    elif projectid[0] == 'R':
        form_action_p = 'regression'
    else:
        form_action_p = 'survival'

    if projectid.split('-')[0][-1] == 'O':
        form_action_s = '_oc_result'
    else:
        form_action_s = '_cp_result'
    form_action = form_action_p + form_action_s
    status = 'preview'
    return render(request, 'status.html', {
        'projectid': projectid,
        'form_action': form_action,
        'status': status,
        'feature_select_method': feature_select_method,
        'model_md5': model_md5,
    })