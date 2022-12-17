from django.shortcuts import render

import os, shutil, pickle
import pandas as pd


from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5
def get_predict_page(request):
    # generate random numbers
    if request.method == 'GET':
        return render(request,'predict.html')
    elif request.method == 'POST':

        return render(request, 'predict.html')

def predict_result(request):
    select_model = request.POST.get('select_model')

    '''
    IMPORT DATA
    '''
    # model load
    obj_model = request.FILES.get('upload_model')
    print(obj_model.name)
    f = open(os.path.join(STATIC_ROOT, 'cache', obj_model.name), 'wb')
    for line in obj_model.chunks():
        f.write(line)
    f.close()

    # profile load
    obj_file = request.FILES.get('upload_file')
    f = open(os.path.join(STATIC_ROOT, 'cache', obj_file.name), 'wb')
    for line in obj_file.chunks():
        f.write(line)
    f.close()

    modelmd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_model.name))
    filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_file.name))
    projectid = 'PRED' + select_model.replace('model_', '').upper() + modelmd5[:6] + '-' + filemd5[:6]
    print(projectid)

    newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
    os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
    shutil.move(STATIC_ROOT + '/cache/' + obj_model.name, newpath)
    shutil.move(STATIC_ROOT + '/cache/' + obj_file.name, newpath)

    with open(STATIC_ROOT + '/cache/' + projectid + '/' + obj_model.name, 'rb') as f:
        model_pickle = pickle.load(f)

    blind_set = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + obj_file.name, header=0, index_col=0).T
    if select_model == 'model_bclass' or select_model == 'model_mclass':
        blind_set = vaildation_data[:15]  # 模拟的blind数据

        predict_reports = {}
        mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
        predict_reports[clf_name] = best_esti[0].predict(blind_set[feature_names[0]])
        predict_reports = pd.DataFrame(predict_reports, index=blind_set.index).applymap(lambda x: mapping[x])
        predict_reports.T
    elif select_model == 'model_reg':
        blind_set = vaildation_data[:15]  # 模拟的blind数据

        predict_reports = {}
        predict_reports[reg_model_name] = best_esti[0].predict(blind_set[feature_names[0]])
        predict_reports = pd.DataFrame(predict_reports, index=blind_set.index)
        predict_reports.T
    else:
        blind_set = vaildation_data[:15]  # 模拟的blind数据

        if sur_model_name != 'SurvivalSVM':
            sur_pred_plot(best_esti[0], blind_set, feature_names[0], sur_model_name)
        else:
            print('The svm model does not support the prediction function')