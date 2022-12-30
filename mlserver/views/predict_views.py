import json

from django.shortcuts import render

import os, shutil, pickle
import numpy as np
import pandas as pd


from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5
from django.contrib import messages
from mlserver.views.preview_views import data_hist,mkcol
def get_predict_page(request):
    # generate random numbers
    if request.method == 'GET':
        return render(request,'predict.html')
    elif request.method == 'POST':

        return render(request, 'predict.html')

def predict_result(request):
    select_model = request.POST.get('select_model')
    file_upload_type = request.POST.get('file_upload_type')
    '''
    IMPORT DATA
    '''
    # blind_set
    if file_upload_type == 'example_data':
        if select_model == 'model_bclass':
            example_name = 'binary_pred.csv'
        elif select_model == 'model_mclass':
            example_name = 'multi_pred.csv'
        elif select_model == 'model_reg':
            example_name = 'reg_pred.csv.csv'
        else:
            example_name = 'survival_pred.csv'

        obj_model = request.FILES.get('upload_model2')
        print(obj_model.name)
        f = open(os.path.join(STATIC_ROOT, 'cache', obj_model.name), 'wb')
        for line in obj_model.chunks():
            f.write(line)
        f.close()
        filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/', example_name))
        modelmd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_model.name))
        projectid = 'PRED' + '-' + select_model.replace('model_', '').upper() + '-' + modelmd5[:6] + '-' + filemd5[:6]
        # blind_set = pd.read_csv(STATIC_ROOT + '/cache/' + 'example/' + example_name, header=0, index_col=0).T
        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
            os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
            shutil.copy(STATIC_ROOT + '/cache/example/' + example_name, os.path.join(STATIC_ROOT, 'cache', projectid))
            newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
            shutil.move(STATIC_ROOT + '/cache/' + obj_model.name, newpath)
            os.rename(os.path.join(STATIC_ROOT, 'cache', projectid, example_name),  \
                      os.path.join(STATIC_ROOT, 'cache', projectid, 'data.csv'))
    else:
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
        filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_file.name))
        modelmd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_model.name))
        projectid = 'PRED' + '-' + select_model.replace('model_', '').upper() + '-' + modelmd5[:6] + '-' + filemd5[:6]

        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):

            newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
            print(newpath)
            os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
            shutil.move(STATIC_ROOT + '/cache/' + obj_model.name, newpath)
            shutil.move(STATIC_ROOT + '/cache/' + obj_file.name, newpath)

            os.rename(os.path.join(STATIC_ROOT, 'cache', projectid, obj_file.name), \
                      os.path.join(STATIC_ROOT, 'cache', projectid, 'data.csv'))



    # blind_set = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + obj_file.name, header=0, index_col=0).T
    # print(projectid)




    '''
        select_model='model_bclass'
        blind_set = pd.read_csv(r'C:/Users/Administrator/Desktop/jupyter_project/example/ml示例数据/tpm_binary_df.csv', header=0, index_col=0).T
        with open(r'E:\/CodeProject/WebServer/ML_WebServer/mlserver/static/cache/BCO-e5e9da-TopK/Naive_Bayes.pkl', 'rb') as f:
            model_pickle = pickle.load(f)
        
        select_model='model_reg'
        blind_set = pd.read_csv(r'C:/Users/Administrator/Desktop/jupyter_project/example/ml示例数据/regression_exemple.csv', header=0, index_col=0).T
        with open(r'E:\/CodeProject/WebServer/ML_WebServer/mlserver/static/cache/RO-a4de3d-TopK/LinearRegression.pkl', 'rb') as f:
            model_pickle = pickle.load(f)
        
        select_model='model_sur'
        blind_set = pd.read_csv(r'C:/Users/Administrator/Desktop/jupyter_project/example/ml示例数据/tpm_gbm_surdata.csv', header=0, index_col=0).T
        with open(r'E:\/CodeProject/WebServer/ML_WebServer/mlserver/static/cache/SO-c319b6-TopK/SurvivalTree.pkl', 'rb') as f:
            model_pickle = pickle.load(f)
    '''

    with open(STATIC_ROOT + '/cache/' + projectid + '/' + obj_model.name, 'rb') as f:
        model_pickle = pickle.load(f)
#model
    method, model_name, model, feature_names = \
        model_pickle['method'], model_pickle['name'], model_pickle['model'], model_pickle['feature_names']
    blind_set = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
    if select_model == method:
        if select_model == 'model_bclass' or select_model == 'model_mclass':

            classes = model_pickle['classes']

            # blind_set = blind_set[5:20]  # 模拟的blind数据
            predict_reports = {}
            mapping = dict(zip(classes.values(), classes.keys()))  # 键值对翻转
            try:
                predict_reports[model_name] = model.predict(blind_set[feature_names])
            except:
                title = 'feature names: ' + ", " .join(feature_names) + ' are not involved in the inputdata!'
                messages.success(request, title)
                return render(request, "predict.html")

            predict_reports = pd.DataFrame(predict_reports, index=blind_set.index).applymap(lambda x: mapping[x])
            predict_reports_dict = predict_reports.reset_index().rename(columns={'index': 'Name', model_name: 'Label'}).to_dict('records')
            showtable = True
            method = 'Classification'
            surv_plot = None
        elif select_model == 'model_reg':
            # blind_set = blind_set[5:20]  # 模拟的blind数据

            predict_reports = {}
            try:
                predict_reports[model_name] = model.predict(blind_set[feature_names])
            except:
                title = 'feature names: ' + ", ".join(feature_names) + ' are not involved in the inputdata!'
                messages.success(request, title)
                return render(request, "predict.html")
            predict_reports = pd.DataFrame(predict_reports, index=blind_set.index)
            predict_reports_dict = predict_reports.reset_index().rename(columns={'index': 'Name', model_name: 'Label'}).to_dict('records')
            showtable = True
            method = 'Regression'
            surv_plot = None
        else:
            # blind_set = blind_set[5:20]  # 模拟的blind数据
            try:
                if model_name != 'SurvivalSVM':
                    surv_plot = sur_pred_plot(model, blind_set, feature_names)
                else:
                    surv_plot = None
                    print('The svm model does not support the prediction function')
            except:
                title = 'feature names: ' + ", ".join(feature_names) + ' are not involved in the inputdata!'
                messages.success(request, title)
                return render(request, "predict.html")
            showtable = False
            predict_reports_dict = None
            method = 'Survival'
        shutil.rmtree(os.path.join(STATIC_ROOT, 'cache', projectid))
        return render(request, 'predict_result.html', {
            'projectid': projectid,
            'method': method,
            'showtable': showtable,
            'predict_reports_dict': json.dumps(predict_reports_dict),
            'surv_plot': json.dumps(surv_plot),
        })
    else:
        title = 'Data type error!'
        messages.success(request,title)
        return render(request, "predict.html")


def sur_pred_plot(model, blind_set, feature_names):
    data = blind_set[list(feature_names)]
    surv = model.predict_survival_function(data)
    surv2 = model.predict_cumulative_hazard_function(data)
    tarce_data = []
    for i in range(len(surv)):
        trace1 = surv_trace_struct(surv[i].x, surv[i].y, 1, data.index[i])
        trace2 = surv_trace_struct(surv2[i].x, surv2[i].y, 2, data.index[i])
        tarce_data.append(trace1), tarce_data.append(trace2)
    return tarce_data

def surv_trace_struct(x, y, num, name):
    trace = {
        # 'key': null,
        'line': {
            'dash': 'solid',
            # 'color': 'red',
            'shape': 'hv',
            'width': 2
        },
        'mode': 'lines',
        'name': name,
        'type': 'scatter',
        'x': list(x),
        'y': list(y),
        'xaxis': 'x' + str(num),
        'yaxis': 'y' + str(num),
        'text': make_surv_text(x, y),
        'hoverinfo': 'text',
        'showlegend': True
        # 'legendgroup': 'High Risk'
    }
    return trace

def make_surv_text(x, y):
    string = 'time: %s<br>surv: %s'
    text = []
    for i in range(len(x)):
        str = string %(x[i], np.round(y[i], 3))
        text.append(str)
    return text