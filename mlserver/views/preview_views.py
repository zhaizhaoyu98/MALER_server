import json

from django.shortcuts import render

import os, shutil, pickle, secrets
import numpy as np
import pandas as pd

from django.conf import settings
from mlserver.views.classification_oc_result_view_webscoket import get_file_md5, split_train_test, JsonEncoder,\
    classification_process,label_pre
from mlserver.views.classification_cp_result_view_websocket import select_class_model, md5_convert
from mlserver.views.regression_cp_result_view_websocket import select_reg_model,regression_preprocess
from mlserver.views.survival_cp_result_view_websocket import select_sur_model,sur_data_process
from django.contrib import messages
import re
from django.core.exceptions import SuspiciousOperation
from mlserver.security import validated_project_id
from mlserver.task_access import issue_task_token

def preview_result(request):
    STATIC_ROOT = settings.STATIC_ROOT
    MLSERVER_STATIC_DIR = settings.MLSERVER_STATIC_DIR
    cache_dir = os.path.join(STATIC_ROOT, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    # projectid='BCO-AN-c319b6-TopK';feature_select_method='TopK';
    # fsm='A';file_upload_type='example_data';select_model='model_bclass'
    # strategy='0';to_mail='';fn='N'
    projectid = request.POST.get('projectid')
    feature_select_method = request.POST.get('feature_select_method') or 'TopK'
    fsm = request.POST.get('fsm') or 'A'
    file_upload_type = request.POST.get('file_upload_type')
    select_model = request.POST.get('select_model')
    strategy = request.POST.get('strategy')
    to_mail = request.POST.get('to_mail') or ''
    fn = request.POST.get('feature_norm') or 'Z'

    if strategy not in ('O', 'C'):
        raise SuspiciousOperation('Unsupported analysis strategy.')
    if select_model not in ('model_bclass', 'model_mclass', 'model_reg', 'model_sur'):
        raise SuspiciousOperation('Unsupported analysis type.')

    model_md5 = None
    # Do not log the optional email address or uploaded-data metadata.

    if file_upload_type == 'example_data':
        if select_model == 'model_bclass':
            prefix, example_name = 'BC', 'binary_classification_example.csv'
            if strategy == 'C':
                model, model_name, gridsearch_para = select_class_model(request)
        elif select_model == 'model_mclass':
            prefix, example_name = 'MC', 'multiclass_classification_example.csv'
            if strategy == 'C':
                model, model_name, gridsearch_para = select_class_model(request)
        elif select_model == 'model_reg':
            prefix, example_name = 'R', 'regression_example.csv'
            if strategy == 'C':
                model, model_name, gridsearch_para = select_reg_model(request)
        else:
            prefix, example_name = 'S', 'survival_example.csv'
            if strategy == 'C':
                model, model_name, gridsearch_para = select_sur_model(request)
        if strategy != None:
            prefix = prefix + strategy
        example_file = os.path.join(MLSERVER_STATIC_DIR, 'cache', 'example', example_name)
        upload_file_md5 = get_file_md5(example_file)

        if strategy == 'O':
            gridsearch_para = {}
            projectid = prefix + '-' + fsm + fn + '-' + upload_file_md5[:6] + '-' + feature_select_method
        else:
            token = request.POST.get('random_token') or secrets.token_hex(8)
            projectid = prefix + '-' + fsm + fn + '-' + upload_file_md5[:6] + '-' + token

        projectid = validated_project_id(projectid + '-' + secrets.token_hex(4))
        print(projectid)



        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
            os.makedirs(os.path.join(STATIC_ROOT, 'cache', projectid), exist_ok=True)
            shutil.copy(example_file, os.path.join(STATIC_ROOT, 'cache', projectid))
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
                'model_name': model_name,
                'gridsearch_para': gridsearch_para
            }
            model_md5 = md5_convert(str(model.get_params()) + str(submodel) + feature_select_method)
            model_set[model_md5] = submodel

            with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'wb') as f:
                pickle.dump(model_set, f)

    else:
        if request.POST.get('data_consent') != 'confirmed':
            raise SuspiciousOperation('Data authorization and privacy confirmation is required.')
        projectid = validated_project_id(projectid)
        projectid = validated_project_id(projectid + '-' + secrets.token_hex(4))
        if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
            # file load
            upload_file = request.FILES.get('upload_file')
            if upload_file is None:
                raise SuspiciousOperation('No upload was provided.')
            if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
                os.makedirs(os.path.join(STATIC_ROOT, 'cache', projectid), exist_ok=True)
            # Never place the client-supplied filename into a filesystem path.
            destination = os.path.join(STATIC_ROOT, 'cache', projectid, 'data.csv')
            with open(destination, 'wb') as handle:
                for line in upload_file.chunks():
                    handle.write(line)

        if strategy == 'C':
            if select_model == 'model_bclass':
                model, model_name, gridsearch_para = select_class_model(request)
            elif select_model == 'model_mclass':
                model, model_name, gridsearch_para = select_class_model(request)
            elif select_model == 'model_reg':
                model, model_name, gridsearch_para = select_reg_model(request)
            else:
                model, model_name, gridsearch_para = select_sur_model(request)

            if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'model_pickle.pkl')):
                model_set = {}
            else:
                with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
                    model_set = pickle.load(f)
            submodel = {
                'model': model,
                'model_name': model_name,
                'gridsearch_para': gridsearch_para
            }
            model_md5 = md5_convert(str(model.get_params()) + str(submodel) + feature_select_method)
            model_set[model_md5] = submodel
            with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'wb') as f:
                pickle.dump(model_set, f)

    projectid = validated_project_id(projectid)
    if projectid[0] == 'B' or projectid[0] == 'M':
        form_action_p = 'classification'
    elif projectid[0] == 'R':
        form_action_p = 'regression'
    else:
        form_action_p = 'survival'

    if projectid.split('-')[0][-1] == 'O':
        gridsearch_para = {}
        form_action_s = '_oc_result'
    else:
        form_action_s = '_cp_result'
    form_action = form_action_p + form_action_s
    status = 'Preview'

    ''' preview '''
    inputdata = pd.read_csv(
        STATIC_ROOT + '/cache/' + projectid + '/data.csv',
        header=0, index_col=0, sep=None, engine='python').T





    ''' check '''
    if to_mail != '':
        if not re.match('^.*?@.*', to_mail):
            title = 'The provided email address is invalid, please provide a correct one.'
            messages.success(request, title)
            return render(request, "analysis.html")
    if select_model == 'model_bclass' and inputdata.iloc[:, 0].nunique(dropna=True) != 2:
        title = 'Selected analysis mode is inconsistent with the input data type!'
        messages.success(request, title)
        return render(request, "analysis.html")

    if select_model == 'model_mclass' and (50 < inputdata.iloc[:, 0].nunique(dropna=True) or inputdata.iloc[:, 0].nunique(dropna=True) <= 2):
        title = 'The number of categories is greater than 50 or less than 2!'
        messages.success(request, title)
        return render(request, "analysis.html")

    if select_model == 'model_reg' and (pd.to_numeric(inputdata.iloc[:, 0],errors='ignore')).unique().dtype != np.dtype('float64'):
        title = 'Selected analysis mode is inconsistent with the input data type!'
        messages.success(request, title)
        return render(request, "analysis.html")
    if select_model == 'model_sur' and inputdata.columns[0].lower()!= 'status':
        title = 'Selected analysis mode is inconsistent with the input data type!'
        messages.success(request, title)
        return render(request, "analysis.html")

    # sample table display
    if form_action_p == 'survival':
        train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')
        hist_values, bin_edges, bins_centers = data_hist(inputdata, datatype='survival')
    else:
        train_set, test_set, blind_set = split_train_test(inputdata)
        hist_values, bin_edges, bins_centers = data_hist(inputdata)
    display_samples = pd.DataFrame({'Train': train_set.shape, 'Test': test_set.shape, 'Blind': blind_set.shape},
                                   index=['Samples', 'Features'])
    display_samples_dict = display_samples.reset_index().rename(columns={'index': 'class'}).to_dict('records')

    # Normalization is intentionally not fitted during preview.  The validated
    # analysis service fits it independently inside every CV training fold.


    # histogram
    hist_trace = [{
        'x': bin_edges,
        'y': hist_values,
        'type': "bar",
        'opacity': 0.9,
    }]
    # data short view
    inputdata_display = inputdata.T.head(50).reset_index().rename(columns={'index': 'features'})
    inputdata_display.columns = [i.replace('.','-') for i in inputdata_display.columns]
    inputdata_columns, title_str = mkcol(inputdata_display)
    inputdata_display = inputdata_display.to_dict("records")
    # save preview pickle
    preview_pickle = {
        'feature_select_method': feature_select_method,
        'form_action': form_action,
        'status': status,
        'display_samples_dict': display_samples_dict,
        'hist_trace': hist_trace,
        'inputdata_display': inputdata_display,
        'inputdata_columns': inputdata_columns,
        'title_str': title_str,
        'to_mail': to_mail,
        'gridsearch_para': gridsearch_para,
        'fsm': fsm,
        'feature_norm': fn,
        'validation_mode': 'nested_fold_local',
    }
    with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'wb') as f:
        pickle.dump(preview_pickle, f)

    access_token = issue_task_token(projectid)
    return render(request, 'status.html', {
        'projectid': projectid,
        'form_action': form_action,
        'status': status,
        'feature_select_method': feature_select_method,
        'model_md5': model_md5,
        'display_samples_dict': json.dumps(display_samples_dict),
        'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
        'inputdata_display': json.dumps(inputdata_display),
        'inputdata_columns': json.dumps(inputdata_columns),
        'title_str': title_str,
        'fsm': fsm,
        'to_mail': to_mail,
        'gridsearch_para': gridsearch_para,
        'feature_norm': fn,
        'access_token': access_token

    })

def data_hist(data, datatype='other'):
    num = (1, 2)[datatype == 'survival']  # datatype == 'survival'时选第三列，否则为第二列
    if 'training' in data.iloc[:, num].unique():
        data2 = data.drop(columns=data.columns[:num + 1])
    else:
        data2 = data.drop(columns=data.columns[:num])
    data2 = data2.T
    data2 = data2.apply(pd.to_numeric,errors='ignore')
    data2 = data2.T
    # data2 = (data2).apply(pd.to_numeric, errors='ignore')
    drop_X_train = data2.select_dtypes(include=['object'])
    data3 = data2.loc[:, ~data2.columns.isin(drop_X_train.columns)]
    data_all = np.array(data3).ravel()
    hist_values, bin_edges = np.histogram(data_all, bins=20, density=True)
    bins_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return hist_values, bin_edges, bins_centers

def mkcol(data):
    col_data, title_str = [], ''
    for col in data.columns:
        subcol = {
            'data': col
        }
        col_data.append(subcol)
        title_str = title_str + '<th>' + col + '</th>'
    return col_data, title_str

def data_normalization(train_set, test_set, fn, projectid):
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler,RobustScaler
    print('fn: ',fn)
    if fn == 'MM':
        scaler = MinMaxScaler()
    elif fn == 'Z':
        scaler = StandardScaler()
    elif fn == 'MA':
        scaler = MaxAbsScaler()
    elif fn == 'R':
        scaler = RobustScaler()

    print('projectid: ',projectid)
    classes = None
    X_test_scaled = []
    validation_label = None
    if projectid.split("-")[0][0] == 'B' or projectid.split("-")[0][0] == 'M':
        data, label = classification_process(train_set)
        label, classes = label_pre(label)
        validation_data = []  # 预先定义
        ifval = False
        if len(test_set) > 0:
            validation_data, validation_label = classification_process(test_set)
            validation_label, classes = label_pre(validation_label)
            ifval = True
    elif projectid.split("-")[0][0] == 'R':
        data, label = regression_preprocess(train_set)
        ifval = False
        if len(test_set) > 0:
            validation_data, validation_label = regression_preprocess(test_set)
            ifval = True
    elif projectid.split("-")[0][0] == 'S':
        data, label = sur_data_process(train_set)
        ifval = False
        if len(test_set) > 0:
            validation_data, validation_label = sur_data_process(test_set)
            ifval = True

    X_train_scaled = scaler.fit_transform(data)
    X_train_scaled = pd.DataFrame(X_train_scaled,index=data.index,columns=data.columns)

    if len(test_set) > 0:
        X_test_scaled = scaler.transform(validation_data)
        X_test_scaled = pd.DataFrame(X_test_scaled, index=validation_data.index, columns=validation_data.columns)

    norm_data = {'train_set': X_train_scaled,
                 'test_set': X_test_scaled,
                 'train_set_label': label,
                 'test_set_label': validation_label,
                 'scaler': scaler,
                 'classes': classes,
                 'ifval': ifval
                 }
    print('normalization!')
    return norm_data

