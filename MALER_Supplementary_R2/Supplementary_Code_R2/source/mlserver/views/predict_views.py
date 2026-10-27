import json

from django.shortcuts import render
from django.conf import settings

import json
import os, shutil, pickle, uuid, secrets
import numpy as np
import pandas as pd

from mlserver.views.classification_oc_result_view_webscoket import JsonEncoder
from ML_WebServer.settings import MLSERVER_STATIC_DIR
from mlserver.views.classification_oc_result_view_webscoket import get_file_md5
from django.contrib import messages
from django.core.exceptions import SuspiciousOperation
from mlserver.views.preview_views import data_hist, mkcol, script_json, html_safe_records
from mlserver.safe_ml import ModelBundleError, load_signed_model_bundle
from mlserver.validated_analysis import validate_sample_header, read_uploaded_matrix
from mlserver.task_access import issue_task_token
from mlserver.security import validated_project_id


TASK_TO_METHOD = {
    'binary': 'model_bclass',
    'multiclass': 'model_mclass',
    'classification': 'model_bclass',
    'regression': 'model_reg',
    'survival': 'model_sur',
}


def _bundle_as_legacy_dict(bundle_path):
    model, manifest = load_signed_model_bundle(
        bundle_path,
        settings.MALER_MODEL_SIGNING_KEY,
        max_model_bytes=settings.MALER_MAX_MODEL_UPLOAD_BYTES,
    )
    metadata = manifest.get('metadata') or {}
    method = metadata.get('method') or TASK_TO_METHOD.get(manifest.get('task'))
    if method is None:
        raise ModelBundleError('The model bundle contains an unsupported task.')
    classes = metadata.get('classes')
    if classes is None and hasattr(model, 'classes_'):
        classes = {str(value): value for value in model.classes_}
    if hasattr(model, 'classes_') and len(model.classes_) == 2:
        if metadata.get('positive_class') not in [str(value) for value in model.classes_]:
            raise ModelBundleError(
                'This binary bundle does not define its positive class; retrain and export it with the current MALER release.')
    return {
        'method': method,
        'name': metadata.get('model_name') or manifest.get('model_class', 'MALERModel').split('.')[-1],
        'model': model,
        'classes': classes or {},
        'feature_names': manifest.get('feature_names', []),
        'manifest': manifest,
        'positive_class': metadata.get('positive_class'),
        'threshold_policy': metadata.get('threshold_policy') or 'estimator_native_predict',
        'threshold_value': (0.5 if metadata.get('threshold_value') is None
                            else metadata['threshold_value']),
    }


def _save_uploaded_file(upload, destination, maximum_bytes):
    if upload is None:
        raise ValueError('Both a model bundle and a data file are required.')
    if upload.size > maximum_bytes:
        raise ValueError('Uploaded file exceeds the configured size limit.')
    with open(destination, 'wb') as handle:
        for chunk in upload.chunks():
            handle.write(chunk)


def get_predict_page(request):
    # generate random numbers
    if request.method == 'GET':
        return render(request, 'predict.html', {'public_uploads_enabled': settings.MALER_PUBLIC_UPLOADS_ENABLED})
    elif request.method == 'POST':
        return render(request, 'predict.html', {'public_uploads_enabled': settings.MALER_PUBLIC_UPLOADS_ENABLED})

def predict_preview(request):
    if request.POST.get('select_model') not in (
            'model_bclass', 'model_mclass', 'model_reg', 'model_sur'):
        raise SuspiciousOperation('Unsupported prediction task.')
    if request.POST.get('file_upload_type') == 'example_data':
        raise SuspiciousOperation(
            'Legacy example models are incompatible with the current signed-bundle format. '
            'Create a new model from the local analysis workflow.')
    if request.POST.get('file_upload_type') != 'example_data' and not settings.MALER_PUBLIC_UPLOADS_ENABLED:
        raise SuspiciousOperation('Public model/data upload is temporarily unavailable; use the local installation.')
    if request.POST.get('file_upload_type') != 'example_data' and request.POST.get('data_consent') != 'confirmed':
        raise SuspiciousOperation('Data authorization and privacy confirmation is required.')
    cache_dir = settings.MALER_TASK_ROOT
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)

    select_model = request.POST.get('select_model')
    file_upload_type = request.POST.get('file_upload_type')
    '''
    IMPORT DATA
    '''
    # blind_set
    if file_upload_type == 'example_data':
        if select_model == 'model_bclass':
            example_name = 'binary_pred.csv'
            example_model_name = 'Binary_RandomForest.pkl'
        elif select_model == 'model_mclass':
            example_name = 'multi_pred.csv'
            example_model_name = 'Multiclass_XGBoost.pkl'
        elif select_model == 'model_reg':
            example_name = 'reg_pred.csv'
            example_model_name = 'Regression_Lasso.pkl'
        else:
            example_name = 'survival_pred.csv'
            example_model_name = 'Survival_ExtraSurvivalTrees.pkl'

        # obj_model = request.FILES.get('upload_model2')
        #
        # f = open(os.path.join(settings.MALER_TASK_ROOT, obj_model.name), 'wb')
        # for line in obj_model.chunks():
        #     f.write(line)
        # f.close()
        example_dir = os.path.join(MLSERVER_STATIC_DIR, 'cache', 'example')
        example_file = os.path.join(example_dir, example_name)
        example_model_file = os.path.join(example_dir, example_model_name)
        filemd5 = get_file_md5(example_file)
        # modelmd5 = get_file_md5(os.path.join(settings.MALER_TASK_ROOT, obj_model.name))
        modelmd5 = get_file_md5(example_model_file)
        projectid = validated_project_id('PRED' + '-' + select_model.replace('model_', '').upper()
                                         + '-' + modelmd5[:6] + '-' + filemd5[:6]
                                         + '-' + secrets.token_hex(4))
        # blind_set = pd.read_csv(settings.MALER_TASK_ROOT + '/' + 'example/' + example_name, header=0, index_col=0).T
        if not os.path.exists(os.path.join(settings.MALER_TASK_ROOT, projectid)):
            os.makedirs(os.path.join(settings.MALER_TASK_ROOT, projectid), exist_ok=True)
            newpath = os.path.join(settings.MALER_TASK_ROOT, projectid)
            shutil.copy(example_file, newpath)
            shutil.copy(example_model_file, newpath)
            os.rename(os.path.join(settings.MALER_TASK_ROOT, projectid, example_name),  \
                      os.path.join(settings.MALER_TASK_ROOT, projectid, 'data.csv'))
            os.rename(os.path.join(settings.MALER_TASK_ROOT, projectid, example_model_name), \
                      os.path.join(settings.MALER_TASK_ROOT, projectid, 'pickle.pkl'))
    else:
        obj_model = request.FILES.get('upload_model')
        obj_file = request.FILES.get('upload_file')
        if obj_model is None or not obj_model.name.lower().endswith('.maler'):
            messages.error(request, 'Only signed .maler model bundles generated by MALER are accepted.')
            return render(request, 'predict.html')
        if not settings.MALER_MODEL_SIGNING_KEY:
            messages.error(request, 'External model upload is disabled until a model signing key is configured.')
            return render(request, 'predict.html')
        if obj_file is None or os.path.splitext(obj_file.name.lower())[1] not in ('.csv', '.tsv', '.txt'):
            messages.error(request, 'Prediction data must be a CSV, TSV, or TXT file.')
            return render(request, 'predict.html')
        try:
            validate_sample_header(obj_file.readline().decode('utf-8-sig'))
        except (UnicodeDecodeError, ValueError) as exc:
            messages.error(request, str(exc))
            return render(request, 'predict.html')
        finally:
            obj_file.seek(0)

        temporary_id = uuid.uuid4().hex
        temporary_model = os.path.join(cache_dir, temporary_id + '.maler')
        temporary_data = os.path.join(cache_dir, temporary_id + '.data')
        try:
            _save_uploaded_file(obj_model, temporary_model, settings.MALER_MAX_MODEL_UPLOAD_BYTES)
            _save_uploaded_file(obj_file, temporary_data, settings.MALER_MAX_DATA_UPLOAD_BYTES)
            model_pickle = _bundle_as_legacy_dict(temporary_model)
        except (ValueError, OSError, ModelBundleError) as exc:
            for path in (temporary_model, temporary_data):
                if os.path.exists(path):
                    os.remove(path)
            messages.error(request, str(exc))
            return render(request, 'predict.html')
        if model_pickle['method'] != select_model:
            for path in (temporary_model, temporary_data):
                if os.path.exists(path):
                    os.remove(path)
            messages.error(request, 'Selected analysis mode is inconsistent with the signed model bundle.')
            return render(request, 'predict.html')

        filemd5 = get_file_md5(temporary_data)
        modelmd5 = get_file_md5(temporary_model)
        projectid = validated_project_id('PRED' + '-' + select_model.replace('model_', '').upper()
                                         + '-' + modelmd5[:6] + '-' + filemd5[:6]
                                         + '-' + secrets.token_hex(4))

        if not os.path.exists(os.path.join(settings.MALER_TASK_ROOT, projectid)):
            newpath = os.path.join(settings.MALER_TASK_ROOT, projectid)
            os.makedirs(os.path.join(settings.MALER_TASK_ROOT, projectid), exist_ok=True)
            shutil.move(temporary_data, os.path.join(newpath, 'data.csv'))
            shutil.move(temporary_model, os.path.join(newpath, 'model.maler'))
        else:
            for path in (temporary_model, temporary_data):
                if os.path.exists(path):
                    os.remove(path)




    # blind_set = pd.read_csv(settings.MALER_TASK_ROOT + '/' + projectid + '/' + obj_file.name, header=0, index_col=0).T
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
    status = 'Preview'
    bundle_path = os.path.join(settings.MALER_TASK_ROOT, projectid, 'model.maler')
    if os.path.exists(bundle_path):
        try:
            model_pickle = _bundle_as_legacy_dict(bundle_path)
        except ModelBundleError as exc:
            messages.error(request, str(exc))
            return render(request, 'predict.html')
    else:
        # The bundled examples are application-controlled legacy artifacts.
        # User-uploaded pickle files are never written to this location.
        with open(settings.MALER_TASK_ROOT + '/' + projectid + '/' + 'pickle.pkl', 'rb') as f:
            model_pickle = pickle.load(f)
#model
    method, model_name, model, feature_names = \
        model_pickle['method'], model_pickle['name'], model_pickle['model'], model_pickle['feature_names']

    try:
        inputdata = read_uploaded_matrix(os.path.join(settings.MALER_TASK_ROOT, projectid, 'data.csv'))
    except (ValueError, UnicodeDecodeError) as exc:
        messages.error(request, str(exc))
        return render(request, 'predict.html')

    # sample table display
    if select_model != method:
        title = 'Selected analysis mode is inconsistent with the input model!'
        messages.success(request, title)
        return render(request, "predict.html")

    if select_model == 'model_sur':
        blind_set,validation_set = pred_val_split(inputdata, datatype='survival')
        hist_values, bin_edges, bins_centers = data_hist(inputdata, datatype='survival')
    else:
        blind_set,validation_set = pred_val_split(inputdata,datatype=method)
        hist_values, bin_edges, bins_centers = data_hist(inputdata)
    display_samples = pd.DataFrame({'Validation': validation_set.shape, 'Blind': blind_set.shape},
                                   index=['Samples', 'Features'])
    display_samples_dict = display_samples.reset_index().rename(columns={'index': 'class'}).to_dict('records')
    # histogram
    hist_trace = [{
        'x': bin_edges,
        'y': hist_values,
        'type': "bar",
        'opacity': 0.9
    }]
    # data short view
    inputdata_display = inputdata.T.head(50).reset_index().rename(columns={'index': 'features'})
    inputdata_display.columns = [i.replace('.', '-') for i in inputdata_display.columns]
    inputdata_columns, title_str = mkcol(inputdata_display)
    inputdata_display = inputdata_display.to_dict("records")
    form_action = 'predict_result'
    preview_pickle = {
        'display_samples_dict': display_samples_dict,
        'hist_trace': hist_trace,
        'inputdata_display': inputdata_display,
        'inputdata_columns': inputdata_columns,
        'title_str': title_str,
        'status': status,
        'form_action': form_action,
    }
    with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'wb') as f:
        pickle.dump(preview_pickle, f)
    access_token = issue_task_token(projectid)
    return render(request, 'status.html', {
        'projectid': projectid,
        # 'model_md5': model_md5,
        'form_action': form_action,
        'display_samples_dict': script_json(display_samples_dict),
        'hist_trace': script_json(hist_trace, ensure_ascii=False, cls=JsonEncoder),
        'inputdata_display': script_json(html_safe_records(inputdata_display)),
        'inputdata_columns': script_json(inputdata_columns),
        'title_str': title_str,
        'status': status,
        'access_token': access_token,
    })


'''def predict_result(request,projectid):
    #predict
    select_model = request.POST.get('select_model')
    # projectid = request.POST.get('projectid')
    print(select_model)
    if not os.path.exists(os.path.join(settings.MALER_TASK_ROOT, projectid, 'pickle.pkl')):
        with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'rb') as f:
            preview_pickle = pickle.load(f)
        if preview_pickle['status'] == 'Preview':
            preview_pickle['status'] = 'Running'
            with open(settings.MALER_TASK_ROOT + '/' + projectid + '/preview_pickle.pkl', 'wb') as f:
                pickle.dump(preview_pickle, f)
        else:
            status = 'Running'
            form_action = preview_pickle['form_action']
            display_samples_dict = preview_pickle['display_samples_dict']
            hist_trace = preview_pickle['hist_trace']
            inputdata_display = preview_pickle['inputdata_display']
            inputdata_columns = preview_pickle['inputdata_columns']
            title_str = preview_pickle['title_str']
            return render(request, 'status.html', {
                'projectid': projectid,
                # 'model_md5': model_md5,
                'form_action': form_action,
                'display_samples_dict': json.dumps(display_samples_dict),
                'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
                'inputdata_display': json.dumps(inputdata_display),
                'inputdata_columns': json.dumps(inputdata_columns),
                'title_str': title_str,
                'status': status,
            })

    with open(settings.MALER_TASK_ROOT + '/' + projectid + '/' + 'pickle.pkl', 'rb') as f:
        model_pickle = pickle.load(f)
    # model
    method, model_name, model, feature_names = \
        model_pickle['method'], model_pickle['name'], model_pickle['model'], model_pickle['feature_names']

    inputdata = pd.read_csv(settings.MALER_TASK_ROOT + '/' + projectid + '/data.csv', header=0, index_col=0).T

    if select_model == 'model_sur':
        blind_set, validation_set = pred_val_split(inputdata, datatype='survival')
    else:
        blind_set, validation_set = pred_val_split(inputdata)

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

    shutil.rmtree(os.path.join(settings.MALER_TASK_ROOT, projectid))
    return render(request, 'predict_result.html', {
        'projectid': projectid,
        'method': method,
        'showtable': showtable,
        'predict_reports_dict': json.dumps(predict_reports_dict),
        'surv_plot': json.dumps(surv_plot),
    })
'''



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
def pred_val_split(data,datatype='other'):
    blind_set, val_set = pd.DataFrame(),pd.DataFrame()
    data = data.apply(pd.to_numeric, errors='ignore')
    if datatype != 'survival':
        # if data.columns[0].lower() == 'label':
        # 类别数，是否为数字判断是否存在label列
        # if len(np.unique(data.iloc[:,0])) < 30 and (not (np.issubdtype(data.iloc[0,0],np.integer) or np.issubdtype(data.iloc[0,0],np.floating))):
        # if len(np.unique(data.iloc[:, 0])) < 30 and (isinstance(data.iloc[0,0],str)):
        if datatype== 'model_bclass' or datatype== 'model_mclass': #分类标签为字符串，回归只能设置行名为label
            observed = data.iloc[:, 0].dropna()
            if str(data.columns[0]).lower() in ('label', 'outcome', 'class') or (
                    data.iloc[:, 0].nunique() < 30 and len(observed)
                    and isinstance(observed.iloc[0], str)):
                blind_set = data[data.iloc[:,:1].isna().T.any()]
                if len(blind_set)>0:
                    blind_set = blind_set.drop(labels=blind_set.columns[0], axis=1)
                val_set = data[~data.index.isin(blind_set.index)]
            else:
                blind_set = data
        elif datatype == 'model_reg':
            if data.columns[0].lower() == 'label':
                blind_set = data[data.iloc[:,:1].isna().T.any()]
                if len(blind_set)>0:
                    blind_set = blind_set.drop(labels=blind_set.columns[0], axis=1)
                val_set = data[~data.index.isin(blind_set.index)]
            else:
                blind_set = data
    else:
        if len(data.columns) >= 2 and str(data.columns[0]).lower() == 'status' and str(data.columns[1]).lower() == 'time':
            blind_set = data[data.iloc[:,:2].isna().T.any()]
            if len(blind_set)>0:
                blind_set = blind_set.drop(labels=blind_set.columns[:2], axis=1)
            val_set = data[~data.index.isin(blind_set.index)]
        else:
            blind_set = data
    blind_set = blind_set.apply(pd.to_numeric, errors='ignore')
    return blind_set,val_set
