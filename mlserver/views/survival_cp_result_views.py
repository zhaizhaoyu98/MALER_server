from django.shortcuts import render

import os, shutil, copy, pickle, json, random, string
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score,cross_validate, GridSearchCV, KFold,StratifiedKFold,RepeatedKFold
from sksurv.datasets import get_x_y
from sksurv.svm import FastKernelSurvivalSVM,FastSurvivalSVM
from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest,ExtraSurvivalTrees,GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import cumulative_dynamic_auc
from lifelines.statistics import logrank_test

from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5, df2bp, split_train_test, JsonEncoder
from mlserver.views.classification_cp_result_views import md5_convert, surv_para_group
from mlserver.views.survival_oc_result_views import sur_data_process, cox_selection, \
    sur_RSKFold, FSS_fun, train_estimator, mk_surv_data,mk_surv_layout, time_dependent_auc, \
    mk_auc_line, pre_screening, train_top3, BSS_fun
import warnings
warnings.filterwarnings("ignore")


def survival_cp_result(request, projectid):
    feature_select_method = request.POST.get('feature_select_method')
    # file_upload_type = request.POST.get('file_upload_type')
    # print('feature_select_method: ', feature_select_method)
    # select_model = request.POST.get('select_model')
    # select_child_model = request.POST.get('select_child_model').replace('task_','')
    # select_child_model = 'survivalsvm'
    '''
    sur_model = 'GradientBoostingSurvival'
    select_model = Survival_gradientboosting()
    sur_model = Survival_gradientboosting(Loss='coxph', Max_depth=3, Min_samples_split=2,
                                                 Min_samples_leaf=1, Max_features=None,
                                                 N_estimators=100, Learning_rate=0.1)
    '''

    model_md5 = request.POST.get('model_md5')
    with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
        model_set = pickle.load(f)
    print(model_set)
    sur_model, sur_model_name = model_set[model_md5]['model'], model_set[model_md5]['model_name']
    # sur_model, sur_model_name = select_sur_model(request)


    # get project id
    # projectid = request.POST.get('projectid')
    # if projectid == '': projectid = 'None'

    if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'cp_cache.pkl')):
        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
        # token = ''.join(random.sample(string.digits + string.ascii_letters, 6))
        # token = request.POST.get('random_token')
        # if file_upload_type == 'user_data':
        #     '''
        #     IMPORRT DATA
        #     '''
        #     obj_file = request.FILES.get('upload_file')
        #     f = open(os.path.join(STATIC_ROOT, 'cache', obj_file.name), 'wb')
        #     for line in obj_file.chunks():
        #         f.write(line)
        #     f.close()
        #
        #     filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', obj_file.name))
        #
        #     projectid = 'RC-' + filemd5[:6] + '-' + token
        #     newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        #     os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        #     shutil.move(STATIC_ROOT + '/cache/' + obj_file.name, newpath)
        #     # shutil.move(STATIC_ROOT + '/cache/' + obj_label.name, newpath)
        #     # rename
        #     os.rename(STATIC_ROOT + '/cache/' + projectid + '/' + obj_file.name, \
        #               STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv")
        #     # os.rename(STATIC_ROOT + '/cache/' + projectid + '/' + obj_label.name, \
        #     #           STATIC_ROOT + '/cache/' + projectid + '/' + "label.csv")
        #     inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv", header=0,
        #                             index_col=0).T
        # else:
        #     filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/survival_example.csv'))
        #     filename = 'survival_example.csv'
        #     projectid = 'RC-' + filemd5[:6] + '-' + token
        #     os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        #     newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        #     shutil.copy(STATIC_ROOT + '/cache/example/' + filename, newpath)
        #     os.rename(newpath + '/' + filename, \
        #               newpath + '/' + "load_data.csv")
        #     inputdata = pd.read_csv(STATIC_ROOT + '/cache/example/' + filename, header=0, index_col=0).T
        '''
        IMPORRT DATA
        '''
        # file load
        # upload_file = request.FILES.get('upload_file')
        # f = open(os.path.join(STATIC_ROOT, 'cache', upload_file.name), 'wb')
        # for line in upload_file.chunks():
        #     f.write(line)
        # f.close()

        # upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', upload_file.name))
        # token = ''.join(random.sample(string.digits + string.ascii_letters, 6))
        # projectid = 'SC-' + upload_file_md5[:6] + '-' + token

        # newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        # os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        # shutil.move(STATIC_ROOT + '/cache/' + upload_file.name, newpath)
        # rename
        # os.rename(STATIC_ROOT + '/cache/' + projectid + '/' + upload_file.name, \
        #           STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv")
        '''
        projectid='SC-e8ce60-FSS'
        data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + 'load_breast_cancer.csv', header=0, index_col=0).T
        '''
        # inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv", header=0, index_col=0).T
        train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')

        x2, y2 = sur_data_process(train_set)
        # x2, vaildation_data, y2, vaildation_label = train_test_split(x, y, random_state=10, train_size=0.7,
        #                                                              stratify=y['Status'])
        if len(test_set) > 0:
            validation_data, validation_label = sur_data_process(test_set)
        cv = KFold(n_splits=5, shuffle=True, random_state=10)
        features = cox_selection(x2, y2)
        # features,ss2 = cox_selection(x2,y2)
        x3 = x2[features]
        train_index, test_index = sur_RSKFold(x3, y2)
        if feature_select_method != 'TopK':
            if feature_select_method == 'FSS':
                sf, ms = FSS_fun(features, sur_model, x3, y2, cv, n_jobs=1)
            else:
                sf, ms = BSS_fun(features, sur_model, x3, y2, cv, n_jobs=1)
            max_index = np.array(ms).argmax()
            max_score = max(ms)
            max_features = (sf[:max_index + 1])


            preds, tests, res = [], [], []
            for i in range(len(train_index)):
                xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                xtrain, xtest = xtrain[max_features], xtest[max_features]
                estimator, test_acc, predict = train_estimator(sur_model, xtrain, ytrain, xtest, ytest)
                tests.append(test_acc), res.append(estimator), preds.append(predict)
        else:
            clf_num, ms = pre_screening(x3, y2, sur_model, features)
            tests, estimators, mean_accs, preds, res = train_top3(sur_model, x3, y2, clf_num, train_index,
                                                                             test_index, features)
            max_features = res
        tmodels = copy.deepcopy(sur_model)
        tmodels.fit(x3[max_features], y2)

        test_acc_reports = pd.DataFrame(data=tests)
        test_acc_reports.columns = [sur_model_name]
        test_acc_reports_dict = df2bp(test_acc_reports)
        test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                     3)
        test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        test_acc_describe_dict = test_acc_describe_.to_dict('records')

        line_chart_data = []
        line_trace = {
            'mode': 'lines+markers',
            'name': sur_model_name,
            'type': 'scatter',
            'x': list(range(1, len(ms)+1)),
            'y': ms
        }
        line_chart_data.append(line_trace)

        parameter, test_acc, best_esti = [], [], []
        feature_names = []
        best_esti.append(tmodels)
        parameter.append(str(tmodels.get_params()))
        test_acc.append(test_acc_describe.iloc[0, 0])
        feature_names.append(str(max_features))
        final_reports = {'Mean C-index': test_acc,
                         'parameter': parameter,
                         'feature_names': feature_names, }
        final_reports = pd.DataFrame(final_reports, index=[sur_model_name]).reset_index().rename(
            columns={'index': 'Method'})
        final_reports_dict = final_reports.to_dict('records')

        data_median = tmodels.predict(pd.DataFrame(x3[max_features].median()).T)[0]
        surv_trace, resultp = mk_surv_data(sur_model_name, x3[max_features],y2,best_esti[0],data_median)
        surv_layout = mk_surv_layout(sur_model_name, resultp)
        surv_data = {'surv_trace': surv_trace, 'surv_layout': surv_layout}

        # validation
        vsurv_trace, vresultp = mk_surv_data(sur_model_name, validation_data[max_features],validation_label,best_esti[0],data_median)
        vsurv_layout = mk_surv_layout(sur_model_name, vresultp)
        vsurv_data = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}

        vlinedata = []
        va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(best_esti[0], validation_data,
                                                                 validation_label, y2,
                                                                 max_features, sur_model_name)

        vlinetrace = mk_auc_line(sur_model_name, va_times, rsf_auc, mean_auc, cindex)
        vlinedata.append(vlinetrace)
        # pickle
        surv_pickle = {
            'sur_model_name': sur_model_name,
            'line_chart_data': line_chart_data,
            'test_acc_reports_dict': test_acc_reports_dict,
            'test_acc_describe_dict': test_acc_describe_dict,
            'final_reports_dict': final_reports_dict,
            'surv_data': surv_data,
            'vsurv_data': vsurv_data,
            'vlinedata': vlinedata
        }
        # make cache
        cp_cache = {}
        para_str = feature_select_method + final_reports['Method'][0] + str(final_reports['parameter'][0])
        para_md5 = md5_convert(para_str)[:6]
        # add parameter md5 and feature select method
        final_reports['md5'], final_reports['fsm'] = para_md5, feature_select_method
        final_reports_dict = final_reports.to_dict('records')

        cp_cache[para_md5] = surv_pickle
        cp_cache['reports'] = final_reports

        with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                  'wb') as f:
            pickle.dump(cp_cache, f)

    else:
        with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
            cp_cache = pickle.load(f)

        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/data.csv', header=0, index_col=0).T
        train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')

        x2, y2 = sur_data_process(train_set)
        if len(test_set) > 0:
            validation_data, validation_label = sur_data_process(test_set)
        # x2, vaildation_data, y2, vaildation_label = train_test_split(x, y, random_state=10, train_size=0.7,
        #                                                              stratify=y['Status'])
        cv = KFold(n_splits=5, shuffle=True, random_state=10)
        features = cox_selection(x2, y2)
        # features,ss2 = cox_selection(x2,y2)
        x3 = x2[features]
        train_index, test_index = sur_RSKFold(x3, y2)
        if feature_select_method != 'TopK':
            if feature_select_method == 'FSS':
                sf, ms = FSS_fun(features, sur_model, x3, y2, cv, n_jobs=1)
            else:
                sf, ms = BSS_fun(features, sur_model, x3, y2, cv, n_jobs=1)
            max_index = np.array(ms).argmax()
            max_score = max(ms)
            max_features = (sf[:max_index + 1])

            preds, tests, res = [], [], []
            for i in range(len(train_index)):
                xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                xtrain, xtest = xtrain[max_features], xtest[max_features]
                estimator, test_acc, predict = train_estimator(sur_model, xtrain, ytrain, xtest, ytest)
                tests.append(test_acc), res.append(estimator), preds.append(predict)
        else:
            clf_num, ms = pre_screening(x3, y2, sur_model, features)
            tests, estimators, mean_accs, preds, res = train_top3(sur_model, x3, y2, clf_num, train_index,
                                                                  test_index, features)
            max_features = res

        tmodels = copy.deepcopy(sur_model)
        tmodels.fit(x3[max_features], y2)

        select_str = feature_select_method + sur_model_name + str(tmodels.get_params())
        select_md5 = md5_convert(select_str)[:6]

        if select_md5 not in cp_cache.keys():

            test_acc_reports = pd.DataFrame(data=tests)
            test_acc_reports.columns = [sur_model_name]
            test_acc_reports_dict = df2bp(test_acc_reports)
            test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                         3)
            test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
            test_acc_describe_dict = test_acc_describe_.to_dict('records')

            line_chart_data = []
            line_trace = {
                'mode': 'lines+markers',
                'name': sur_model_name,
                'type': 'scatter',
                'x': list(range(1, len(ms)+1)),
                'y': ms
            }
            line_chart_data.append(line_trace)

            parameter, test_acc, best_esti = [], [], []
            feature_names = []
            best_esti.append(tmodels)
            parameter.append(str(tmodels.get_params()))
            test_acc.append(test_acc_describe.iloc[0, 0])
            feature_names.append(str(max_features))
            final_reports = {'Mean C-index': test_acc,
                             'parameter': parameter,
                             'feature_names': feature_names, }
            final_reports = pd.DataFrame(final_reports, index=[sur_model_name]).reset_index().rename(
                columns={'index': 'Method'})
            final_reports['md5'], final_reports['fsm'] = select_md5, feature_select_method
            final_reports_dict = final_reports.to_dict('records')

            data_median = tmodels.predict(pd.DataFrame(x3[max_features].median()).T)[0]
            surv_trace, resultp = mk_surv_data(sur_model_name, x3[max_features], y2, best_esti[0], data_median)
            surv_layout = mk_surv_layout(sur_model_name, resultp)
            surv_data = {'surv_trace': surv_trace, 'surv_layout': surv_layout}

            # validation
            vsurv_trace, vresultp = mk_surv_data(sur_model_name, validation_data[max_features], validation_label,
                                                 best_esti[0], data_median)
            vsurv_layout = mk_surv_layout(sur_model_name, vresultp)
            vsurv_data = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}

            vlinedata = []
            va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(best_esti[0], validation_data,
                                                                     validation_label, y2,
                                                                     max_features, sur_model_name)

            vlinetrace = mk_auc_line(sur_model_name, va_times, rsf_auc, mean_auc, cindex)
            vlinedata.append(vlinetrace)
            # pickle
            surv_pickle = {
                'sur_model_name': sur_model_name,
                'line_chart_data': line_chart_data,
                'test_acc_reports_dict': test_acc_reports_dict,
                'test_acc_describe_dict': test_acc_describe_dict,
                'final_reports_dict': final_reports_dict,
                'surv_data': surv_data,
                'vsurv_data': vsurv_data,
                'vlinedata': vlinedata
            }
            para_str = feature_select_method + final_reports['Method'][0] + str(final_reports['parameter'][0])
            para_md5 = md5_convert(para_str)[:6]
            final_reports = pd.concat([cp_cache['reports'], final_reports], axis=0).drop_duplicates(keep='last')
            final_reports_dict = final_reports.to_dict('records')
            cp_cache[para_md5] = surv_pickle
            cp_cache['reports'] = final_reports

            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                      'wb') as f:
                pickle.dump(cp_cache, f)

        else:
            final_reports_dict = cp_cache['reports'].to_dict('records')
            sur_model_name = cp_cache[select_md5]['sur_model_name']
            line_chart_data = cp_cache[select_md5]['line_chart_data']
            test_acc_reports_dict = cp_cache[select_md5]['test_acc_reports_dict']
            test_acc_describe_dict = cp_cache[select_md5]['test_acc_describe_dict']
            surv_data = cp_cache[select_md5]['surv_data']
            vsurv_data = cp_cache[select_md5]['vsurv_data']
            vlinedata = cp_cache[select_md5]['vlinedata']



    return render(request, 'survival_cp_result.html', {
        'projectid': projectid,
        'sur_model_name': sur_model_name,
        'line_chart_data': json.dumps(line_chart_data),
        'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
        'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
        'final_reports_dict': json.dumps(final_reports_dict),
        'surv_data': json.dumps(surv_data),
        'vsurv_data': json.dumps(vsurv_data),
        'vlinedata': json.dumps(vlinedata),
    })

def show_prev_page(request, projectid_paramd5):
    if len(projectid_paramd5.split('-')[2]) != 4:
        if len(projectid_paramd5.split('_')) == 2:
            projectid = projectid_paramd5.split('_')[0]
            paramd5 = projectid_paramd5.split('_')[1]
            if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl'):
                status = 'Running'
                return render(request, 'status.html', {
                    'status': status,
                    'projectid': projectid,
                })
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)
        else:
            projectid = projectid_paramd5
            if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl'):
                status = 'Running'
                # return render(request, 'status.html', {
                #     'status': status,
                #     'projectid': projectid,
                # })
                with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'rb') as f:
                    preview_pickle = pickle.load(f)
                feature_select_method = preview_pickle['feature_select_method']
                form_action = preview_pickle['form_action']
                display_samples_dict = preview_pickle['display_samples_dict']
                hist_trace = preview_pickle['hist_trace']
                inputdata_display = preview_pickle['inputdata_display']
                inputdata_columns = preview_pickle['inputdata_columns']
                title_str = preview_pickle['title_str']
                return render(request, 'status.html', {
                    'projectid': projectid,
                    'form_action': form_action,
                    'status': status,
                    'feature_select_method': feature_select_method,
                    # 'model_md5': model_md5,
                    'display_samples_dict': json.dumps(display_samples_dict),
                    'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
                    'inputdata_display': json.dumps(inputdata_display),
                    'inputdata_columns': json.dumps(inputdata_columns),
                    'title_str': title_str,
                })
            else:
                with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                    cp_cache = pickle.load(f)

                with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
                    model_pickle = pickle.load(f)
                status = 'Running'
                if len(model_pickle.keys()) > cp_cache['reports'].shape[0]:
                    with open(STATIC_ROOT + '/cache/' + projectid + '/preview_pickle.pkl', 'rb') as f:
                        preview_pickle = pickle.load(f)
                    feature_select_method = preview_pickle['feature_select_method']
                    form_action = preview_pickle['form_action']
                    display_samples_dict = preview_pickle['display_samples_dict']
                    hist_trace = preview_pickle['hist_trace']
                    inputdata_display = preview_pickle['inputdata_display']
                    inputdata_columns = preview_pickle['inputdata_columns']
                    title_str = preview_pickle['title_str']
                    return render(request, 'status.html', {
                        'projectid': projectid,
                        'form_action': form_action,
                        'status': status,
                        'feature_select_method': feature_select_method,
                        # 'model_md5': model_md5,
                        'display_samples_dict': json.dumps(display_samples_dict),
                        'hist_trace': json.dumps(hist_trace, ensure_ascii=False, cls=JsonEncoder),
                        'inputdata_display': json.dumps(inputdata_display),
                        'inputdata_columns': json.dumps(inputdata_columns),
                        'title_str': title_str,
                    })
            # load pickle
            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
                cp_cache = pickle.load(f)
            paramd5 = cp_cache['reports']['md5'][0]
        # projectid = projectid_paramd5.split('_')[0]
        # paramd5 = projectid_paramd5.split('_')[1]
        # # load pickle
        # with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
        #     cp_cache = pickle.load(f)

        final_reports_dict = cp_cache['reports'].to_dict('records')
        sur_model_name = cp_cache[paramd5]['sur_model_name']
        line_chart_data = cp_cache[paramd5]['line_chart_data']
        test_acc_reports_dict = cp_cache[paramd5]['test_acc_reports_dict']
        test_acc_describe_dict = cp_cache[paramd5]['test_acc_describe_dict']
        surv_data = cp_cache[paramd5]['surv_data']
        vsurv_data = cp_cache[paramd5]['vsurv_data']
        vlinedata = cp_cache[paramd5]['vlinedata']

        return render(request, 'survival_cp_result.html', {
            'projectid': projectid,
            'final_reports_dict': json.dumps(final_reports_dict),
            'sur_model_name': sur_model_name,
            'line_chart_data': json.dumps(line_chart_data),
            'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
            'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
            'surv_data': json.dumps(surv_data),
            'vsurv_data': json.dumps(vsurv_data),
            'vlinedata': json.dumps(vlinedata),
            'change_page': True,
        })
    else:
        projectid = projectid_paramd5
        if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl'):
            status = 'Running'
            return render(request, 'status.html', {
                'status': status,
                'projectid': projectid,
            })
        else:
            with open(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl', 'rb') as f:
                surv_pickle = pickle.load(f)

            line_chart_data = surv_pickle['line_chart_data']
            test_acc_reports_dict = surv_pickle['test_acc_reports_dict']
            test_acc_describe_dict = surv_pickle['test_acc_describe_dict']
            max_reports_dict = surv_pickle['max_reports_dict']
            surv_dict = surv_pickle['surv_dict']
            subplot_sur = surv_pickle['subplot_sur']
            para_dict = surv_pickle['para_dict']
            vsubplot_sur = surv_pickle['vsubplot_sur']
            vpara_dict = surv_pickle['vpara_dict']
            vlinedata = surv_pickle['vlinedata']
            return render(request, 'survival_oc_result.html', {
                'projectid': projectid,
                'line_chart_data': json.dumps(line_chart_data),
                'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
                'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
                'max_reports_dict': json.dumps(max_reports_dict),
                'surv_dict': json.dumps(surv_dict),
                'subplot_sur': json.dumps(subplot_sur),
                'para_dict': json.dumps(para_dict),
                'vsubplot_sur': json.dumps(vsubplot_sur),
                'vpara_dict': json.dumps(vpara_dict),
                'vlinedata': json.dumps(vlinedata),
            })



'''
CUSTOMIZED MODELS
'''
def Survival_svm(Alpha=1,Degree=3,Coef0=1,Gamma=None,Random_state=10,Max_iter=20,Optimizer="rbtree",Kernel='rbf'):
    sur_svm = FastKernelSurvivalSVM(alpha=Alpha,degree=Degree,coef0=Coef0,gamma=Gamma,random_state=Random_state,
                                    max_iter=Max_iter,optimizer=Optimizer,kernel=Kernel)
    return sur_svm

def Survival_tree(Splitter='best',Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features=None):
    sur_tree = SurvivalTree(splitter=Splitter,max_depth=Max_depth,min_samples_split=Min_samples_split,
                            min_samples_leaf=Min_samples_leaf,max_features=Max_features,random_state=10)
    return sur_tree

def Survival_extratrees(N_estimators=100,Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features='sqrt'):
    sur_extratrees = ExtraSurvivalTrees(n_estimators=N_estimators,max_depth=Max_depth,min_samples_split=Min_samples_split,
                                       min_samples_leaf=Min_samples_leaf,max_features=Max_features,
                                       n_jobs=1,random_state=10)
    return sur_extratrees

def Survival_randomforest(N_estimators=100,Max_depth=None,Min_samples_split=6,Min_samples_leaf=3,Max_features=None):
    sur_randomforest = RandomSurvivalForest(n_estimators=N_estimators,max_depth=Max_depth,min_samples_split=Min_samples_split,
                                       min_samples_leaf=Min_samples_leaf,max_features=Max_features,
                                       n_jobs=1,random_state=10)
    return sur_randomforest

def Survival_gradientboosting(Loss='coxph',Learning_rate=0.1,N_estimators=100,Min_samples_split=2,
                              Min_samples_leaf=1,Max_depth=3,Max_features=None,Subsample=1.0):
    sur_gb = GradientBoostingSurvivalAnalysis(loss=Loss,learning_rate=Learning_rate,n_estimators=N_estimators,
                                              min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,
                                             max_depth=Max_depth,max_features=Max_features,subsample=Subsample)
    return sur_gb


'''
METHODS
'''
def select_sur_model(request):
    select_child_model = request.POST.get('select_child_model').replace('task_', '')
    if select_child_model == 'survivalsvm':
        select_model_name = 'SurvivalSVM'
        kernel, optimizer, alpha, degree, gamma, coef0 = request.POST.get('survivalsvm_kernel'), \
                                                        request.POST.get('survivalsvm_optimizer'), \
                                                        float(request.POST.get('survivalsvm_alpha')), \
                                                        request.POST.get('survivalsvm_degree'), \
                                                        request.POST.get('survivalsvm_gamma'), \
                                                        request.POST.get('survivalsvm_coef0')
        if kernel == 'linear':
            select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Optimizer=optimizer)
        elif kernel == 'ploy':
            if gamma == '': gamma = None
            degree = int(degree)
            coef0 = float(coef0)
            select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Degree=degree, Gamma=gamma, Coef0=coef0)
        elif kernel == 'rbf':
            if gamma == '': gamma = None
            select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Gamma=gamma)
        elif kernel == 'sigmoid':
            coef0 = float(coef0)
            select_model = Survival_svm(Kernel=kernel, Alpha=alpha, Coef0=coef0)
        else:
            select_model = Survival_svm(Kernel=kernel, Alpha=alpha)
    elif select_child_model == 'survivaltree':
        select_model_name = 'SurvivalTree'
        splitter, max_depth, min_samples_split, min_samples_leaf, max_features = \
            request.POST.get('survivaltree_splitter'),request.POST.get('survivaltree_max_depth'), \
            request.POST.get('survivaltree_min_samples_split'), request.POST.get('survivaltree_min_samples_leaf'), \
            request.POST.get('survivaltree_max_features')
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = Survival_tree(Splitter=splitter, Max_depth=max_depth,
                                     Min_samples_split=min_samples_split, Min_samples_leaf=min_samples_leaf,
                                     Max_features=max_features)
    elif select_child_model == 'extrasurvivaltrees':
        select_model_name = 'ExtraSurvivalTrees'
        max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators = \
            request.POST.get('extrasurvivaltrees_max_depth'), request.POST.get('extrasurvivaltrees_min_samples_split'), \
            request.POST.get('extrasurvivaltrees_min_samples_leaf'), request.POST.get('extrasurvivaltrees_max_features'), \
            int(request.POST.get('extrasurvivaltrees_n_estimators'))
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = Survival_extratrees(Max_depth=max_depth, Min_samples_split=min_samples_split,
                                           Min_samples_leaf=min_samples_leaf, Max_features=max_features,
                                           N_estimators=n_estimators)
    elif select_child_model == 'randomsurvivalforest':
        select_model_name = 'RandomSurvivalForest'
        max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators = \
            request.POST.get('randomsurvivalforest_max_depth'), np.float(request.POST.get('randomsurvivalforest_min_samples_split')), \
            np.float(request.POST.get('randomsurvivalforest_min_samples_leaf')), request.POST.get('randomsurvivalforest_max_features'), \
            int(request.POST.get('randomsurvivalforest_n_estimators'))
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = Survival_randomforest(Max_depth=max_depth, Min_samples_split=min_samples_split,
                                             Min_samples_leaf=min_samples_leaf, Max_features=max_features,
                                             N_estimators=n_estimators)
    else:
        select_model_name = 'GradientBoostingSurvival'
        loss, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators, learning_rate ,subsample= \
            request.POST.get('gradientboostingsurvival_loss'), request.POST.get('gradientboostingsurvival_max_depth'), \
            request.POST.get('gradientboostingsurvival_min_samples_split'), request.POST.get('gradientboostingsurvival_min_samples_leaf'), \
            request.POST.get('gradientboostingsurvival_max_features'), int(request.POST.get('gradientboostingsurvival_n_estimators')), \
            np.float(request.POST.get('gradientboostingsurvival_learning_rate')), \
            np.float(request.POST.get('gradientboostingsurvival_subsample'))
        print(loss, max_depth, min_samples_split, min_samples_leaf, max_features, n_estimators, learning_rate,subsample)
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = Survival_gradientboosting(Loss='coxph', Max_depth=max_depth, Min_samples_split=min_samples_split,
                                                 Min_samples_leaf=min_samples_leaf, Max_features=max_features,
                                                 N_estimators=n_estimators, Learning_rate=learning_rate,Subsample=subsample)
    return select_model, select_model_name


