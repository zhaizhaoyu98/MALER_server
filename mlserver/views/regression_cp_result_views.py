import json

from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, random, string, shutil, pickle, re
from sklearn.model_selection import RepeatedKFold,KFold
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import f_classif,chi2,VarianceThreshold,mutual_info_classif,f_regression
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings("ignore")

from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5, split_train_test
from mlserver.views.regression_oc_result_views import mkvregpredplot, mkvreportbarplot, JsonEncoder
from mlserver.views.classification_cp_result_views import md5_convert
from mlserver.views.survival_cp_result_views import surv_para_group

def regression_cp_result(request, projectid):
    feature_select_method = request.POST.get('feature_select_method')
    # file_upload_type = request.POST.get('file_upload_type')
    # print('feature_select_method: ', feature_select_method)
    # print('regsvm_degree: ',request.POST.get('regsvm_degree') == None)
    # print('regsvm_gamma: ', request.POST.get('regsvm_gamma'))
    # reg_model_name = 'LinearRegression'
    # reg_cust_model = LinearRegression()  # 选择模型

    model_md5 = request.POST.get('model_md5')
    with open(STATIC_ROOT + '/cache/' + projectid + '/model_pickle.pkl', 'rb') as f:
        model_set = pickle.load(f)
    print(model_set)
    reg_cust_model, reg_model_name = model_set[model_md5]['model'], model_set[model_md5]['model_name']
    # reg_cust_model, reg_model_name = select_reg_model(request)
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
        #     filemd5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/regression_example.csv'))
        #     filename = 'regression_example.csv'
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
        # projectid = 'RC-' + upload_file_md5[:6] + '-' + token

        # newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        # os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        # shutil.move(STATIC_ROOT + '/cache/' + upload_file.name, newpath)
        # rename
        # os.rename(STATIC_ROOT + '/cache/' + projectid + '/' + upload_file.name, \
        #           STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv")
        '''
        projectid='RC-19f4d5-L3BYDx'
        data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv", header=0, index_col=0).T

        '''
        # inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "load_data.csv", header=0, index_col=0).T
        train_set, test_set, blind_set = split_train_test(inputdata)

        nordata4, nor_age4 = regression_preprocess(train_set)
        if len(test_set) > 0:
            validation_data, validation_label = regression_preprocess(test_set)
        # print('test_set:', test_set)
        # nordata4, vaildation_data, nor_age4, vaildation_label = train_test_split(x_dum, y, random_state=10,
        #                                                                          train_size=0.7)  # 分验证集
        features = selectkbest_top20(nordata4, nor_age4, score_func=f_regression, k=50)
        nordata4 = nordata4[features]

        train_index, test_index = RegressionKFold(nordata4, nor_age4)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=10)

        # Alphas=[0.0001,0.001,0.005,0.05,0.1,0.01]

        # fss,bss
        if feature_select_method == 'FSS' or feature_select_method == 'BSS':
            if feature_select_method == 'BSS':
                sf, ms = BSS_fun(features, reg_cust_model, nordata4, nor_age4, cv, n_jobs=6)
            else:
                sf, ms = FSS_fun(features, reg_cust_model, nordata4, nor_age4, cv, n_jobs=6)
            max_index = np.array(ms).argmax()
            # max_index = ms.index(np.nanmax(ms))
            max_score = max(ms)
            max_features = (sf[:max_index + 1])
            preds, tests, res = [], [], []
            for i in range(len(train_index)):
                xtrain, ytrain = nordata4.iloc[train_index[i], :], nor_age4[train_index[i]]
                xtest, ytest = nordata4.iloc[test_index[i], :], nor_age4[test_index[i]]
                xtrain, xtest = xtrain[max_features], xtest[max_features]
                estimator, test_acc, predict = train_estimator(reg_cust_model, xtrain, ytrain, xtest, ytest)
                tests.append(test_acc), res.append(estimator), preds.append(predict)
        else:
            clf_num, ms = pre_screening(nordata4, nor_age4, reg_cust_model, features)
            tests, estimators, mean_accs, preds, res = train_top3(reg_cust_model, nordata4, nor_age4,
                                                                                  clf_num, train_index, test_index,
                                                                                  features)

            # max_features = list(nordata4.iloc[:, clf_num].columns)
            max_features = res
        line_chart_data = []
        line_trace = {
            'mode': 'lines+markers',
            'name': reg_model_name,
            'type': 'scatter',
            'x': list(range(1, len(ms)+1)),
            'y': ms
        }
        line_chart_data.append(line_trace)

        cust_reports, cust_reports_describe = cust_cv_reports(preds, test_index, nor_age4, tests)

        cust_reports_dict = df2bp(cust_reports)
        cust_reports_describe_ = cust_reports_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        cust_reports_describe_dict = cust_reports_describe_.to_dict('records')

        tmodels = copy.deepcopy(reg_cust_model)
        tmodels.fit(nordata4[max_features], nor_age4)
        parameter, test_acc, best_esti = [], [], []
        feature_names = []
        best_esti.append(tmodels)
        parameter.append(tmodels.get_params())
        if reg_model_name == 'Ridge' or reg_model_name == 'Lasso':
            parameter[0]['alphas'] = tmodels.alpha_
        parameter[0] = str(parameter[0])
        test_acc.append(cust_reports_describe.iloc[0, 0])
        feature_names.append(str(max_features))
        final_reports = {'parameter': parameter,
                         'feature_names': feature_names,
                         'Mean R-square': cust_reports.mean()[0],
                         'MAE': cust_reports.mean()[1],
                         'MSE': cust_reports.mean()[2], }

        final_reports = pd.DataFrame(final_reports, index=[reg_model_name]).reset_index().rename(
            columns={'index': 'Method'})
        final_reports[['Mean R-square', 'MAE', 'MSE']] = np.round(final_reports[['Mean R-square', 'MAE', 'MSE']], 3)
        final_reports_dict = final_reports.to_dict('records')

        # validation
        val_report = reg_cust_val(best_esti,validation_data,validation_label,max_features, reg_model_name)

        validate_predict = best_esti[0].predict(validation_data[max_features])
        vregpred_trace = mkvregpredplot([validate_predict], validation_label, [reg_model_name])
        vreport_trace = []
        i=0
        for m in list(val_report.columns):
            subtrace = mkvreportbarplot(val_report[[m]])
            subtrace[0]['xaxis'], subtrace[0]['yaxis'] = 'x' + str(i + 1), 'y' + str(i + 1)
            i += 1
            vreport_trace.append(subtrace[0])

        val_report = np.round(val_report, 3)
        val_report = val_report.reset_index().rename(columns={'index': 'Method'})
        val_report_dict = val_report.to_dict('records')

        # pickle
        reg_pickle = {
            'reg_model_name': reg_model_name,
            'line_chart_data': line_chart_data,
            'cust_reports_dict': cust_reports_dict,
            'cust_reports_describe_dict': cust_reports_describe_dict,
            'final_reports_dict': final_reports_dict,
            'val_report_dict': val_report_dict,
            'vregpred_trace': vregpred_trace,
            'vreport_trace':vreport_trace
        }
        # make cache
        cp_cache = {}
        para_str = feature_select_method + final_reports['Method'][0] + str(final_reports['parameter'][0])
        para_md5 = md5_convert(para_str)[:6]
        # add parameter md5 and feature select method
        final_reports['md5'], final_reports['fsm'] = para_md5, feature_select_method
        final_reports_dict = final_reports.to_dict('records')

        cp_cache[para_md5] = reg_pickle
        cp_cache['reports'] = final_reports

        with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                  'wb') as f:
            pickle.dump(cp_cache, f)
    else:
        with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl', 'rb') as f:
            cp_cache = pickle.load(f)

        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + "data.csv", header=0, index_col=0).T
        train_set, test_set, blind_set = split_train_test(inputdata)

        nordata4, nor_age4 = regression_preprocess(train_set)
        if len(test_set) > 0:
            validation_data, validation_label = regression_preprocess(test_set)
        # nordata4, vaildation_data, nor_age4, vaildation_label = train_test_split(x_dum, y, random_state=10,
        #                                                                          train_size=0.7)  # 分验证集
        features = selectkbest_top20(nordata4, nor_age4, score_func=f_regression, k=50)
        nordata4 = nordata4[features]

        train_index, test_index = RegressionKFold(nordata4, nor_age4)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=10)

        # Alphas=[0.0001,0.001,0.005,0.05,0.1,0.01]

        # fss,bss
        if feature_select_method != 'TopK':
            sf, ms = BSS_fun(features, reg_cust_model, nordata4, nor_age4, cv, n_jobs=6)
            max_index = np.array(ms).argmax()
            # max_index = ms.index(np.nanmax(ms))
            max_score = max(ms)
            max_features = (sf[:max_index + 1])
            preds, tests, res = [], [], []
            for i in range(len(train_index)):
                xtrain, ytrain = nordata4.iloc[train_index[i], :], nor_age4[train_index[i]]
                xtest, ytest = nordata4.iloc[test_index[i], :], nor_age4[test_index[i]]
                xtrain, xtest = xtrain[max_features], xtest[max_features]
                estimator, test_acc, predict = train_estimator(reg_cust_model, xtrain, ytrain, xtest, ytest)
                tests.append(test_acc), res.append(estimator), preds.append(predict)

        else:
            clf_num, ms = pre_screening(nordata4, nor_age4, reg_cust_model, features)
            tests, estimators, mean_accs, preds, res = train_top3(reg_cust_model, nordata4, nor_age4,
                                                                  clf_num, train_index, test_index,
                                                                  features)

            # max_features = list(nordata4.iloc[:, clf_num].columns)
            max_features = res
        tmodels = copy.deepcopy(reg_cust_model)
        tmodels.fit(nordata4[max_features], nor_age4)
        paras = reg_cust_model.get_params()
        if reg_model_name == 'Ridge' or reg_model_name == 'Lasso':
            paras['alphas'] = tmodels.alpha_
        select_str = feature_select_method + reg_model_name + str(paras)
        select_md5 = md5_convert(select_str)[:6]


        if select_md5 not in cp_cache.keys():

            line_chart_data = []
            line_trace = {
                'mode': 'lines+markers',
                'name': reg_model_name,
                'type': 'scatter',
                'x': list(range(1, len(ms)+1)),
                'y': ms
            }
            line_chart_data.append(line_trace)

            cust_reports, cust_reports_describe = cust_cv_reports(preds, test_index, nor_age4, tests)

            cust_reports_dict = df2bp(cust_reports)
            cust_reports_describe_ = cust_reports_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
            cust_reports_describe_dict = cust_reports_describe_.to_dict('records')


            parameter, test_acc, best_esti = [], [], []
            feature_names = []
            best_esti.append(tmodels)
            parameter.append(tmodels.get_params())

            parameter[0] = str(parameter[0])
            test_acc.append(cust_reports_describe.iloc[0, 0])
            feature_names.append(str(max_features))
            final_reports = {'parameter': parameter,
                             'feature_names': feature_names,
                             'Mean R-square': cust_reports.mean()[0],
                             'MAE': cust_reports.mean()[1],
                             'MSE': cust_reports.mean()[2], }

            final_reports = pd.DataFrame(final_reports, index=[reg_model_name]).reset_index().rename(
                columns={'index': 'Method'})
            final_reports[['Mean R-square', 'MAE', 'MSE']] = np.round(final_reports[['Mean R-square', 'MAE', 'MSE']], 3)
            final_reports['md5'], final_reports['fsm'] = select_md5, feature_select_method
            final_reports_dict = final_reports.to_dict('records')

            # validation
            val_report = reg_cust_val(best_esti, validation_data, validation_label, max_features, reg_model_name)

            validate_predict = best_esti[0].predict(validation_data[max_features])
            vregpred_trace = mkvregpredplot([validate_predict], validation_label, [reg_model_name])
            vreport_trace = []
            i = 0
            for m in list(val_report.columns):
                subtrace = mkvreportbarplot(val_report[[m]])
                subtrace[0]['xaxis'], subtrace[0]['yaxis'] = 'x' + str(i + 1), 'y' + str(i + 1)
                i += 1
                vreport_trace.append(subtrace[0])

            val_report = np.round(val_report, 3)
            val_report = val_report.reset_index().rename(columns={'index': 'Method'})
            val_report_dict = val_report.to_dict('records')

            # pickle
            reg_pickle = {
                'reg_model_name': reg_model_name,
                'line_chart_data': line_chart_data,
                'cust_reports_dict': cust_reports_dict,
                'cust_reports_describe_dict': cust_reports_describe_dict,
                'final_reports_dict': final_reports_dict,
                'val_report_dict': val_report_dict,
                'vregpred_trace': vregpred_trace,
                'vreport_trace': vreport_trace
            }

            final_reports = pd.concat([cp_cache['reports'], final_reports], axis=0).drop_duplicates(keep='last')
            final_reports[['Mean R-square','MAE','MSE']] = np.round(final_reports[['Mean R-square','MAE','MSE']],3)
            final_reports_dict = final_reports.to_dict('records')
            cp_cache[select_md5] = reg_pickle
            cp_cache['reports'] = final_reports

            with open(STATIC_ROOT + '/cache/' + projectid + '/cp_cache.pkl',
                      'wb') as f:
                pickle.dump(cp_cache, f)
        else:
            final_reports_dict = cp_cache['reports'].to_dict('records')
            reg_model_name = cp_cache[select_md5]['reg_model_name']
            line_chart_data = cp_cache[select_md5]['line_chart_data']
            cust_reports_dict = cp_cache[select_md5]['cust_reports_dict']
            cust_reports_describe_dict = cp_cache[select_md5]['cust_reports_describe_dict']
            val_report_dict = cp_cache[select_md5]['val_report_dict']
            vregpred_trace = cp_cache[select_md5]['vregpred_trace']
            vreport_trace = cp_cache[select_md5]['vreport_trace']

    return render(request, 'regression_cp_result.html', {
        'projectid': projectid,
        'reg_model_name': reg_model_name,
        'line_chart_data': json.dumps(line_chart_data),
        'cust_reports_dict': json.dumps(cust_reports_dict),
        'cust_reports_describe_dict': json.dumps(cust_reports_describe_dict),
        'final_reports_dict': json.dumps(final_reports_dict),
        'val_report_dict': json.dumps(val_report_dict),
        'vregpred_trace': json.dumps(vregpred_trace,ensure_ascii=False, cls=JsonEncoder),
        'vreport_trace': json.dumps(vreport_trace,ensure_ascii=False, cls=JsonEncoder),
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
                return render(request, 'status.html', {
                    'status': status,
                    'projectid': projectid,
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
        reg_model_name = cp_cache[paramd5]['reg_model_name']
        line_chart_data = cp_cache[paramd5]['line_chart_data']
        cust_reports_dict = cp_cache[paramd5]['cust_reports_dict']
        cust_reports_describe_dict = cp_cache[paramd5]['cust_reports_describe_dict']
        val_report_dict = cp_cache[paramd5]['val_report_dict']
        vregpred_trace = cp_cache[paramd5]['vregpred_trace']
        vreport_trace = cp_cache[paramd5]['vreport_trace']

        return render(request, 'regression_cp_result.html', {
            'projectid': projectid,
            'reg_model_name': reg_model_name,
            'line_chart_data': json.dumps(line_chart_data),
            'cust_reports_dict': json.dumps(cust_reports_dict),
            'cust_reports_describe_dict': json.dumps(cust_reports_describe_dict),
            'final_reports_dict': json.dumps(final_reports_dict),
            'val_report_dict': json.dumps(val_report_dict),
            'vregpred_trace': json.dumps(vregpred_trace, ensure_ascii=False, cls=JsonEncoder),
            'vreport_trace': json.dumps(vreport_trace, ensure_ascii=False, cls=JsonEncoder),
            'change_page': True,
        })
    else:
        projectid = projectid_paramd5
        if not os.path.exists(STATIC_ROOT + '/cache/' + projectid + '/regression_pickle.pkl'):
            status = 'Running'
            return render(request, 'status.html', {
                'status': status,
                'projectid': projectid,
            })
        else:
            with open(STATIC_ROOT + '/cache/' + projectid + '/regression_pickle.pkl', 'rb') as f:
                reg_pickle = pickle.load(f)

            line_chart_data = reg_pickle['line_chart_data']
            test_acc_reports_dict = reg_pickle['test_acc_reports_dict']
            test_acc_describe_dict = reg_pickle['test_acc_describe_dict']
            MAE_report_dict = reg_pickle['MAE_report_dict']
            MAE_report_describe_dict = reg_pickle['MAE_report_describe_dict']
            MSE_report_dict = reg_pickle['MSE_report_dict']
            MSE_report_describe_dict = reg_pickle['MSE_report_describe_dict']
            final_reports_dict = reg_pickle['final_reports_dict']
            vregpred_trace = reg_pickle['vregpred_trace']
            vreport_trace = reg_pickle['vreport_trace']
            val_report_dict = reg_pickle['val_report_dict']
            # radar_dict = reg_pickle['radar_dict']
            # radar_range = reg_pickle['radar_range']

            # print(vregpred_trace)
            return render(request, 'regression_oc_result.html', {
                'projectid': projectid,
                'line_chart_data': json.dumps(line_chart_data),
                'test_acc_reports_dict': json.dumps(test_acc_reports_dict),
                'test_acc_describe_dict': json.dumps(test_acc_describe_dict),
                'MAE_report_describe_dict': json.dumps(MAE_report_describe_dict),
                'MSE_report_describe_dict': json.dumps(MSE_report_describe_dict),
                'MAE_report_dict': json.dumps(MAE_report_dict),
                'MSE_report_dict': json.dumps(MSE_report_dict),
                'final_reports_dict': json.dumps(final_reports_dict),
                'vregpred_trace': json.dumps(vregpred_trace, ensure_ascii=False, cls=JsonEncoder),
                'vreport_trace': json.dumps(vreport_trace, ensure_ascii=False, cls=JsonEncoder),
                'val_report_dict': json.dumps(val_report_dict),
                # 'radar_dict': json.dumps(radar_dict),
                # 'radar_range': json.dumps(radar_range)
            })
'''
CUSTOMIZED MODELS
'''
def regression_linear(Fit_intercept=True,Positive=False):
    lr = LinearRegression(n_jobs=4,fit_intercept=Fit_intercept,positive=Positive)
    return lr

def regression_SVM(Kernel='rbf',Degree=3,Coef0=0,CC=1,Gamma='scale'):
    svm = SVR(kernel=Kernel,degree=Degree,coef0=Coef0,C=CC,gamma=Gamma,cache_size=5000,max_iter=2000)
    return svm

def regression_ridge(Alphas=[0.1, 1.0, 10.0],Fit_intercept=True,Positive=False,Gcv_mode='auto'):
    ridge = RidgeCV(alphas=Alphas,fit_intercept=Fit_intercept,gcv_mode=Gcv_mode)
    return ridge

def regression_lasso(Eps=0.001,N_alphas=100,Alphas=None,Fit_intercept=False,Selection='cyclic',Positive=False):
    lasso = LassoCV(eps=Eps,n_alphas=N_alphas,alphas=Alphas,fit_intercept=Fit_intercept,
                    selection=Selection,n_jobs=4,random_state=10,positive=Positive)
    return lasso

def regression_dtree(Criterion='squared_error',Splitter='best',Max_depth=None,Min_samples_split=2,Min_samples_leaf=1,Max_features=None):
    reg_dt = DecisionTreeRegressor(random_state=10,criterion=Criterion,splitter=Splitter,max_depth=Max_depth,
                                  min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,max_features=Max_features)
    return reg_dt

def regression_xgboost(Max_depth = 6,Learning_rate=0.3,N_estimators=100,Booster='gbtree',Gamma=0,Min_child_weight=1,
                           Colsample_bytree=1,Reg_alpha=0,Reg_lambda=1):
    reg_xgb = XGBRegressor(max_depth =Max_depth,learning_rate=Learning_rate,n_estimators=N_estimators,booster=Booster,
                           gamma=Gamma,min_child_weight=Min_child_weight,colsample_bytree=Colsample_bytree,
                           reg_alpha=Reg_alpha,reg_lambda=Reg_lambda,random_state=10)
    return reg_xgb

def regression_randomforest(N_estimators = 100,Criterion='squared_error',Max_depth=None,Min_samples_split=2,
                            Min_samples_leaf=1,Max_features=1.0):
    reg_rf = RandomForestRegressor(n_jobs=4,random_state=10,n_estimators = N_estimators,criterion=Criterion,max_depth=Max_depth,
                                  min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,max_features=Max_features)
    return reg_rf

def regression_adaboost(N_estimators=50,Learning_rate=1.0,Loss='linear'):
    reg_ada = AdaBoostRegressor(n_estimators=N_estimators,learning_rate=Learning_rate,loss=Loss,random_state=10)
    return reg_ada

def regression_GBR(Loss='squared_error',Learning_rate=0.1,N_estimators=100,Alpha=0.9,
                   Subsample=1.0,Min_samples_split=2,Min_samples_leaf=1,Max_depth=3,Max_features=None):
    feg_gbr = GradientBoostingRegressor(loss=Loss,learning_rate=Learning_rate,n_estimators=N_estimators,
                                       subsample=Subsample,min_samples_split=Min_samples_split,min_samples_leaf=Min_samples_leaf,
                                        max_depth=Max_depth,max_features=Max_features,random_state=10,alpha=Alpha)
    return feg_gbr

'''
METHODS
'''
def select_reg_model(request):
    select_child_model = request.POST.get('select_child_model').replace('task_', '')
    if select_child_model == 'linearregression':
        select_model_name = 'LinearRegression'
        fit_intercept, positive = request.POST.get('linearregression_fit_intercept'), \
                                  request.POST.get('linearregression_positive')
        fit_intercept = True if fit_intercept == 'True' else False
        positive = True if positive == 'True' else False
        select_model = regression_linear(Fit_intercept=fit_intercept,Positive=positive)

    elif select_child_model == 'regsvm':
        select_model_name = 'SVM'
        kernel, degree, gamma, coef0, C = request.POST.get('regsvm_kernel'), \
                                          request.POST.get('regsvm_degree'), \
                                          request.POST.get('regsvm_gamma'), \
                                          request.POST.get('regsvm_coef0'), \
                                          float(request.POST.get('regsvm_c'))
        if kernel == 'linear':
            select_model = regression_SVM(Kernel=kernel, CC=C)
        elif kernel == 'poly':
            degree = int(degree)
            coef0 = float(coef0)
            C = float(C)
            select_model = regression_SVM(Kernel=kernel ,Degree=degree, Coef0=coef0, CC=C, Gamma=gamma)
        elif kernel == 'rbf':
            select_model = regression_SVM(Kernel=kernel ,CC=C, Gamma=gamma)
        else:
            coef0 = float(coef0)
            select_model = regression_SVM(Kernel=kernel ,Coef0=coef0, CC=C, Gamma=gamma)

    elif select_child_model == 'ridge':
        select_model_name = 'Ridge'
        alphas, fit_intercept, positive, gcv_mode = request.POST.get('ridge_alphas'), \
                                                    request.POST.get('ridge_fit_intercept'), \
                                                    request.POST.get('ridge_positive'), \
                                                    request.POST.get('ridge_gcv_mode')
        if '[' in alphas and ']' in alphas:
            alphas = re.sub(r'[\[|\]| ]', '', alphas).split(',')
            alphas = [float(i) for i in alphas]
        else:
            alphas = float(alphas)
        fit_intercept = True if fit_intercept == 'True' else False
        positive = True if positive == 'True' else False
        select_model = regression_ridge(Alphas=alphas, Fit_intercept=fit_intercept, Positive=positive, Gcv_mode=gcv_mode)

    elif select_child_model == 'lasso':
        select_model_name = 'Lasso'
        mode, eps, n_alphas, alphas, fit_intercept, selection, positive = \
                                                request.POST.get('lasso_mode'), \
                                                request.POST.get('lasso_eps'), \
                                                request.POST.get('lasso_n_alphas'), \
                                                request.POST.get('lasso_alphas'), \
                                                request.POST.get('lasso_fit_intercept'), \
                                                request.POST.get('lasso_selection'), \
                                                request.POST.get('lasso_positive')
        fit_intercept = True if fit_intercept == 'True' else False
        positive = True if positive == 'True' else False
        if mode == 'eps+n_alphas':
            eps, n_alphas, alphas = float(eps), int(n_alphas), None
        else:
            if '[' in alphas and ']' in alphas:
                alphas = re.sub(r'[\[|\]| ]', '', alphas).split(',')
                alphas = [float(i) for i in alphas]
            else:
                alphas = float(alphas)
            eps, n_alphas = None, None
        select_model = regression_lasso(Eps=eps, N_alphas=n_alphas, Alphas=alphas, Fit_intercept=fit_intercept,
                                        Selection=selection, Positive=positive)
    elif select_child_model == 'regdecisiontree':
        select_model_name = 'DecisionTree'
        criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features = \
                                            request.POST.get('regdecisiontree_criterion'), \
                                            request.POST.get('regdecisiontree_splitter'), \
                                            request.POST.get('regdecisiontree_max_depth'), \
                                            request.POST.get('regdecisiontree_min_samples_split'), \
                                            request.POST.get('regdecisiontree_min_samples_leaf'), \
                                            request.POST.get('regdecisiontree_max_features')
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = regression_dtree(Criterion=criterion, Splitter=splitter, Max_depth=max_depth, Min_samples_split=min_samples_split,
                         Min_samples_leaf=min_samples_leaf, Max_features=max_features)

    elif select_child_model == 'regxgboost':
        select_model_name = 'XGBoost'
        booster, learning_rate, max_depth, n_estimators, gamma, min_child_weight, colsample_bytree, \
        reg_alpha, reg_lambda = request.POST.get('regxgboost_booster'), \
                                  float(request.POST.get('regxgboost_learning_rate')), \
                                  int(request.POST.get('regxgboost_max_depth')), \
                                  int(request.POST.get('regxgboost_n_estimators')), \
                                  int(request.POST.get('regxgboost_gamma')), \
                                  int(request.POST.get('regxgboost_min_child_weight')), \
                                  int(request.POST.get('regxgboost_colsample_bytree')), \
                                  int(request.POST.get('regxgboost_reg_alpha')), \
                                  int(request.POST.get('regxgboost_reg_lambda'))
        select_model = regression_xgboost(Max_depth=max_depth, Learning_rate=learning_rate, N_estimators=n_estimators,
                                          Booster=booster, Gamma=gamma, Min_child_weight=min_child_weight,
                                          Colsample_bytree=colsample_bytree, Reg_alpha=reg_alpha, Reg_lambda=reg_lambda)
    elif select_child_model == 'regrandomforest':
        select_model_name = 'RandomForest'
        criterion, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = \
                                       request.POST.get('regrandomforest_criterion'), \
                                       int(request.POST.get('regrandomforest_n_estimators')), \
                                       request.POST.get('regrandomforest_max_depth'), \
                                       request.POST.get('regrandomforest_min_samples_split'), \
                                       request.POST.get('regrandomforest_min_samples_leaf'), \
                                       request.POST.get('regrandomforest_max_features')
        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = regression_randomforest(N_estimators=n_estimators, Criterion=criterion, Max_depth=max_depth,
                                               Min_samples_split=min_samples_split,
                                               Min_samples_leaf=min_samples_split, Max_features=max_features)
    elif select_child_model == 'regadaboost':
        select_model_name = 'Adaboost'
        n_estimators, learning_rate, loss = int(request.POST.get('regadaboost_n_estimators')), \
                                            float(request.POST.get('regadaboost_learning_rate')), \
                                            request.POST.get('regadaboost_loss')
        select_model = regression_adaboost(N_estimators=n_estimators, Learning_rate=learning_rate, Loss=loss)
    else:
        select_model_name = 'GradientBoost'
        loss, learning_rate, n_estimators, subsample, min_samples_split, min_samples_leaf, max_depth, \
        max_features = request.POST.get('gradientboost_loss'), \
                         float(request.POST.get('gradientboost_learning_rate')), \
                         int(request.POST.get('gradientboost_n_estimators')), \
                         float(request.POST.get('gradientboost_subsample')), \
                         request.POST.get('gradientboost_min_samples_split'), \
                         request.POST.get('gradientboost_min_samples_leaf'), \
                         request.POST.get('gradientboost_max_depth'), \
                         request.POST.get('gradientboost_max_features')

        max_depth, min_samples_split, min_samples_leaf, max_features = \
            surv_para_group(max_depth, min_samples_split, min_samples_leaf, max_features)
        select_model = regression_GBR(Loss=loss,Learning_rate=learning_rate,N_estimators=n_estimators,
                   Subsample=subsample,Min_samples_split=min_samples_split,Min_samples_leaf=min_samples_leaf,Max_depth=max_depth,Max_features=max_features)
    return select_model, select_model_name
'''
ML FUNCTIONS
'''

def regression_preprocess(data):
    data = data.apply(pd.to_numeric, errors='ignore')
    x = data.iloc[:, data.columns != data.columns[0]]
    x = x.fillna(x.mean())  # 填充缺失值
    x_dum = pd.get_dummies(x)  # 独热编码
    y = np.array(data.iloc[:, 0]).ravel()
    return x_dum, y

def selectkbest_top20(data, label, k=20, score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k='all').fit(data, label)
    df_scores = pd.DataFrame(selector.scores_)
    df_columns = pd.DataFrame(data.columns)
    df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
    df_feature_scores.columns = ['Feature', 'Score']
    feature_names = list(df_feature_scores.sort_values(by='Score', ascending=False)['Feature'])
    if len(data.columns) > 50:
        feature_names = feature_names[:k]
    return feature_names


def RegressionKFold(data, label, n=10, k=5):
    train_index = []
    test_index = []
    kf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=10)
    for train, test in kf.split(data, label):
        train_index.append(train)
        test_index.append(test)
    return train_index, test_index


def pre_screening(data2, label, model, features):
    # 第一步筛选
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=10)
    feature_names = features
    data2 = data2[feature_names].to_numpy()
    # ifs方法得到前三分类器选择的特征数
    clf = copy.deepcopy(model)
    features_num = min([len(features), 20])
    cv_scores = [cross_val_score(clf, data2[:, :i], label, cv=cv, n_jobs=4).mean() for i in range(1, features_num + 1)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:, 0].sort_values(ascending=False).index[:3] + 1)
    return clf_num, cv_scores

# def pre_screening(data2,label,model,features,cv):
#     #第一步筛选
#     feature_names = features
#     data2 = data2[feature_names].to_numpy()
#     #ifs方法得到前三分类器选择的特征数
#     clf = copy.deepcopy(model)
#     cv_scores = [cross_val_score(clf,data2[:,:i],label,cv=cv,).mean() for i in range(1,21)]
#     clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
#     return clf_num,cv_scores


# top3训练
def train_estimator(clf, xtrain, ytrain, xtest, ytest):
    clf2 = copy.deepcopy(clf)
    # print(xtrain.dtypes)
    # print(xtrain, ytrain)
    res = clf2.fit(xtrain, ytrain)
    predict = res.predict(xtest)
    test_acc = res.score(xtest, ytest)
    # print('test_acc: ',test_acc)
    return res, test_acc, predict


def train_top3(clf, data, label, clf_num, train_index, test_index, feature_names):
    test_accs, estimators, predicts, f_names = {}, {}, {}, {}
    mean_accs = []
    for j in range(len(clf_num)):  # top3分类器
        preds, tests, res, f_name = [], [], [], []
        for i in range(len(train_index)):
            xtrain, ytrain = data.iloc[train_index[i], :], label[train_index[i]]
            xtest, ytest = data.iloc[test_index[i], :], label[test_index[i]]
            xtrain, xtest = xtrain.loc[:, feature_names[:clf_num[j]]], xtest.loc[:, feature_names[:clf_num[j]]]
            estimator, test_acc, predict = train_estimator(clf, xtrain, ytrain, xtest, ytest)
            tests.append(test_acc), res.append(estimator), preds.append(predict)
        mean_accs.append(np.mean(tests))
        test_accs[clf_num[j]] = tests
        estimators[clf_num[j]] = res
        predicts[clf_num[j]] = preds
        f_names[clf_num[j]] = feature_names[:clf_num[j]]
    # 选择得分最高的topk
    topk = clf_num[mean_accs.index(max(mean_accs))]
    test_accs = test_accs[topk]
    estimators = estimators[topk]
    predicts = predicts[topk]
    f_names = f_names[topk]
    return test_accs, estimators, mean_accs, predicts, f_names

def FSS_fun(feature_names,clf,data,label,cv,n_jobs=4):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    features_num = min([len(feature_names),20])#判断特征数目是否大于20
    for i in range(features_num):
        cv_scores = []
        for feature in feature_names2:
            train_feature = [feature] + selected_feature
            data1 = pd.DataFrame(data.loc[:,train_feature])
            cv_score = cross_val_score(clf,data1,label,cv=cv,n_jobs=n_jobs,error_score='raise').mean()
            cv_scores.append(cv_score)
        max_index = np.array(cv_scores).argmax()
        max_score = max(cv_scores)
        max_scores.append(max_score)
        selected_feature.append(feature_names2[max_index])
        feature_names2.remove(feature_names2[max_index])
    return selected_feature,max_scores


def BSS_fun(feature_names, clf, data, label, cv, n_jobs=4):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    max_scores.append(cross_val_score(clf, data, label, cv=cv, n_jobs=n_jobs).mean())
    features_num = min([len(feature_names), 50])
    for i in range(features_num - 1):
        cv_scores = []
        for feature in feature_names2:
            train_feature = feature_names2[:]  # 切片，独立于原列表
            train_feature.remove(feature)
            data1 = pd.DataFrame(data.loc[:, train_feature])
            cv_score = cross_val_score(clf, data1, label, cv=cv, n_jobs=n_jobs).mean()
            cv_scores.append(cv_score)
        max_index = np.array(cv_scores).argmax()
        max_score = max(cv_scores)
        max_scores.append(max_score)
        selected_feature.append(feature_names2[max_index])
        del feature_names2[max_index]
    selected_feature.append(feature_names2[0])
    selected_feature.reverse()  # 反向排序
    max_scores.reverse()
    return selected_feature, max_scores

def cust_cv_reports(preds,test_index,label,tests):
    maes,mses = [],[]
    for i in range(len(preds)):
        ytest = label[test_index[i]]
        maes.append(mean_absolute_error(ytest, preds[i]))
        mses.append(mean_squared_error(ytest, preds[i]))
    cust_reports = {'R-square':tests,'MAE':maes,'MSE':mses,}
    cust_reports = pd.DataFrame(cust_reports)
    cust_reports_describe = np.round(cust_reports.describe().loc[("mean",'min','max','std'),:], 3)
    return cust_reports, cust_reports_describe

def reg_cust_val(est, vdata, vlabel, features, reg_model_name):
    validate_predict = est[0].predict(vdata[features])
    validate_r2 = est[0].score(vdata[features],vlabel)
    validate_mae = mean_absolute_error(vlabel,validate_predict)
    validate_mse = mean_squared_error(vlabel,validate_predict)
    val_report = pd.DataFrame({'R-square':validate_r2,'MAE':validate_mae,'MSE':validate_mse},index=[reg_model_name])
    return val_report

def df2bp(df):
    data = []
    i=0
    for col in df.columns:
        trace = {
            'type': 'box',
            'name': col.replace('test_accuracy', 'accuracy'),
            'y': df[col].to_list(),
            'xaxis': 'x' + str(i + 1),
            'yaxis': 'y' + str(i + 1)
        }
        i += 1
        data.append(trace)
    return data