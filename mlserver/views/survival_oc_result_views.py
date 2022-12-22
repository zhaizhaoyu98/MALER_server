from django.shortcuts import render

import os, shutil, copy, pickle, json, time
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
from mlserver.views.classification_oc_result_views import get_file_md5, df2bp, mklinechart, split_train_test, classification_process
# from mlserver.views.regression_cp_result_views import pre_screening
import warnings
warnings.filterwarnings("ignore")


def survival_oc_result(request, projectid):
    sur_models = [FastKernelSurvivalSVM(kernel='linear', random_state=10, max_iter=100),
                  SurvivalTree(random_state=10),
                  ExtraSurvivalTrees(random_state=10, n_jobs=4),
                  RandomSurvivalForest(min_samples_split=10, min_samples_leaf=15, max_features='sqrt', random_state=10,
                                       n_jobs=4),
                  GradientBoostingSurvivalAnalysis(random_state=10)]
    sur_names = ['SurvivalSVM', 'SurvivalTree', 'ExtraSurvivalTrees', 'RandomSurvivalForest',
                 'GradientBoostingSurvival']

    feature_select_method = projectid.split('-')[2]
    select_model = 'model_reg'
    # select_model = request.POST.get('select_model')
    # file_upload_type = request.POST.get('file_upload_type')
    # Feature selection methods
    # feature_select_method = request.POST.get('feature_select_method')
    # print('feature_select_method: ', feature_select_method)

    # projectid = request.POST.get('projectid')

    # if file_upload_type == 'example_data':
    #     upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/survival_example.csv'))
    #     projectid = 'SO-' + upload_file_md5[:6] + '-' + feature_select_method


    if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid, 'surv_pickle.pkl')):
        inputdata = pd.read_csv(
            STATIC_ROOT + '/cache/' + projectid + '/' + 'data.csv',
            header=0, index_col=0).T
        # if file_upload_type == 'user_data':
        #     '''
        #         IMPORRT DATA
        #         '''
        #     # file load
        #     upload_file = request.FILES.get('upload_file')
        #     f = open(os.path.join(STATIC_ROOT, 'cache', upload_file.name), 'wb')
        #     for line in upload_file.chunks():
        #         f.write(line)
        #     f.close()
        #
        #     upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', upload_file.name))
        #     projectid = 'SO-' + upload_file_md5[:6] + '-' + feature_select_method
        #     newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        #     os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        #     shutil.move(STATIC_ROOT + '/cache/' + upload_file.name, newpath)
        #
        #     inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + upload_file.name, header=0,
        #                             index_col=0).T
        # else:
        #     newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        #     os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        #     shutil.copy(STATIC_ROOT + '/cache/example/survival_example.csv', newpath)
        #     inputdata = pd.read_csv(STATIC_ROOT + '/cache/example/survival_example.csv', header=0, index_col=0).T
        #
        #

        '''
        projectid='SO-c319b6-TopK'
        feature_select_method = 'TopK'
        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + 'tpm_gbm_surdata.csv', header=0, index_col=0).T
        '''

        train_set, test_set, blind_set = split_train_test(inputdata, datatype='survival')
        x2, y2 = sur_data_process(train_set)
        # x2, validation_data, y2, validation_label = train_test_split(x, y, random_state=10, train_size=0.7,stratify=y['Status'])
        if len(test_set) > 0:
            validation_data, validation_label = sur_data_process(test_set)

        cv = KFold(n_splits=5, shuffle=True, random_state=10)
        features = cox_selection(x2, y2)
        x3 = x2[features]
        train_index, test_index = sur_RSKFold(x3, y2)
        if feature_select_method != 'TopK':
            selected_feature, max_scores = [], []
            for each_model in sur_models:
                start = time.perf_counter()
                if feature_select_method == 'FSS':
                    sf, ms = FSS_fun(features, each_model, x3, y2, cv, n_jobs=4)
                else:
                    sf, ms = BSS_fun(features, each_model, x3, y2, cv, n_jobs=4)
                selected_feature.append(sf), max_scores.append(ms)
                end = time.perf_counter()
                print(round(end - start, 3))



            max_indexs, max_score, max_features = [], [], []
            for i in range(len(max_scores)):
                max_index = np.array(max_scores[i]).argmax()
                max_indexs.append(max_index)
                max_score.append(max(max_scores[i]))
                max_features.append(selected_feature[i][:max_index + 1])

            test_accs, estimators, predicts = {}, {}, {}
            for j in range(len(sur_names)):
                preds, tests, res = [], [], []
                start = time.perf_counter()
                for i in range(len(train_index)):
                    xtrain, ytrain = x3.iloc[train_index[i], :], y2[train_index[i]]
                    xtest, ytest = x3.iloc[test_index[i], :], y2[test_index[i]]
                    xtrain, xtest = xtrain[max_features[j]], xtest[max_features[j]]
                    estimator, test_acc, predict = train_estimator(sur_models[j], xtrain, ytrain, xtest, ytest)
                    tests.append(test_acc), res.append(estimator), preds.append(predict)
                test_accs[j] = tests
                estimators[j] = res
                predicts[j] = preds
                end = time.perf_counter()
                print(sur_names[j], ':', round(end - start, 2))
        else:
            clf_nums, max_scores, max_indexs = [], [], []
            test_accs, estimators, mean_accs, predicts, f_names = {}, {}, {}, {}, {}
            for i in range(len(sur_names)):
                start = time.perf_counter()
                clf_num, ms = pre_screening(x3, y2, sur_models[i], features)
                clf_nums.append(clf_num)
                max_scores.append(ms)
                max_indexs.append(np.array(ms).argmax())
                test_accs[i], estimators[i], mean_accs[i], predicts[i], f_names[i] = train_top3(sur_models[i], x3, y2,
                                                                                                clf_num, train_index,
                                                                                                test_index, features)
                end = time.perf_counter()
                print(round(end - start, 2))
            max_features = f_names

        line_chart_data = mklinechart(max_scores, sur_names)

        test_acc_reports = pd.DataFrame(data=test_accs)
        test_acc_reports.columns = sur_names
        test_acc_reports_dict = df2bp(test_acc_reports)
        test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                     3)
        test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        test_acc_describe_dict = test_acc_describe_.to_dict('records')

        tmodels = []
        for i in range(len(sur_models)):
            model2 = copy.deepcopy(sur_models[i])
            res = model2.fit(x3[max_features[i]], y2)
            tmodels.append(res)

        parameter, test_acc, best_esti = [], [], []
        feature_names = []
        for i in range(len(sur_names)):
            best_esti.append(tmodels[i])
            parameter.append(str(estimators[i][0].get_params()))
            test_acc.append(test_acc_describe.loc['mean',][i])
            feature_names.append(max_features[i])

        max_reports = {'Mean C-index': test_acc,
                       'parameter': parameter,
                       'feature_names': [str(list(f)) for f in feature_names], }
        max_reports = pd.DataFrame(max_reports, index=sur_names).reset_index().rename(
            columns={'index': 'Method'})
        max_reports_dict = max_reports.to_dict('records')

        # survival
        data_medians_dict, surv_dict, para_dict = [], {}, {}
        for i in range(len(sur_names)):
            data_median = tmodels[i].predict(pd.DataFrame(x3[max_features[i]].median()).T)[0]
            data_medians_dict.append(data_median)
            surv_trace, resultp = mk_surv_data(str(i+1),x3[max_features[i]],y2,best_esti[i],data_median)
            surv_layout = mk_surv_layout(sur_names[i], resultp)
            t_dict = {'surv_trace': surv_trace, 'surv_layout': surv_layout}
            surv_dict[sur_names[i]] = t_dict

            para_dict[sur_names[i]] = np.round(resultp.p_value, 6)

        subplot_sur = []
        for i in surv_dict.keys():
            subplot_sur.append(surv_dict[i]['surv_trace'][0])
            subplot_sur.append(surv_dict[i]['surv_trace'][1])

        # validation
        vsurv_dict, vpara_dict = {}, {}
        for i in range(len(sur_names)):
            vsurv_trace, vresultp = mk_surv_data(str(i+1), validation_data[max_features[i]], validation_label, best_esti[i],
                                        data_medians_dict[i])
            vsurv_layout = mk_surv_layout(sur_names[i], vresultp)
            t_dict = {'surv_trace': vsurv_trace, 'surv_layout': vsurv_layout}
            vsurv_dict[sur_names[i]] = t_dict
            vpara_dict[sur_names[i]] = np.round(vresultp.p_value, 6)

        vsubplot_sur = []
        for i in vsurv_dict.keys():
            vsubplot_sur.append(vsurv_dict[i]['surv_trace'][0])
            vsubplot_sur.append(vsurv_dict[i]['surv_trace'][1])

        vlinedata = []
        for i in range(len(sur_names)):
            va_times, rsf_auc, mean_auc, cindex = time_dependent_auc(tmodels[i],validation_data,validation_label,y2[train_index[max_indexs[i]]],
                                                                        max_features[i],sur_names[i])

            vlinetrace = mk_auc_line(sur_names[i], va_times, rsf_auc, mean_auc, cindex)
            vlinedata.append(vlinetrace)
        # pickle
        surv_pickle = {
            'line_chart_data': line_chart_data,
            'test_acc_reports_dict': test_acc_reports_dict,
            'test_acc_describe_dict': test_acc_describe_dict,
            'max_reports_dict': max_reports_dict,
            'surv_dict': surv_dict,
            'subplot_sur': subplot_sur,
            'para_dict': para_dict,
            'vsubplot_sur': vsubplot_sur,
            'vpara_dict': vpara_dict,
            'vlinedata': vlinedata
        }

        with open(STATIC_ROOT + '/cache/' + projectid + '/surv_pickle.pkl',
                  'wb') as f:
            pickle.dump(surv_pickle, f)

        for i in range(max_reports.shape[0]):
            t = sur_names[i].replace(' ', '_')
            model = best_esti[i]
            model_pickle = {
                'method': select_model,
                'name': t,
                'model': model,
                'feature_names': feature_names[i]
            }
            with open(STATIC_ROOT + '/cache/' + projectid + '/' + t + '.pkl', 'wb') as f:
                pickle.dump(model_pickle, f)
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
machine learning codes
'''
def sur_data_process(data):
    data = data.rename(columns={data.columns[0]:'Status', data.columns[1]:'time'})
    data2 = data[data['time'].astype(float) >0]
    data2 = data2.apply(pd.to_numeric,errors='ignore')

    if type(data2['Status'].unique()[0]) == str :
        data2['Status'] = data2['Status'].str.lower()
        x,y = get_x_y(data2,data2.columns[:2],pos_label='dead')
    else:
        x,y = get_x_y(data2,data2.columns[:2],pos_label=1)
    #转变数据类型，并进行独热编码
    x = x.apply(pd.to_numeric,errors='ignore')
    x = pd.get_dummies(x)
    return x,y

# def cox_selection(X, y, fnames):
#     n_features = X.shape[1]
#     scores = np.empty(n_features)
#     m = CoxPHSurvivalAnalysis()
#     for j in range(n_features):
#         Xj = X[:, j:j+1]
#         m.fit(Xj, y)
#         scores[j] = m.score(Xj, y)
#     res_c = pd.Series(scores, index=fnames).sort_values(ascending=False)
#     if len(res_c )>50:
#         res_c = res_c[:50]
#     res_c2 = res_c[res_c>0.5]
#     return res_c2.index

def cox_selection(x,y):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    ss =[]
    df = pd.concat([pd.DataFrame(y,index=x.index),x],axis=1)
    for i in range(2,len(df.columns)):
        cph.fit(df.iloc[:,[0,1,i]], 'time',event_col='Status')
        ss.append(cph.summary['p'])
    ss2 = pd.concat(ss)
    features = ss2[ss2 <0.05].sort_values()[:50].index
    return features

def sur_RSKFold (data,label,n=10,k=5):
    train_index = []
    test_index = []
    kf = RepeatedKFold(n_splits=k,n_repeats=n,random_state=10)
    for train, test in kf.split(data,label):
        train_index.append(train)
        test_index.append(test)
    return train_index, test_index

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
def BSS_fun(feature_names,clf,data,label,cv,n_jobs=4):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    max_scores.append(cross_val_score(clf,data,label,cv=cv,n_jobs=n_jobs).mean())#计算全部特征下的训练结果
    features_num = min([len(feature_names),50])#判断特征数目是否大于50
    for i in range(features_num-1):
        cv_scores = []
        for feature in feature_names2:
            train_feature = feature_names2[:] #切片，独立于原列表
            train_feature.remove(feature)
            data1 = pd.DataFrame(data.loc[:,train_feature])
            cv_score = cross_val_score(clf,data1,label,cv=cv,n_jobs=n_jobs).mean()
            cv_scores.append(cv_score)
        max_index = np.array(cv_scores).argmax()
        max_score = max(cv_scores)
        max_scores.append(max_score)
        selected_feature.append(feature_names2[max_index])
        del feature_names2[max_index]
    selected_feature.append(feature_names2[0])
    selected_feature.reverse() #反向排序
    max_scores.reverse()
    return selected_feature,max_scores

def train_estimator(clf,xtrain,ytrain,xtest,ytest):
    clf2 = copy.deepcopy(clf)
    res = clf2.fit(xtrain,ytrain)
    predict = res.predict(xtest)
    test_acc = res.score(xtest,ytest)
    return res,test_acc,predict

#初筛
def pre_screening(data2,label,model,features):
    #第一步筛选
    cv = KFold(n_splits=10, shuffle=True, random_state=10)
    feature_names = features
    data2 = data2[feature_names].to_numpy()
    #ifs方法得到前三分类器选择的特征数
    clf = copy.deepcopy(model)
    # cv_scores = [cross_val_score(clf,data2[:,:i],label,cv=cv,n_jobs=4).mean() for i in range(1,21)]
    features_num = min([len(features), 20])
    cv_scores = [cross_val_score(clf, data2[:, :i], label, cv=cv, n_jobs=4).mean() for i in range(1, features_num + 1)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
    return clf_num, cv_scores

def train_top3(clf,data,label,clf_num,train_index,test_index,feature_names):
    test_accs,estimators,predicts,f_names = {},{},{},{}
    mean_accs = []
    for j in range(len(clf_num)):    #top3分类器
        preds,tests,res,f_name = [],[],[],[]
        for i in range(len(train_index)):
            xtrain,ytrain = data.iloc[train_index[i],:],label[train_index[i]]
            xtest,ytest = data.iloc[test_index[i],:],label[test_index[i]]
            xtrain,xtest = xtrain.loc[:,feature_names[:clf_num[j]]],xtest.loc[:,feature_names[:clf_num[j]]]
            estimator,test_acc,predict = train_estimator(clf,xtrain,ytrain,xtest,ytest)
            tests.append(test_acc),res.append(estimator),preds.append(predict)
        mean_accs.append(np.mean(tests))
        test_accs[clf_num[j]] = tests
        estimators[clf_num[j]] = res
        predicts[clf_num[j]] = preds
        f_names[clf_num[j]] = feature_names[:clf_num[j]]
    #选择得分最高的topk
    topk = clf_num[mean_accs.index(max(mean_accs))]
    test_accs = test_accs[topk]
    estimators = estimators[topk]
    predicts = predicts[topk]
    f_names = f_names[topk]
    return test_accs,estimators,mean_accs,predicts,f_names
'''
plot data
'''


def mk_surv_data(sur_name_i, datax,label,estimator,data_median):
    if sur_name_i == '1' or len(sur_name_i) > 3:
        showlegend = True
    else:
        showlegend = False
    data = datax.copy()
    data['predict'] = estimator.predict(data)
    high = data['predict'] > data_median
    low = data['predict'] <= data_median

    h_time_treatment, h_survival_prob_treatment = kaplan_meier_estimator(
            label['Status'][high],
            label['time'][high])
    l_time_treatment, l_survival_prob_treatment = kaplan_meier_estimator(
            label['Status'][low],
            label['time'][low])
    trace_h = {
        # 'key': null,
        'line': {
            'dash': 'solid',
            'color': 'red',
            'shape': 'hv',
            'width': 2
        },
        'mode': 'lines',
        'name': 'High Risk',
        'type': 'scatter',
        'x': list(h_time_treatment),
        'y': list(h_survival_prob_treatment),
        'xaxis': 'x' + sur_name_i,
        'yaxis': 'y' + sur_name_i,
        'text': make_surv_text(h_time_treatment, h_survival_prob_treatment),
        'hoverinfo': 'text',
        'showlegend': showlegend,
        # 'legendgroup': 'High Risk'
    }
    trace_l = {
        # 'key': null,
        'line': {
            'dash': 'solid',
            'color': 'blue',
            'shape': 'hv',
            'width': 2
        },
        'mode': 'lines',
        'name': 'Low Risk',
        'type': 'scatter',
        'x': list(l_time_treatment),
        'y': list(l_survival_prob_treatment),
        'xaxis': 'x' + sur_name_i,
        'yaxis': 'y' + sur_name_i,
        'text': make_surv_text(l_time_treatment, l_survival_prob_treatment),
        'hoverinfo': 'text',
        'showlegend': showlegend,
        # 'legendgroup': 'Low Risk'
    }
    trace_data = [trace_h, trace_l]
    resultp = logrank_test(label['time'][high],label['time'][low],
                           label['Status'][high],label['Status'][low])
    return trace_data, resultp

def make_surv_text(x, y):
    string = 'time: %s<br>surv: %s'
    text = []
    for i in range(len(x)):
        str = string %(x[i], np.round(y[i], 3))
        text.append(str)
    return text

def mk_surv_layout(name, resultp):
    layout = {
        'title': r'Survival curves of {}<br>logrank test p-value = {}'.format(name,np.round(resultp.p_value, 6)),
        'xaxis': {
            'range': [0, ],
            'title': 'time <i>t</i>',
            'tickfont': {
                'size': 12.75,
            },
            'titlefont': {
                'size': 15,
            },
        },
        'yaxis': {
            'title': 'probability of survival <i>&#348;(t)</i>',
            'tickfont': {
                'size': 12.75,
            },
            'titlefont': {
                'size': 15,
            },
        },
        'margin': {
            'b': 45,
            'l': 75,
            'r': 7,
            't': 75
        },
        'shapes': [
            {
                'x0': 0,
                'x1': 1,
                'y0': 0,
                'y1': 1,
                'line': {
                    # 'color': 'rgba(127,127,127,1)',
                    'width': 1,
                    'linetype': 'solid'
                },
                'xref': 'paper',
                'yref': 'paper',
                'fillcolor': 'transparent'
        }
        ],
        'titlefont': {
            'size': 20,
        },
    }
    return layout

# def time_dependent_auc(estimator,data,label,ytrain,max_feature,name):
#     cindex = estimator.score(data[max_feature], label)
#     va_times = np.arange(max([min(ytrain['time']), min(label['time'])]),
#                          min([max(ytrain['time']), max(label['time']), 3650]), 30)  ##时间要修改加判断
#     if name == 'SurvivalSVM' or name == "Lasso":
#         cph_risk_scores = estimator.predict(data[max_feature])
#         rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
#             ytrain, label, cph_risk_scores, va_times
#         )
#     else:
#         rsf_chf_funcs = estimator.predict_cumulative_hazard_function(
#             data[max_feature])
#         va_times = np.arange(max([min(ytrain['time']), min(label['time']), min(rsf_chf_funcs[0].x)]),
#                              min([max(rsf_chf_funcs[0].x), 3650]), 30)
#         rsf_risk_scores = np.row_stack([chf(va_times) for chf in rsf_chf_funcs])
#         rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
#             ytrain, label, rsf_risk_scores, va_times
#         )
#     mean_auc = np.nan_to_num(rsf_auc).mean()
#     cindex = np.round(cindex,3)
#     return va_times, rsf_auc, mean_auc, cindex

def time_dependent_auc(estimator,data,label,ytrain,max_features,name):
    cindex = estimator.score(data[max_features],label)
    va_times = np.arange(max([min(ytrain['time']),min(label['time'])]),
                         min([max(ytrain['time']),max(label['time']),3650]),30)  ##时间要修改加判断
    if name =='SurvivalSVM' or name == "Lasso":
        cph_risk_scores = estimator.predict(data[max_features])
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
            ytrain, label, cph_risk_scores, va_times
        )
    else:
        rsf_chf_funcs = estimator.predict_cumulative_hazard_function(
            data[max_features])
        va_times = np.arange(max([min(ytrain['time']),min(label['time']),min(rsf_chf_funcs[0].x)]),
                             min([max(rsf_chf_funcs[0].x),max(label['time']),3650]),30)
        rsf_risk_scores = np.row_stack([chf(va_times) for chf in rsf_chf_funcs])
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
            ytrain,label, rsf_risk_scores,va_times
        )
    mean_auc = np.nan_to_num(rsf_auc).mean()
    cindex = np.round(cindex, 3)
    return va_times, rsf_auc, mean_auc, cindex

def mk_auc_line(name, va_times, rsf_auc, mean_auc, cindex):
    trace = {
        'x': list(va_times),
        'y': list(rsf_auc),
        'mode': 'lines',
        'name': "{} (mean AUC = {:.3f},C-index = {})".format(name, mean_auc, cindex)
    }
    return trace