from django.shortcuts import render

import os, shutil, copy, pickle, json, time
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif,chi2,VarianceThreshold,mutual_info_classif,f_regression
from sklearn.model_selection import cross_val_score,cross_validate , GridSearchCV, KFold,\
    StratifiedKFold,RepeatedKFold
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

from ML_WebServer.settings import STATIC_ROOT
from mlserver.views.classification_oc_result_views import get_file_md5, df2bp, JsonEncoder, split_train_test, mkradar

models_str = ['LinearRegression', 'SVM', 'Ridge', 'Lasso', 'DecisionTree', 'XGBoost',
                  'RandomForest', 'AdaBoost', 'GradientBoost', ]
def regression_oc_result(request):
    Alphas = [0.01, 0.05, 0.1, 1.0, 2.0, 5.0, 10.0]
    models = [LinearRegression(n_jobs=4), SVR(kernel='linear', max_iter=5000), RidgeCV(alphas=Alphas),
              LassoCV(n_jobs=4, alphas=Alphas),
              DecisionTreeRegressor(random_state=10), XGBRegressor(n_jobs=4),
              RandomForestRegressor(n_jobs=4, random_state=10),
              AdaBoostRegressor(random_state=10), GradientBoostingRegressor(random_state=10), ]
    models_str = ['LinearRegression', 'SVM', 'Ridge', 'Lasso', 'DecisionTree', 'XGBoost',
                  'RandomForest', 'AdaBoost', 'GradientBoost', ]

    select_model = request.POST.get('select_model')
    feature_select_method = request.POST.get('feature_select_method')
    file_upload_type = request.POST.get('file_upload_type')

    print('feature_select_method: ', feature_select_method)
    if file_upload_type == 'user_data':
        '''
        IMPORRT DATA
        '''
        # file load
        upload_file = request.FILES.get('upload_file')
        f = open(os.path.join(STATIC_ROOT, 'cache', upload_file.name), 'wb')
        for line in upload_file.chunks():
            f.write(line)
        f.close()
        upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache', upload_file.name))
    else:
        upload_file_md5 = get_file_md5(os.path.join(STATIC_ROOT, 'cache/example/regression_example.csv'))

    projectid = 'RO-' + upload_file_md5[:6] + '-' + feature_select_method
    if not os.path.exists(os.path.join(STATIC_ROOT, 'cache', projectid)):
        newpath = os.path.join(STATIC_ROOT, 'cache', projectid)
        os.mkdir(os.path.join(STATIC_ROOT, 'cache', projectid))
        shutil.move(STATIC_ROOT + '/cache/' + upload_file.name, newpath)
        '''
        feature_select_method='TopK'
        projectid='RO-19f4d5-TopK'
        data = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + 'regression_data.csv', header=0, index_col=0).T
        '''
        inputdata = pd.read_csv(STATIC_ROOT + '/cache/' + projectid + '/' + upload_file.name, header=0, index_col=0).T
        train_set, test_set, blind_set = split_train_test(inputdata)
        # data, label = classification_process(train_set)
        nordata4, nor_age4 = regression_preprocess(train_set)
        if len(test_set) > 0:
            validation_data, validation_label = regression_preprocess(test_set)

        # nordata4, vaildation_data, nor_age4, vaildation_label = train_test_split(x_dum, y, random_state=10,
        #                                                                          train_size=0.7)  # 分验证集
        features = selectkbest_top20(nordata4, nor_age4, score_func=f_regression, k=50)

        nordata4 = nordata4[features]
        train_index, test_index = RegressionKFold(nordata4, nor_age4)
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=10)
        ''' TopK '''
        if feature_select_method == 'TopK':
            clf_nums, cv_scores = [], []
            test_accs, estimators, mean_accs, predicts, max_features = {}, {}, {}, {}, {}
            for i in range(len(models)):
                start = time.perf_counter()
                clf_num, ms = pre_screening(nordata4, nor_age4, models[i], features, cv)
                clf_nums.append(clf_num), cv_scores.append(ms)
                # print('model num', i)
                test_accs[i], estimators[i], mean_accs[i], predicts[i], max_features[i] = train_top3(models[i],nordata4, nor_age4,clf_num,train_index,test_index,features)
                end = time.perf_counter()
                print(round(end - start, 2))

            line_chart_data = []
            for f in range(len(cv_scores)):
                if len(np.argwhere(np.isnan(cv_scores[f]))) == 1:
                    xnum = list(range(1, 21))
                    xnum.pop(np.argwhere(np.isnan(cv_scores[f]))[0][0])
                    ynum = cv_scores[f]
                    ynum.pop(np.argwhere(np.isnan(cv_scores[f]))[0][0])
                    trace = {
                        'mode': 'lines+markers',
                        'name': models_str[f],
                        'type': 'scatter',
                        'x': xnum,
                        'y': ynum
                    }
                else:
                    trace = {
                        'mode': 'lines+markers',
                        'name': models_str[f],
                        'type': 'scatter',
                        'x': list(range(1, 21)),
                        'y': cv_scores[f]
                    }
                line_chart_data.append(trace)

        # BSS or FSS
        elif feature_select_method == 'FSS' or feature_select_method == 'BSS':
            selected_feature, max_scores = [], []
            for each_model in models:
                start = time.perf_counter()
                if feature_select_method == 'FSS':
                    sf, ms = FSS_fun(features, each_model, nordata4[features], nor_age4, cv)
                else:
                    sf, ms = BSS_fun(features, each_model, nordata4[features], nor_age4, cv)
                selected_feature.append(sf), max_scores.append(ms)
                end = time.perf_counter()
                print(round(end - start, 2))

            max_indexs, max_score, max_features = [], [], []
            for i in range(len(models_str)):
                max_index = np.array(max_scores[i]).argmax()
                max_indexs.append(max_index)
                max_score.append(max(max_scores[i]))
                max_features.append(selected_feature[i][:max_index + 1])

            line_chart_data = []
            for f in range(len(max_scores)):
                if len(np.argwhere(np.isnan(max_scores[f]))) == 1:
                    xnum = list(range(1, 21))
                    xnum.pop(np.argwhere(np.isnan(max_scores[f]))[0][0])
                    ynum = max_scores[f]
                    ynum.pop(np.argwhere(np.isnan(max_scores[f]))[0][0])
                    trace = {
                        'mode': 'lines+markers',
                        'name': models_str[f],
                        'type': 'scatter',
                        'x': xnum,
                        'y': ynum
                    }
                else:
                    trace = {
                        'mode': 'lines+markers',
                        'name': models_str[f],
                        'type': 'scatter',
                        'x': list(range(1, 21)),
                        'y': max_scores[f]
                    }
                line_chart_data.append(trace)

                # nordata4,nor_age4
                test_accs, estimators, predicts = {}, {}, {}
                for j in range(len(models_str)):
                    preds, tests, res = [], [], []
                    start = time.perf_counter()
                    for i in range(len(train_index)):
                        xtrain, ytrain = nordata4.iloc[train_index[i], :], nor_age4[train_index[i]]
                        xtest, ytest = nordata4.iloc[test_index[i], :], nor_age4[test_index[i]]
                        xtrain, xtest = xtrain[max_features[j]], xtest[max_features[j]]
                        estimator, test_acc, predict = train_estimator(models[j], xtrain, ytrain, xtest, ytest)
                        tests.append(test_acc), res.append(estimator), preds.append(predict)
                    test_accs[j] = tests
                    estimators[j] = res
                    predicts[j] = preds
                    end = time.perf_counter()
                    print(round(end - start, 2))
        else:
            print('error')
            return render(request, 'ERROR.html', {
                'error_msg': 'Invalid input!'
            })

        test_acc_reports = pd.DataFrame(data=test_accs)
        test_acc_reports.columns = models_str
        test_acc_reports_dict = df2bp(test_acc_reports)
        test_acc_describe = np.round(test_acc_reports.describe().loc[("mean", 'min', 'max', 'std'), :],
                                     3)
        test_acc_describe_ = test_acc_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        test_acc_describe_dict = test_acc_describe_.to_dict('records')

        MAE_report, MSE_report, MAE_report_describe, MSE_report_describe = mae_mse_report(predicts, nor_age4,
                                                                                          test_index,
                                                                                          models_str=models_str)

        MAE_report_describe = np.round(MAE_report_describe.describe().loc[("mean", 'min', 'max', 'std'), :],
                                     3)
        MAE_report_describe_ = MAE_report_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        MAE_report_describe_dict = MAE_report_describe_.to_dict('records')

        MSE_report_describe = np.round(MSE_report_describe.describe().loc[("mean", 'min', 'max', 'std'), :],
                                       3)
        MSE_report_describe_ = MSE_report_describe.reset_index().rename(columns={'index': 'Method'})  # 测试集准确率指数
        MSE_report_describe_dict = MSE_report_describe_.to_dict('records')


        MAE_report_dict = df2bp(MAE_report)
        MSE_report_dict = df2bp(MSE_report)

        # # radar plot
        # test_acc_radar = test_acc_describe_.loc[test_acc_describe_['Method'] == 'mean']
        # test_acc_radar['Method'] = 'R-square'
        # mae_radar = MAE_report_describe_.loc[MAE_report_describe_['Method'] == 'mean']
        # mae_radar['Method'] = 'MAE'
        # mse_radar = MSE_report_describe_.loc[MSE_report_describe_['Method'] == 'mean']
        # mse_radar['Method'] = 'MSE'
        # mean_method_model = pd.concat(
        #     [test_acc_radar, mae_radar, mse_radar]).set_index('Method')
        # radar_dict = mkradar(mean_method_model)
        # radar_min = mean_method_model.min().min()
        # radar_max = mean_method_model.max().max()
        # radar_range = [radar_min, radar_max]

        tmodels = []
        for i in range(len(models)):
            model2 = copy.deepcopy(models[i])
            res = model2.fit(nordata4[max_features[i]], nor_age4)
            tmodels.append(res)

        #
        parameter, best_esti = [], [],
        R2, Mae, Mse = [], [], []
        feature_names = []
        for i in range(len(estimators)):
            best_esti.append(tmodels[i])
            par = tmodels[i].get_params()
            if i == 2 or i == 3:
                par['alphas'] = tmodels[i].alpha_
            parameter.append(str(par))
            R2.append(test_acc_describe.iloc[0, :][i])
            Mae.append(MAE_report_describe.iloc[0, :][i])
            Mse.append(MSE_report_describe.iloc[0, :][i])
            feature_names.append(max_features[i])
        final_reports = {'parameter': parameter,
                         'feature_names': [str(f) for f in feature_names],
                         'Mean R-square': R2,
                         'Mean MAE': Mae,
                         'Mean MSE': Mse}
        final_reports = pd.DataFrame(final_reports, index=models_str).reset_index().rename(
            columns={'index': 'Method'})
        final_reports_dict = final_reports.to_dict('records')

        # validation
        val_report, validate_predicts = regression_valreport(best_esti, validation_data, validation_label, max_features)
        vregpred_trace = mkvregpredplot(validate_predicts, validation_label, models_str=models_str)
        vreport_trace = mkvreportbarplot(val_report)

        val_report = np.round(val_report, 3)
        val_report = val_report.reset_index().rename(columns={'index': 'Method'})
        val_report_dict = val_report.to_dict('records')

        # pickle
        reg_pickle = {
            'line_chart_data': line_chart_data,
            'test_acc_reports_dict': test_acc_reports_dict,
            'test_acc_describe_dict': test_acc_describe_dict,
            'MAE_report_dict': MAE_report_dict,
            'MAE_report_describe_dict': MAE_report_describe_dict,
            'MSE_report_dict': MSE_report_dict,
            'MSE_report_describe_dict': MSE_report_describe_dict,
            'final_reports_dict': final_reports_dict,
            'vregpred_trace': vregpred_trace,
            'vreport_trace': vreport_trace,
            'val_report_dict': val_report_dict,
            # 'radar_dict': radar_dict,
            # 'radar_range': radar_range,
        }

        with open(STATIC_ROOT + '/cache/' + projectid + '/regression_pickle.pkl',
                  'wb') as f:
            pickle.dump(reg_pickle, f)

        for i in range(final_reports.shape[0]):
            t = models_str[i].replace(' ', '_')
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
        'vregpred_trace': json.dumps(vregpred_trace,ensure_ascii=False, cls=JsonEncoder),
        'vreport_trace': json.dumps(vreport_trace,ensure_ascii=False, cls=JsonEncoder),
        'val_report_dict': json.dumps(val_report_dict),
        # 'radar_dict': json.dumps(radar_dict),
        # 'radar_range': json.dumps(radar_range)
    })


'''
REGRESSION METHODS
'''
def regression_preprocess(data):
    data = data.apply(pd.to_numeric,errors='ignore')
    x=data.iloc[:,data.columns!=data.columns[0]]
    x=x.fillna(x.mean())  #填充缺失值
    x_dum=pd.get_dummies(x)#独热编码
    y=np.array(data.iloc[:,0]).ravel()
    return x_dum,y

def selectkbest_top20(data,label,k=20,score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k='all').fit(data,label)
    df_scores = pd.DataFrame(selector.scores_)
    df_columns = pd.DataFrame(data.columns)
    df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
    df_feature_scores.columns = ['Feature', 'Score']
    feature_names=list(df_feature_scores.sort_values(by='Score', ascending=False)['Feature'])
    if len(data.columns) > 50:
        feature_names = feature_names[:k]
    return feature_names

def RegressionKFold (data,label,n=10,k=5):
    train_index = []
    test_index = []
    kf = RepeatedKFold(n_splits=k,n_repeats=n,random_state=10)
    for train, test in kf.split(data,label):
        train_index.append(train)
        test_index.append(test)
    return train_index,test_index

def pre_screening(data2,label,model,features,cv):
    #第一步筛选
    feature_names = features
    data2 = data2[feature_names].to_numpy()
    #ifs方法得到前三分类器选择的特征数
    clf = copy.deepcopy(model)
    cv_scores = [cross_val_score(clf,data2[:,:i],label,cv=cv,).mean() for i in range(1,21)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
    return clf_num,cv_scores

def train_top3(clf,data,label,clf_num,train_index,test_index,feature_names):
    test_accs,estimators,predicts,f_names = {},{},{},{}
    mean_accs = []
    for j in range(len(clf_num)):    #top3分类器
        preds,tests,res,f_name = [],[],[],[]
        for i in range(len(train_index)):
            xtrain,ytrain = data.iloc[train_index[i],:],label[train_index[i]]
            xtest,ytest = data.iloc[test_index[i],:],label[test_index[i]]
            xtrain,xtest = xtrain.loc[:,feature_names[:clf_num[j]]],xtest.loc[:,feature_names[:clf_num[j]]]
            # print(i)
            # print(len(ytrain))
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

#top3训练
def train_estimator(clf,xtrain,ytrain,xtest,ytest):
    clf2 = copy.deepcopy(clf)
    res = clf2.fit(xtrain,ytrain)
    predict = res.predict(xtest)
    test_acc = res.score(xtest,ytest)
    return res,test_acc,predict


def FSS_fun(feature_names, clf, data, label, cv, n_jobs=4):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    features_num = min([len(data.columns), 20])  # 判断特征数目是否大于20
    for i in range(features_num):
        cv_scores = []
        for feature in feature_names2:
            train_feature = [feature] + selected_feature
            data1 = pd.DataFrame(data.loc[:, train_feature])
            cv_score = cross_val_score(clf, data1, label, cv=cv, n_jobs=n_jobs, error_score='raise').mean()
            cv_scores.append(cv_score)
        max_index = np.array(cv_scores).argmax()
        max_score = max(cv_scores)
        max_scores.append(max_score)
        selected_feature.append(feature_names2[max_index])
        feature_names2.remove(feature_names2[max_index])
    return selected_feature, max_scores


def BSS_fun(feature_names, clf, data, label, cv, n_jobs=4):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    max_scores.append(cross_val_score(clf, data, label, cv=cv, n_jobs=n_jobs).mean())  # 计算全部特征下的训练结果
    features_num = min([len(data.columns), 50])  # 判断特征数目是否大于50
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

def mae_mse_report(predicts,label,test_index,models_str=models_str):
    MAE,MSE = {},{}
    for t in range(len(predicts[0])):
        maes,mses = [],[]
        for i in range(len(predicts)):
            ytest = label[test_index[t]]
            maes.append(mean_absolute_error(ytest, predicts[i][t]))
            mses.append(mean_squared_error(ytest, predicts[i][t]))
        MAE[t] = maes
        MSE[t] = mses
    MAE_report = pd.DataFrame(MAE,index=models_str).T
    MSE_report = pd.DataFrame(MSE,index=models_str).T
    MAE_report_describe = MAE_report.describe().loc[("mean",'min','max','std'),:]
    MSE_report_describe = MSE_report.describe().loc[("mean",'min','max','std'),:]
    return MAE_report,MSE_report,MAE_report_describe,MSE_report_describe

def regression_valreport(est,vdata,vlabel,maxfeatures):
    validate_r2,validate_predicts,validate_mae,validate_mse = [],[],[],[]
    for i in range(len(est)):
        validate_predict = est[i].predict(vdata[maxfeatures[i]])
        validate_predicts.append(validate_predict)
        validate_r2.append(est[i].score(vdata[maxfeatures[i]],vlabel))
        validate_mae.append(mean_absolute_error(vlabel,validate_predict))
        validate_mse.append(mean_squared_error(vlabel,validate_predict))
    val_report = pd.DataFrame({'R-square':validate_r2,'MAE':validate_mae,'MSE':validate_mse},index=models_str).T
    return val_report,validate_predicts


'''
PLOT METHODS
'''
# def mkvregpredplot(validate_predicts, vaildation_label, models_str=models_str):

#     trace = []
#     x = [i for i in range(len(vaildation_label))]
#     for i in range(len(validate_predicts)):
#         index = i + 1
#         if index == 1:
#             legendshow = True
#         else:
#             legendshow = False
#         sub_tracev = subplot_trace(x, vaildation_label, 'Actual', index, legendshow, models_str[i], '#1f77b4')
#         sub_tracep = subplot_trace(x, validate_predicts[i], 'Predicted', index, legendshow, models_str[i], '#fd7e14')
#         trace.append(sub_tracev)
#         trace.append(sub_tracep)
#     return trace

def get_p_value(arrA, arrB):
    r = stats.pearsonr(arrA,arrB)  #获取相关性
    return r

def mkvregpredplot(validate_predicts, vaildation_label, models_str=models_str):
    rs = []
    for i in range(len(validate_predicts)):
        rs.append(get_p_value(vaildation_label, validate_predicts[i]))
    data = []
    j = 0
    for pred, i in zip(validate_predicts, range(len(models_str))):
        marker = {
            'mode': 'markers',
            'type': 'scatter',
            'x': list(vaildation_label),
            'y': list(pred),
            'xaxis': 'x' + str(j + 1),
            'yaxis': 'y' + str(j + 1),
            'marker': {
                # line: {
                #     color: 'rgba(0,0,0,1.0)',
                #     width: 1.0
                # },
                # 'size': 4.47213595499958,
                'color': '#1f77b4',
                'symbol': 'dot'
            },
            'showlegend': False
        }
        minpred = min(pred)
        maxpred = max(pred)
        xl = np.arange(minpred, maxpred, (maxpred - minpred) / 10)
        line = {
            'line': {
                'dash': 'solid',
                'color': '#1f77b4',
                'width': 1.0
            },
            'mode': 'lines',
            # 'name': 'fit line',
            'type': 'scatter',
            'text': 'R: ' + str(np.round(rs[j][0], 3)) + '<br>pvalue: ' + str(np.round(rs[j][1], 3)),
            'x': xl,
            'y': xl,
            'xaxis': 'x' + str(j + 1),
            'yaxis': 'y' + str(j + 1),
            'showlegend': False
        }
        data.append(marker), data.append(line)
        j += 1
    return data



# def subplot_trace(x, y, name, index, legendshow, model_name, color):
#     trace = {
#         'line': {
#             'dash': 'solid',
#             'color': color,
#             # 'shape': 'hv',
#             # 'width': 2
#         },
#         'mode': 'lines',
#         'name': name,
#         'type': 'scatter',
#         'x': list(x),
#         'y': list(y),
#         'xaxis': 'x' + str(index),
#         'yaxis': 'y' + str(index),
#         'text': model_name,
#         'hoverinfo': 'text',
#         'showlegend': legendshow,
#         # 'legendgroup': 'High Risk'
#     }
#     return trace

def mkvreportbarplot(val_report):
    if 'Method' in val_report.columns:
        val_report = val_report.set_index('Method')
    trace = []
    j = 0
    for i in val_report.index:
        x = list(val_report.columns)
        y = list(val_report.loc[i, :])
        subtrace = {
            'x': x,
            'y': y,
            'type': 'bar',
            'xaxis': 'x' + str(j + 1),
            'yaxis': 'y' + str(j + 1),
            'showlegend': False
        }
        j += 1
        trace.append(subtrace)
    return trace

