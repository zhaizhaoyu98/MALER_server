from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split,RepeatedKFold,KFold
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

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x:'%.3f'%x)

def regression_cp_result(request):
    import sys
    print(sys.path)
    data = pd.read_csv(r'C:\Users\Administrator\Desktop\jupyter_project\regression\regression_data.csv', header=0, index_col=0).T

    x_dum, y = regression_preprocess(data)
    nordata4, vaildation_data, nor_age4, vaildation_label = train_test_split(x_dum, y, random_state=10,
                                                                             train_size=0.7)  # 分验证集
    features = selectkbest_top20(nordata4, nor_age4, score_func=f_regression, k=50)
    nordata4 = nordata4[features]

    train_index, test_index = RegressionKFold(nordata4, nor_age4)
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=10)
    reg_model_name = 'LinearRegression'  # 获取用户选择的模型

    # Alphas=[0.0001,0.001,0.005,0.05,0.1,0.01]
    reg_cust_model = LinearRegression()  # 选择模型
    # fss,bss
    sf, ms = BSS_fun(features, reg_cust_model, nordata4, nor_age4, cv, n_jobs=6)
    print('ms: ',ms)
    max_index = np.array(ms).argmax()
    # max_index = ms.index(np.nanmax(ms))
    max_score = max(ms)
    max_features = (sf[:max_index + 1])
    preds, tests, res = [], [], []
    for i in range(len(train_index)):
        xtrain, ytrain = nordata4.iloc[train_index[i], :], nor_age4[train_index[i]]
        xtest, ytest = nordata4.iloc[test_index[i], :], nor_age4[test_index[i]]
        xtrain, xtest = xtrain[max_features], xtest[max_features]
        print('max_features: ',max_features, len(max_features))
        estimator, test_acc, predict = train_estimator(reg_cust_model, xtrain, ytrain, xtest, ytest)
        tests.append(test_acc), res.append(estimator), preds.append(predict)
    return render(request, 'regression_cp_result.html', {

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
    cv = KFold(n_splits=10, shuffle=True, random_state=10)
    feature_names = features
    data2 = data2[feature_names].to_numpy()
    # ifs方法得到前三分类器选择的特征数
    clf = copy.deepcopy(model)
    features_num = min([len(features), 20])
    cv_scores = [cross_val_score(clf, data2[:, :i], label, cv=cv, n_jobs=4).mean() for i in range(1, features_num + 1)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:, 0].sort_values(ascending=False).index[:3] + 1)
    return clf_num, cv_scores


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

def FSS_fun(feature_names, clf, data, label, cv, n_jobs=4):
    feature_names2 = list(feature_names)
    selected_feature = []
    max_scores = []
    features_num = min([len(feature_names), 20])
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