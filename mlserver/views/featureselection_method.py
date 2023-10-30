from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB,BernoulliNB,ComplementNB,MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_curve,auc,classification_report,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import copy
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# from mrmr import mrmr_classif,mrmr_regression

# info = form_action
def mrmr_fs(data,label,info,k=50):
    from mrmr import mrmr_classif, mrmr_regression
    k = min(len(data.columns),k)
    if info.split(",")[0] == 'classification':
        selected_id = mrmr_classif(X=data, y=label, K=k,n_jobs=4)
    else:
        selected_id = mrmr_regression(X=data, y=label, K=k,n_jobs=4)
    feature_names = list(data.loc[:,selected_id].columns)
    return feature_names

def selectkbest_top20(data,label,k=20,score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k='all').fit(data,label)
    df_scores = pd.DataFrame(selector.scores_)
    df_columns = pd.DataFrame(data.columns)
    df_feature_scores = pd.concat([df_columns, df_scores], axis=1)
    df_feature_scores.columns = ['Feature', 'Score']
    feature_names=df_feature_scores.sort_values(by='Score', ascending=False)[:k]['Feature']
    return feature_names

# Optimal Feature Subset Search Methods

def FSS_fun(feature_names,clf,data,label,cv,n_jobs=1):
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
def BSS_fun(feature_names,clf,data,label,cv,n_jobs=1):
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
    clf = copy.deepcopy(clf)
    res = clf.fit(xtrain,ytrain)
    predict = res.predict(xtest)
    test_acc = accuracy_score(ytest,predict,normalize=True,)
    return res,test_acc,predict

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

def pre_screening(data2,label,model,features,cv=2):
    #第一步筛选
    feature_names = features
    data2 = data2[feature_names].to_numpy()
    #ifs方法得到前三分类器选择的特征数
    clf = model
    # cv_scores = [cross_val_score(clf,data2[:,:i],label,cv=cv,).mean() for i in range(1,21)]
    features_num = min([len(features), 20])
    cv_scores = [cross_val_score(clf, data2[:, :i], label, cv=cv, n_jobs=1).mean() for i in range(1, features_num + 1)]
    clf_num = list(pd.DataFrame(cv_scores).iloc[:,0].sort_values(ascending=False).index[:3]+1)
    return clf_num, cv_scores