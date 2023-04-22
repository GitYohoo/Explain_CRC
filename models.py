import pandas as pd
import numpy as np
from Create_Ensemble import Create_ensemble
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

class models(object):
    def __init__(self, rfe_xtrain_bing_same, ytrain, j):
        self.rfe_xtrain_bing_same = rfe_xtrain_bing_same
        self.ytrain = ytrain
        self.j = j


    def fit_model(self, model_to_fit):
        for fitted_models in model_to_fit:
            fitted_models.fit(self.rfe_xtrain_bing_same, self.ytrain)

    def predict_vales(self, fitted_models, test):
        first_train_label = []
        first_test_label = []
        for model in fitted_models:
            train_label = model.predict(self.rfe_xtrain_bing_same)
            test_label = model.predict(test)
            first_train_label.append(train_label)
            first_test_label.append(test_label)
        first_train_label = np.array(first_train_label).T
        first_test_label = np.array(first_test_label).T

        return first_train_label, first_test_label 
    
    def read_param(self):
        #Stacking第一层当中分类器的设置
        j = self.j
        SVM_param = pd.read_csv(r'D:\Desktop\Stacking\Stacking_Direct_classification\SVM\selected_feature_num_298\test\all_performance\{}_result_SVM.csv'.format(j))
        SVM_param = np.array(SVM_param)
        SVM_clf = SVC(random_state=42,kernel=SVM_param[0,1],C=SVM_param[0,2], probability=True)

        LR_param = pd.read_csv(r'D:\Desktop\Stacking\Stacking_Direct_classification\LR\selected_feature_num_298\test\all_performance\{}_result_LR.csv'.format(j))
        LR_param = np.array(LR_param)
        LR_clf = LogisticRegression(random_state=42, solver=LR_param[0, 1], C=LR_param[0, 2])
        
        XGB_param = pd.read_csv(r'D:\Desktop\Stacking\Stacking_Direct_classification\XGB\selected_feature_num_298\test\all_performance\{}_result_XGB.csv'.format(j))
        XGB_param = np.array(XGB_param)
        XGB_clf = XGBClassifier(random_state=42,n_estimators=int(XGB_param[0,1]),learning_rate=XGB_param[0,2],max_depth=int(XGB_param[0,3]), eval_metric='mlogloss')
        
        RF_param = pd.read_csv(r'D:\Desktop\Stacking\Stacking_Direct_classification\RF\selected_feature_num_298\test\all_performance\{}_result_RF.csv'.format(j))
        RF_param = np.array(RF_param)
        RF_clf = RandomForestClassifier(random_state=42, n_estimators=int(RF_param[0, 1]), min_samples_split=int(RF_param[0, 2]),min_samples_leaf=int(RF_param[0, 3]),max_depth=int(RF_param[0, 4]))

        model_to_fit = [SVM_clf, LR_clf, XGB_clf, RF_clf]
        #Stacking第二层当中分类器的设置
        DT_param = pd.read_csv(r'D:\Desktop\Stacking\Stacking_Direct_classification\DT\selected_feature_num_298\test\all_performance\{}_result_DT.csv'.format(j))
        DT_param = np.array(DT_param)
        self.DT_clf = DecisionTreeClassifier(random_state=42, min_samples_split=int(DT_param[0, 1]), min_samples_leaf=int(DT_param[0, 2]),max_depth=int(DT_param[0, 3]))

        return model_to_fit

    def Stacking(self, test, return_first_labels=False, proba=False, return_firstlayer_models=False):
        self.test = test
        
        models_to_fit = self.read_param() #读取参数
        kfd1 = Create_ensemble(n_splits=10, base_models=models_to_fit) #十折交叉验证
        first_train_label, first_test_label, clf1 = kfd1.predict(self.rfe_xtrain_bing_same, self.ytrain, self.test) #一级模型预测结果

        kfd2 = Create_ensemble(n_splits=10, base_models=[self.DT_clf]) #十折交叉验证
        second_train_label, second_test_label, clf2 = kfd2.predict(first_train_label, self.ytrain, first_test_label) #二级模型预测jieguo 
        
        if return_first_labels:
            return first_train_label, first_test_label, clf2 #二级模型的训练、测试集以及模型
        if proba == True and return_first_labels == False:
            second_test_proba = clf2.predict_proba(first_test_label)
            return second_test_proba #二级模型测试集预测概率
        if return_firstlayer_models:
            firstlayer_models = kfd1.predict(self.rfe_xtrain_bing_same, self.ytrain, self.test, return_firstlayer_models=True)
            return firstlayer_models #一级模型
        else:
            return  second_test_label.ravel()#二级模型测试集预测结果
    
    def proba_value(self, testi):
        values = self.Stacking(testi, proba=True)
        return values
    




    
