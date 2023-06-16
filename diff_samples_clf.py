#%%
import read_data
import pandas as pd
import numpy as np
from Create_Ensemble import Create_ensemble
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

SVM_ACC = []
SVM_F1 = []
SVM_Recall = []
SVM_Precision = []

LR_ACC = []
LR_F1 = []
LR_Recall = []
LR_Precision = []

XGB_ACC = []
XGB_F1 = []
XGB_Recall = []
XGB_Precision = []

RF_ACC = []
RF_F1 = []
RF_Recall = []
RF_Precision = []

j = 10
for i  in range(10):
    Data = read_data.Read_data()
    x_train, x_test, y_train, y_test= Data.Transformer_data_286(Normalization=True, data2tensor=False, zero=False)
    
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

    # DT_param = pd.read_csv(r'D:\Desktop\Stacking\Stacking_Direct_classification\DT\selected_feature_num_298\test\all_performance\{}_result_DT.csv'.format(j))
    # DT_param = np.array(DT_param)
    # DT_clf = DecisionTreeClassifier(random_state=42, min_samples_split=int(DT_param[0, 1]), min_samples_leaf=int(DT_param[0, 2]),max_depth=int(DT_param[0, 3]))

    models_to_fit = [SVM_clf, LR_clf, XGB_clf, RF_clf]

    kfd = Create_ensemble(n_splits=10, base_models=models_to_fit) #十折交叉验证
    first_train_label, first_test_label, clf = kfd.predict(x_train, y_train, x_test) #模型预测结果
  
    SVM_result = first_test_label[:, 0]
    LR_result = first_test_label[:, 1]
    XGB_result = first_test_label[:, 2]
    RF_result = first_test_label[:, 3]

    #画SVM的混淆矩阵

    print(confusion_matrix(y_test,SVM_result))
    print(classification_report(y_test,SVM_result))
    #添加SVM的评价指标
    SVM_ACC.append(classification_report(y_test,SVM_result,output_dict=True)['accuracy'])
    SVM_F1.append(classification_report(y_test,SVM_result,output_dict=True)['weighted avg']['f1-score'])
    SVM_Recall.append(classification_report(y_test,SVM_result,output_dict=True)['weighted avg']['recall'])
    SVM_Precision.append(classification_report(y_test,SVM_result,output_dict=True)['weighted avg']['precision'])

    #画LR的混淆矩阵
    print(confusion_matrix(y_test,LR_result))
    print(classification_report(y_test,LR_result))
    #添加LR的评价指标
    LR_ACC.append(classification_report(y_test,LR_result,output_dict=True)['accuracy'])
    LR_F1.append(classification_report(y_test,LR_result,output_dict=True)['weighted avg']['f1-score'])
    LR_Recall.append(classification_report(y_test,LR_result,output_dict=True)['weighted avg']['recall'])
    LR_Precision.append(classification_report(y_test,LR_result,output_dict=True)['weighted avg']['precision'])

    #画XGB的混淆矩阵
    print(confusion_matrix(y_test,XGB_result))
    print(classification_report(y_test,XGB_result))
    #添加XGB的评价指标
    XGB_ACC.append(classification_report(y_test,XGB_result,output_dict=True)['accuracy'])
    XGB_F1.append(classification_report(y_test,XGB_result,output_dict=True)['weighted avg']['f1-score'])
    XGB_Recall.append(classification_report(y_test,XGB_result,output_dict=True)['weighted avg']['recall'])
    XGB_Precision.append(classification_report(y_test,XGB_result,output_dict=True)['weighted avg']['precision'])

    #画RF的混淆矩阵
    print(confusion_matrix(y_test,RF_result))
    print(classification_report(y_test,RF_result))
    #添加RF的评价指标
    RF_ACC.append(classification_report(y_test,RF_result,output_dict=True)['accuracy'])
    RF_F1.append(classification_report(y_test,RF_result,output_dict=True)['weighted avg']['f1-score'])
    RF_Recall.append(classification_report(y_test,RF_result,output_dict=True)['weighted avg']['recall'])
    RF_Precision.append(classification_report(y_test,RF_result,output_dict=True)['weighted avg']['precision'])

# %%
#计算各个指标的平均值
SVM_ACC_mean = np.mean(SVM_ACC)
SVM_F1_mean = np.mean(SVM_F1)
SVM_Recall_mean = np.mean(SVM_Recall)
SVM_Precision_mean = np.mean(SVM_Precision)

LR_ACC_mean = np.mean(LR_ACC)
LR_F1_mean = np.mean(LR_F1)
LR_Recall_mean = np.mean(LR_Recall)
LR_Precision_mean = np.mean(LR_Precision)

XGB_ACC_mean = np.mean(XGB_ACC)
XGB_F1_mean = np.mean(XGB_F1)
XGB_Recall_mean = np.mean(XGB_Recall)
XGB_Precision_mean = np.mean(XGB_Precision)

RF_ACC_mean = np.mean(RF_ACC)
RF_F1_mean = np.mean(RF_F1)
RF_Recall_mean = np.mean(RF_Recall)
RF_Precision_mean = np.mean(RF_Precision)

#打印
print('SVM_ACC_mean:',SVM_ACC_mean)
print('SVM_F1_mean:',SVM_F1_mean)
print('SVM_Recall_mean:',SVM_Recall_mean)
print('SVM_Precision_mean:',SVM_Precision_mean)

print('LR_ACC_mean:',LR_ACC_mean)
print('LR_F1_mean:',LR_F1_mean)
print('LR_Recall_mean:',LR_Recall_mean)
print('LR_Precision_mean:',LR_Precision_mean)

print('XGB_ACC_mean:',XGB_ACC_mean)
print('XGB_F1_mean:',XGB_F1_mean)
print('XGB_Recall_mean:',XGB_Recall_mean)
print('XGB_Precision_mean:',XGB_Precision_mean)

print('RF_ACC_mean:',RF_ACC_mean)
print('RF_F1_mean:',RF_F1_mean)
print('RF_Recall_mean:',RF_Recall_mean)
print('RF_Precision_mean:',RF_Precision_mean)
# %%
