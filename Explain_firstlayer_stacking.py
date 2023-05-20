#%%
import numpy as np
import pandas as pd
from lime import discretize,lime_tabular
from  joblib import dump, load
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split   # 数据集划分
from models import models
from read_data import Read_data
import csv

data = Read_data()
x_train, x_test, y_train, y_test, feature_names1 = data.data()

#%%第二层解释
secondlayer_model = load(r'D:\Desktop\Explain_CEC_Recording\jobmodels\secondlayer_clf.joblib')
test = pd.read_csv('data/xtest.csv', header=0)
train = pd.read_csv('data/xtrain.csv', header=0)
proba = secondlayer_model.predict_proba(test)
test_predict = secondlayer_model.predict(test)
test_acc = accuracy_score(y_test, test_predict) #测试​集准确率
print("这是第2个分类器")
print("测试集准确率: {0:.3f}".format(test_acc))
print(confusion_matrix(y_test,test_predict))
print(classification_report(y_test,test_predict))
#创建解释器
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称
#创建一个0~len(test+1)的数组，用于存放离散化的特征
categorical_features = np.arange(len(train))
explainer2 = lime_tabular.LimeTabularExplainer(train, discretize_continuous=True,    #true是选择分位
                                                discretizer='quartile',
                                                kernel_width=None, verbose=True, feature_names=feature_names,
                                                mode='classification', class_names=class_names,
                                                training_labels=y_train,
                                                feature_selection='lasso_path',
                                                categorical_features=categorical_features)  
                                                # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation'
                                                # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
print('开始第二层解释....')
from Save_exp import save_exp
print('开始解释....')
sample = [46]
for i in sample:
    output = []
    truelabel = y_train[i]
    print("该样本的真实标签为", truelabel)
    output.append("该样本的真实标签为"+str(truelabel))
    predlabel = secondlayer_model.predict(train[i].reshape(1, -1))
    predlabel = int(predlabel)
    print("该样本的预测标签为", predlabel)
    output.append("该样本的预测标签为"+str(predlabel))
    if truelabel != predlabel: #跳出循环
        continue    
    exp = explainer2.explain_instance(train[i],
                                    secondlayer_model.predict_proba,num_features=num_features,
                                    top_labels=1, model_regressor=None, num_samples=10000) #model_regressor:简单模型

    # exp_picture = exp.show_in_notebook(show_table=True, show_all=False)
    exp.save_to_file(r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\html\\train_{}.html'.format(i))

    local_exp_values = exp.local_exp[truelabel]
    #取出 local_exp_values中的第一列
    sortted_index = [i[0] for i in local_exp_values]
    #取出local_exp_values中的第二列
    sortted_contribution = [j[1] for j in local_exp_values]
    #将train的值按照sortted_index的顺序进行排序
    sortted_value = train[i][sortted_index]
    #将sortted_index中的特征名字提取出来
    sortted_models = [feature_names[i] for i in sortted_index]

    x = 0
    the_firstlayer_model_to_explian = []
    for value in sortted_value:
        if value == truelabel and sortted_contribution[x] > 0:
            print(sortted_models[x], "排名第", x + 1)
            the_firstlayer_model_to_explian.append(sortted_models[x])
        
        if value == truelabel and sortted_contribution[x] < 0:
            print("虽然", sortted_models[x], "与预测相同但是它的贡献值为负")

        if the_firstlayer_model_to_explian is None:
            print("没有模型预测与真实标签相同，拒绝解释")
            #抛出错误并退出
            raise SystemExit
        x = x + 1
    categorical_features2 = [0, 1, 2, 24, 25, 38]
    explainer = lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=True,    #true是选择分位
                                                    discretizer='quartile',
                                                    kernel_width=None, verbose=True, feature_names=feature_names1,
                                                    mode='classification', class_names=class_names,
                                                    training_labels=y_train,
                                                    feature_selection='lasso_path',
                                                    categorical_features=categorical_features2)  
                                                    # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation'
                                                    # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
    fitted_firstlayer_models = []
    for i in range(4):
        fitted_firstlayer_models.append(load('D:\\Desktop\\Explain_CEC_Recording\\jobmodels\the{}th_firstlayer_clf.joblib'.format(i)))
    for firstlayer_model in the_firstlayer_model_to_explian:
        if firstlayer_model == 'SVM':
            firstlayer_model = fitted_firstlayer_models[0]
            path = r"D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\SVM\\"
        if firstlayer_model == 'LR':
            firstlayer_model = fitted_firstlayer_models[1]
            path = r"D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\LR\\"
        if firstlayer_model == 'XGB':
            firstlayer_model = fitted_firstlayer_models[2]
            path = r"D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\XGB\\"
        if firstlayer_model == 'RF':
            firstlayer_model = fitted_firstlayer_models[3] 
            path = r"D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\RF\\"
        else:
            path = ""
        
        print("正在解释的是", firstlayer_model)
        #十折交叉验证
                                                    
        first_exp =  explainer.explain_instance(x_train[i],
                                        firstlayer_model.predict_proba,num_features=num_features,
                                        top_labels=4, model_regressor=None, num_samples=10000) #model_regressor:简单模型
        # first_exp_picture = first_exp.show_in_notebook(show_table=True, show_all=False)
        for label in range(4):
            csv_path = path + '\\label_{}\\train_{}.csv'.format(label+1, i)
            html_path = path + 'html\\train_{}.html'.format(i)
            save_exp(exp, i, output, label, csv_path, html_path)
#%%*******************************************************************************************************************
from Count_exp import count_exp
from Count_exp import sum_all
from Count_exp import quotation_marks
filenames = ['LR', 'RF', 'SVM', 'XGB']
for filename in filenames:
    org_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\{}\label_{}\\'.format(filename)
    count_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\{}\\'.format(filename)
    count_exp(org_path, count_path)
    sum_path =  r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\{}\\'.format(filename)
    sum_all(sum_path)  
    quotation_marks(sum_path)


# %%
