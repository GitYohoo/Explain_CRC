#%%
import numpy as np
import pandas as pd
from libraries.lime import lime_tabular
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
from  joblib import dump, load
import csv
import os

from read_data import Read_data

data = Read_data()
x_train, x_test, y_train, y_test, feature_names = data.data()

#%%
secondlayer_model = load(f'D:\Desktop\Explain_CEC_Recording\jobmodels\secondlayer_clf.joblib')
wsm = load(f'D:\Desktop\Explain_CEC_Recording\jobmodels\whole_stacking_clf.joblib')
test = pd.read_csv('data/xtest.csv', header=0)
train = pd.read_csv('data/xtrain.csv', header=0)
proba = secondlayer_model.predict_proba(test)
test_predict = secondlayer_model.predict(test)
test_acc = accuracy_score(y_test, test_predict) #测试​集准确率
print("这是第2个分类器")
print("测试集准确率: {0:.3f}".format(test_acc))
print(confusion_matrix(y_test,test_predict))
print(classification_report(y_test,test_predict)) 
#%%进行可解释性研究
#创建解释器
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
# feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称
categorical_features = [0, 1, 2, 24, 25, 38]
explainer = lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=True,    #true是选择分位
                                                discretizer='quartile',
                                                kernel_width=None, verbose=True, feature_names=feature_names,
                                                mode='classification', class_names=class_names,
                                                training_labels=y_train,
                                                feature_selection='lasso_path',
                                                categorical_features=categorical_features)  
                                                # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation'
                                                # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
#%%
from Save_exp import save_exp
print('开始解释....')
sample = [46]
# sample = list(range(len(test)))
for i in sample:
    output = []
    truelabel = y_train[i]
    print("该样本的真实标签为", truelabel)
    output.append("该样本的真实标签为"+str(truelabel))
    predlabel = wsm.predict(x_train[i].reshape(1, -1))
    predlabel = int(predlabel)
    print("该样本的预测标签为", predlabel)
    output.append("该样本的预测标签为"+str(predlabel))
    # if predlabel != truelabel:
    #     continue
    exp= explainer.explain_instance(x_train[i],
                                    wsm.proba_value,num_features=num_features,
                                    top_labels=4, model_regressor=None, num_samples=10000) #model_regressor:简单模型

    # exp.show_in_notebook(show_table=True, show_all=False)

    for label in range(4): 
        csv_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\label_{}\\atrain_{}.csv'.format(label+1, i)
        html_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\html\\atrain_{}.html'.format(i)
        save_exp(exp, i, output, label, csv_path, html_path)
#%%
from Count_exp import count_exp
org_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\label_{}\\'
count_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\\'
count_exp(org_path, count_path)
# %% 统计所有的解释
from Count_exp import sum_all
sum_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\\'
sum_all(sum_path)

from Count_exp import quotation_marks
quotation_marks(sum_path)

# %%
from lime import submodular_pick

sp_exp = submodular_pick.SubmodularPick(explainer,
                                            data=x_train,
                                            predict_fn=wsm.proba_value,
                                            method='full',
                                            sample_size=1000,
                                            num_exps_desired=5,
                                            num_features=10)
[exp.show_in_notebook() for exp in sp_exp.sp_explanations]
# %%
[exp.as_pyplot_figure(label=exp.available_labels()[0]) for exp in sp_exp.sp_explanations]

# %% anchors解释
from anchor import anchor_tabular
import numpy as np
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
categorical_features = [0, 1, 2, 24, 25, 38]
# feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称 
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names = class_names, #类别名
    feature_names = feature_names, #特征名
    train_data = x_train,
    categorical_names={0: ['0', '1'], 1: ['0', '1'], 2: ['0', '1'], 24: ['0', '1'], 24: ['0', '1'],}
    )
idx = 7
# for idx in range(10):
np.random.seed(1)
exp = explainer.explain_instance(x_test[idx], 
                                wsm.predict, 
                                threshold=0.95) #thershold临界点

print('Prediction: ', explainer.class_names[int(wsm.predict(x_test[idx].reshape(1, -1))[0])])
print("该样本的真实标签为", class_names[int(y_test[idx])])
exp.show_in_notebook()
# %%
print('Prediction: ', explainer.class_names[int(wsm.predict(x_test[idx].reshape(1, -1))[0])])