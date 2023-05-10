#%%
import numpy as np
import pandas as pd
from libraries.lime import lime_tabular, discretize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
from models import models
from  joblib import dump, load
from read_data import Read_data
from sklearn.tree import export_graphviz
from graphviz import Source
import csv

x_train, x_test, y_train, y_test, feature_names = Read_data.data()
#%%
secondlayer_model = load(r'D:\Desktop\Explain_CEC_Recording\jobmodels\secondlayer_clf.joblib')
test = pd.read_csv('data/xtest.csv', header=0)
train = pd.read_csv('data/xtrain.csv', header=0)
proba = secondlayer_model.predict_proba(test)
test_predict = secondlayer_model.predict(test)
test_acc = accuracy_score(y_test, test_predict) #测试​集准确率
print("这是第2个分类器")
print("测试集准确率: {0:.3f}".format(test_acc))
print(confusion_matrix(y_test,test_predict))
print(classification_report(y_test,test_predict)) #
#%%创建解释器
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称 
train = np.array(train)
test = np.array(test)
#创建一个0~len(train+1)的数组，用于存放离散化的特征
categorical_features = np.arange(len(train)) #每一个特征都是离散化的
# categorical_features2 = [0, 1, 2, 24, 25, 38]
explainer = lime_tabular.LimeTabularExplainer(train, discretize_continuous=True,    #true是选择分位
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
    # #如果i=7, 22, 24, 35就跳过
    # if i == 7 or i == 22 or i == 24 or i == 35:
    #     continue
    output = []
    truelabel = y_train[i]
    print("该样本的真实标签为", truelabel)
    output.append("该样本的真实标签为"+str(truelabel))
    predlabel = secondlayer_model.predict(train[i].reshape(1, -1))
    predlabel = int(predlabel)
    print("该样本的预测标签为", predlabel)
    output.append("该样本的预测标签为"+str(predlabel))
    if truelabel != predlabel:
        continue
    exp = explainer.explain_instance(train[i],
                                    secondlayer_model.predict_proba,num_features=num_features,
                                    top_labels=4, model_regressor=None, num_samples=20000) #model_regressor:简单模型

    exp.show_in_notebook(show_table=True, show_all=False)
    # exp.as_pyplot_figure(label=int(truelabel))

    for label in range(4):
        csv_path = 'D:\\Desktop\\CRC_Explaining the Predictions\\save_CRC_explaining\\secondlayer\\label_{}\\train_{}.csv'.format(label+1, i)
        html_path = 'D:\\Desktop\\CRC_Explaining the Predictions\\save_CRC_explaining\\secondlayer\\html\\train_{}.html'.format(i)
        save_exp(exp, i, output, label, csv_path, html_path)

#%%
from Count_exp import count_exp
org_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\label_{}\\'
count_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\\'
count_exp(org_path, count_path)
#%%
from Count_exp import sum_all
sum_path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\\'
sum_all(sum_path)








#%%决策树决策过程
from sklearn.tree import export_graphviz
from graphviz import Source
dot_data = export_graphviz(secondlayer_model, out_file=None, feature_names=feature_names, 
                            filled=True, rounded=True, special_characters=True,
                            precision=2, class_names=class_names, impurity=False, 
                            node_ids=True, proportion=False, label='none', 
                            leaves_parallel=False, rotate=False)
graph = Source(dot_data)
# graph.view()
path = secondlayer_model.decision_path(test[i].reshape(1, -1))
# 获取样本在决策树上的节点列表
nodes = path.indices.tolist()
# 将节点列表转换为字符串格式
node_list = ' -> '.join(map(str, nodes))

# 在图像中添加文本标签以显示该样本的决策路径
graph.render(view=False)
graph.render(filename='DT', directory='./images/', format='png')
graph.render('DT', view=False)

# 打印该样本的决策路径和分类结果
print('样本{}的决策路径：{}'.format(i, node_list))
print('样本{}的分类结果：{}'.format(i, path))

#%% shap解释
import libraries.shap
from libraries.shap import links
libraries.shap.initjs()
# explain the model's predictions using SHAP values
explainer2 = libraries.shap.Explainer(secondlayer_model.predict,test, feature_names=feature_names) 
shap_values = explainer2(test)
# plot the first prediction's explanation
# for i in range(len(shap_values)):
#     shap.plots.waterfall(shap_values[i])
# shap.plots.force(shap_values)
libraries.shap.plots.waterfall(shap_values[i])



# %% optiLIME
from libraries.lime_stability.stability import LimeTabularExplainerOvr
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称
categorical_features = [0, 1, 2, 3]
explainer = LimeTabularExplainerOvr(train, feature_names=feature_names, class_names=class_names,
                                    categorical_features=categorical_features, verbose=True, mode='classification')

# %%
params = {"data_row": test[i],
          "predict_fn": secondlayer_model.predict_proba,
          "num_samples": 5000,
          "num_features": 10,
          "distance_metric": "euclidean"}
exp = explainer.explain_instance(**params)
exp.show_in_notebook(show_table=True)
# %%
csi, vsi = explainer.check_stability(n_calls=10,**params,index_verbose=False)
print("CSI: ",csi,"\nVSI: ",vsi,"\n") 
# %% anchors解释
from anchor import anchor_tabular
import numpy as np
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称 
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names = class_names, #类别名
    feature_names = feature_names, #特征名
    train_data = train,
    categorical_names={0: ['0', '1', '2', '3'], 1: ['0', '1', '2', '3'], 2: ['0', '1', '2', '3'], 3: ['0', '1', '2', '3']}
    )
# idx = 7
for idx in range(10):
    # np.random.seed(1)
    exp = explainer.explain_instance(test[idx], 
                                    secondlayer_model.predict, 
                                    threshold=0.95) #thershold临界点

    print('Prediction: ', explainer.class_names[int(secondlayer_model.predict(test[idx].reshape(1, -1))[0])])
    print("该样本的真实标签为", class_names[int(y_test[idx])])
    exp.show_in_notebook()

# %%
