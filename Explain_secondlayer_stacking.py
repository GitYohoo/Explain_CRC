#%%
import numpy as np
import pandas as pd
from lime import lime_tabular, discretize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
from models import models
from  joblib import dump, load
from read_data import Read_data
from sklearn.tree import export_graphviz
from graphviz import Source
import csv

x_train, x_test, y_train, y_test, feature_names = Read_data.data(
        csv_path=r'E:\结直肠癌研究\abc\com_patient_sample_mrna.csv',
        selected_feature_name_same_path=r'E:\结直肠癌研究\50_selected_feature_name_same.csv',
        selected_feature_name_diff_path=r'E:\结直肠癌研究\selected_feature_num_50\排序后重要度特征排序\importance_paixu_50.csv'
    )
#%%
# for j in range(10):
#     clf = models(x_train, y_train, j)
#     xtrain, xtest, secondlayer_model = clf.Stacking(x_test, return_first_labels=True)
#     # 将训练好的分类器保存到文件中
#     dump(secondlayer_model, 'jobmodels\\the{}th_secondlayer_clf.joblib'.format(j))
#     test_predict = clf.Stacking(x_test) #二级模型输出结果标签
#     proba = clf.proba_value(x_test) #二级模型输出概率

#     test_acc = accuracy_score(y_test, test_predict)
#     print("这是第{0}个分类器".format(j))  
#     print("测试集准确率: {0:.3f}".format(test_acc))
#     print(confusion_matrix(y_test,test_predict))
#     print(classification_report(y_test,test_predict))
# secondlayer_model = load('jobmodels\\the2th_secondlayer_clf.joblib') 
clf = models(x_train, y_train, 2)
train, test, secondlayer_model = clf.Stacking(x_test, return_first_labels=True)
proba = secondlayer_model.predict_proba(test)
test_predict = secondlayer_model.predict(test)
test_acc = accuracy_score(y_test, test_predict) #
print("这是第2个分类器")
print("测试集准确率: {0:.3f}".format(test_acc))
print(confusion_matrix(y_test,test_predict))
print(classification_report(y_test,test_predict)) #
#%%创建解释器
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称 
#创建一个0~len(train+1)的数组，用于存放离散化的特征
categorical_features = np.arange(len(train))
categorical_features2 = [0, 1, 2, 24, 25, 38]
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
print('开始解释....')
i = 46 
# if i == 46:
for i in range(len(train)):
    # #如果i=7, 22, 24, 35就跳过
    # if i == 7 or i == 22 or i == 24 or i == 35:
    #     continue
    output = []
    truelabel = y_train[i]
    print("该样本的真实标签为", truelabel)
    output.append("该样本的真实标签为"+str(truelabel))
    predlabel = clf.Stacking(x_train[i].reshape(1, -1))
    predlabel = int(predlabel)
    print("该样本的预测标签为", predlabel)
    output.append("该样本的预测标签为"+str(predlabel))
    if truelabel != predlabel:
        continue
    exp = explainer.explain_instance(train[i],
                                    secondlayer_model.predict_proba,num_features=num_features,
                                    top_labels=4, model_regressor=None, num_samples=20000) #model_regressor:简单模型

    # exp.show_in_notebook(show_table=True, show_all=False)
    # exp.as_pyplot_figure(label=int(truelabel))

    for label in range(4): 
        #对每一个类别都进行解释的保存
        local_exp_values = exp.local_exp[label]
        #取出 local_exp_values中的第一列
        sortted_index = [i[0] for i in local_exp_values]
        #获取解释的各个特征
        list_exp_values  = exp.as_list(label=label)
        #去掉括号和引号
        for x in range(len(list_exp_values)):
            list_exp_values_str = str(list_exp_values[x])
            list_exp_values[x] = list_exp_values_str.replace('(', '').replace(')', '').replace("'", '')
        #拼接
        merged_exp_values = list(zip(local_exp_values, list_exp_values))
        #按照逗号分隔
        merged_exp_values = [str(i[0][0]) + ',' + str(i[1]) for i in merged_exp_values]
        #按照逗号分割成三列
        merged_exp_values = [i.split(',') for i in merged_exp_values]
        header = ['feature_numbers', 'feature_bins', 'contributions']
        pd.DataFrame(merged_exp_values).to_csv('save_CRC_explaining\\secondlayer\\label_{}\\train_{}.csv'.format(label+1, i), 
                                            index=False, header=header)
        #追加标签信息到csv
        with open('save_CRC_explaining\\secondlayer\\label_{}\\train_{}.csv'.format(label+1, i), 'a', newline='', encoding='gbk') as csvfile:
            for true_or_pred_label in output:
                writer = csv.writer(csvfile)
                writer.writerow([true_or_pred_label])
        exp.save_to_file('save_CRC_explaining\\secondlayer\\label_{}\\train_{}.html'.format(label+1, i))

#%%
import os
import pandas as pd
import csv
path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\label_{}\\'
def get_char_count(path):
    #统计排序
    files = os.listdir(path)
    char_count_positive = {}
    char_count_negative = {}

    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file), encoding='gbk')
            for i in range(0, 4):
                col = df.iloc[i, 1]
                contribution = df.iloc[i, 2]
                if contribution > 0:
                    if col in char_count_positive:
                        char_count_positive[col] += 1
                    else:
                        char_count_positive[col] = 1
                if contribution < 0:
                    if col in char_count_negative:
                        char_count_negative[col] += 1
                    else:
                        char_count_negative[col] = 1
    return  char_count_positive.items()

def write_csv(char_count_sorted, label):
    #把最终结果写到csv
    #按照逗号分隔
    values = [str(i[0]) + ',' + str(i[1]) for i in char_count_sorted]
    #按照逗号分割成三列
    char_count_sorted = [i.split(',') for i in values]
    with open('D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\label_{}_p.csv'.format(label), 'w', newline='', encoding='gbk') as csvfile:
        for char in char_count_sorted:
            writer = csv.writer(csvfile)
            writer.writerow(char)
#排序
char_count_sorted_1 = sorted(get_char_count(path.format(1)), key=lambda x: x[1], reverse=True)
char_count_sorted_2 = sorted(get_char_count(path.format(2)), key=lambda x: x[1], reverse=True)
char_count_sorted_3 = sorted(get_char_count(path.format(3)), key=lambda x: x[1], reverse=True)
char_count_sorted_4 = sorted(get_char_count(path.format(4)), key=lambda x: x[1], reverse=True)
#写入csv
for i in range(4):
    write_csv(eval('char_count_sorted_{}'.format(i+1)), i+1)


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


#%%
import shap
from shap import links
shap.initjs()
# explain the model's predictions using SHAP values
explainer2 = shap.Explainer(secondlayer_model.predict,test, feature_names=feature_names) 
shap_values = explainer2(test)
# plot the first prediction's explanation
# for i in range(len(shap_values)):
#     shap.plots.waterfall(shap_values[i])
# shap.plots.force(shap_values)
shap.plots.waterfall(shap_values[i])

#%%
import os
import pandas as pd

# 获取SVM下所有label的csv文件路径
path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\\'
csv_files = []
csv_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]
#将0和1调换位置，2和3调换位置，4和5调换位置，4和5调换位置，6和7调换位置
for i in range(len(csv_files)):
    if i % 2 == 0:#
        csv_files[i], csv_files[i+1] = csv_files[i+1], csv_files[i]
# 读取所有csv文件并拼接
labels_df = pd.DataFrame()
for csv_file in csv_files:
    #横向拼接
    labels_df = pd.concat([labels_df, pd.read_csv(csv_file, encoding='gbk')], axis=1)

# 将拼接后的DataFrame保存为labels.csv
headers = ['AWNP+','AWNP-','AWP+','AWP-','DWNP+','DWNP-','DWNP+','DWNP-']
labels_df.to_csv(r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\secondlayer\labels.csv', index=False)
# %% optiLIME
from lime_stability.stability import LimeTabularExplainerOvr
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
