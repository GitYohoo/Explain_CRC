#%%
import numpy as np
import pandas as pd
from lime import discretize,lime_tabular
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
from  joblib import dump, load
import csv
import os

from read_data import Read_data

x_train, x_test, y_train, y_test, feature_names = Read_data.data()

#%%
clf = load('jobmodels\\the2th_clf.joblib')
train, test, model = clf.Stacking(x_test, return_first_labels=True)
proba = model.predict_proba(test)
test_predict = model.predict(test)
test_acc = accuracy_score(y_test, test_predict)
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
print('开始解释....')
i = 10
if i == 10:
# for i in range(len(x_train)):
    output = []
    truelabel = y_train[i]
    print("该样本的真实标签为", truelabel)
    output.append("该样本的真实标签为"+str(truelabel))
    predlabel = clf.Stacking(x_train[i].reshape(1, -1))
    predlabel = int(predlabel)
    print("该样本的预测标签为", predlabel)
    output.append("该样本的预测标签为"+str(predlabel))
    # if predlabel != truelabel:
    #     continue
    exp= explainer.explain_instance(x_train[i],
                                    clf.proba_value,num_features=num_features,
                                    top_labels=4, model_regressor=None, num_samples=10000) #model_regressor:简单模型

    # exp.show_in_notebook(show_table=True, show_all=False)

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
        pd.DataFrame(merged_exp_values).to_csv('save_CRC_explaining\\whole_explain\\label_{}\\train_{}.csv'.format(label+1, i), 
                                            index=False, header=header)
        #追加output到csv
        with open('save_CRC_explaining\\whole_explain\\label_{}\\train_{}.csv'.format(label+1, i), 'a', newline='', encoding='gbk') as csvfile:
            for true_or_pred_label in output:
                writer = csv.writer(csvfile)
                writer.writerow([true_or_pred_label])
    exp.save_to_file('D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\html\\train_{}.html'.format(i))

#%%
import os
import pandas as pd
import csv
path = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\label_{}\\'
def get_char_count(path):
    #统计排序
    files = os.listdir(path)
    char_count_positive = {}
    char_count_negative = {}

    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file), encoding='gbk')
            for i in range(0, 10):
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
    return  char_count_negative.items()

def write_csv(char_count_sorted, label):
    #把最终结果写到csv
    #按照逗号分隔
    values = [str(i[0]) + ',' + str(i[1]) for i in char_count_sorted]
    #按照逗号分割成三列
    char_count_sorted = [i.split(',') for i in values]
    with open('D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\label_{}_n.csv'.format(label), 'w', newline='', encoding='gbk') as csvfile:
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
# %% 统计所有的解释
import os
import pandas as pd
import csv

# 获取SVM下所有label的csv文件路径
path_labels = r'D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\whole_explain\\'
#删除path——labels下的labels.csv
if os.path.exists(path_labels + 'labels.csv'):
    os.remove(path_labels + 'labels.csv')
csv_files = []
csv_files = [os.path.join(path_labels, f) for f in os.listdir(path_labels) if os.path.isfile(os.path.join(path_labels, f)) and f.endswith('.csv')]
#将0和1调换位置，2和3调换位置，4和5调换位置，4和5调换位置，6和7调换位置
for i in range(len(csv_files)):
    if i % 2 == 0:
        csv_files[i], csv_files[i+1] = csv_files[i+1], csv_files[i]
# 读取所有csv文件并拼接
labels_df = pd.DataFrame()
for csv_file in csv_files:
    #横向拼接
    labels_df = pd.concat([labels_df, pd.read_csv(csv_file, encoding='gbk')], axis=1)

# 将拼接后的DataFrame保存为labels.csv
labels_df.to_csv(path_labels + 'labels.csv', index=False)


#检查D:\Desktop\CRC_Explaining the Predictions\save_CRC_explaining\firstlayer\XGB\labels.csv这个文件的每一个单元格，在所有单元格前面加一个单引号
with open(path_labels + 'labels.csv', 'r', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

# Modify the rows and write them to a new CSV file
modified_rows = []
for row in rows:
    modified_row = ["'" + cell for cell in row]
    modified_rows.append(modified_row)

with open(path_labels + 'labels_new.csv', 'w', newline='', encoding='gbk') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(modified_rows)

# %%
from lime import submodular_pick

sp_exp = submodular_pick.SubmodularPick(explainer,
                                            data=x_train,
                                            predict_fn=clf.proba_value,
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
                                clf.Stacking, 
                                threshold=0.95) #thershold临界点

print('Prediction: ', explainer.class_names[int(clf.Stacking(x_test[idx].reshape(1, -1))[0])])
print("该样本的真实标签为", class_names[int(y_test[idx])])
exp.show_in_notebook()
# %%
print('Prediction: ', explainer.class_names[int(clf.Stacking(x_test[idx].reshape(1, -1))[0])])