#%%
import numpy as np
import pandas as pd
from lime import discretize,lime_tabular
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split   # 数据集划分
from models import models

class Read_data(object):
    def __init__(self) -> None:
        pass       

    def data():
        #读入原始数据
        df = pd.read_csv(r'E:\结直肠癌研究\abc\com_patient_sample_mrna.csv',header=None,index_col=0)
        data=df.T
        data = data.dropna(axis=1, how='any')
        data=data.drop(['P_ID'],axis=1)
        data['label']=pd.to_numeric(data['label'],errors='coerce')
        data=pd.DataFrame(data,dtype=np.float)

        feature=data.drop(['label'],axis=1)
        label=data['label'].values
        label=np.array(label)-1

        #  从原始数据中提取特征
        random_state=42       #随机数种子
        n_splits=10            #十折交叉验证
        pt=278
        p = 50 #基本特征数

        xtrain, xtest, y_train, y_test = train_test_split(feature, label, test_size=0.1, stratify=label,random_state=42)
        selected_feature_name_same = pd.read_csv(r'E:\结直肠癌研究\50_selected_feature_name_same.csv'.format(p))
        same_selected_feature_name = np.array(selected_feature_name_same)[0]    # 相同的特征名字

        selected_feature_name_diff = pd.read_csv(r'E:\结直肠癌研究\selected_feature_num_50\排序后重要度特征排序\importance_paixu_50.csv'.format(p, p))
        diff_selected_feature_name = np.array(selected_feature_name_diff)[0]   # 不同的特征名字

        same_diff_feature_name = np.zeros((2, len(same_selected_feature_name) + pt), dtype=object) 
        for s in range(len(same_selected_feature_name)):
            same_diff_feature_name[0, s] = same_selected_feature_name[s]
        for d in range(len(diff_selected_feature_name)):
            s = s + 1
            if d < pt:
                same_diff_feature_name[0, s] = diff_selected_feature_name[d]
        same_diff_selected = same_diff_feature_name[0]

        x_train = pd.DataFrame(xtrain, columns=same_diff_selected) #训练集
        x_test = pd.DataFrame(xtest, columns=same_diff_selected)  #测试集
        feature_names = x_test.columns

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        return x_train, x_test, y_train, y_test, feature_names
# %%
