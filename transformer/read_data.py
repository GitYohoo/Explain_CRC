#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class Read_data(object):
    def __init__(self) -> None:
        pass       

    def data(self, csv_path=r'data\com_patient_sample_mrna.csv',
        selected_feature_name_same_path=r'data\50_selected_feature_name_same.csv',
        selected_feature_name_diff_path=r'data\importance_paixu_50.csv', 
        test_size=0.1, random_state=42):
        # 读入原始数据
        df = pd.read_csv(csv_path, header=None, index_col=0, low_memory=False)
        data = df.T
        data = data.dropna(axis=1, how='any')
        data = data.drop(['P_ID'], axis=1)
        data['label'] = pd.to_numeric(data['label'], errors='coerce')
        data = pd.DataFrame(data, dtype=np.float32)

        feature = data.drop(['label'], axis=1)
        label = data['label'].values
        label = np.array(label) - 1

        # 从原始数据中提取特征
        pt = 278

        xtrain, xtest, y_train, y_test = train_test_split(feature, label, test_size=test_size, stratify=label, random_state=random_state)
        selected_feature_name_same = pd.read_csv(selected_feature_name_same_path)
        same_selected_feature_name = np.array(selected_feature_name_same)[0]  # 相同的特征名字

        selected_feature_name_diff = pd.read_csv(selected_feature_name_diff_path)
        diff_selected_feature_name = np.array(selected_feature_name_diff)[0]  # 不同的特征名字

        same_diff_feature_name = np.zeros((2, len(same_selected_feature_name) + pt), dtype=object)
        for s in range(len(same_selected_feature_name)):
            same_diff_feature_name[0, s] = same_selected_feature_name[s]
        for d in range(len(diff_selected_feature_name)):
            s = s + 1 # type: ignore
            if d < pt:
                same_diff_feature_name[0, s] = diff_selected_feature_name[d]
        same_diff_selected = same_diff_feature_name[0]

        x_train = pd.DataFrame(xtrain, columns=same_diff_selected)  # 训练集
        x_test = pd.DataFrame(xtest, columns=same_diff_selected)  # 测试集
        feature_names = x_test.columns

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        return x_train, x_test, y_train, y_test, feature_names
    
    def Transformer_data(self, category=5):
        # 加载数据的同时去掉第一行,第一行是特征名称
        feature_name_data = pd.read_csv("..\\data\\score_selected_feature_name.csv", header=0)
        # 取出第一行
        feature_name = feature_name_data.iloc[0, :]
        rowdata = pd.read_csv("..\\data\\com_patient_sample_mrna.csv", header=None, index_col=0, low_memory=False)

        #把row index列中与feature_name中相同的行取出来
        data = rowdata.loc[feature_name, :]
        #data转置
        data = data.T
        data  = data.astype(float)
        #取出rowdata的label行作为标签
        targets = rowdata.loc["label", :]
        #转化为int
        targets = targets.astype(int)

        scaler = MinMaxScaler() # # 创建MinMaxScaler对象
        normalized_data = scaler.fit_transform(data)# # 对data进行归一化
        data = pd.DataFrame(normalized_data, columns=data.columns)# # 将归一化后的数据重新转换为DataFrame
        if category == 1:
            # 为了提高多头注意力机制头的个数，补充四列0到data的后面
            # 创建一个包含4列零值的数组
            zeros = np.zeros((data.shape[0], 3))
            # 将零值数组转换为DataFrame
            zeros_df = pd.DataFrame(zeros, columns=["Zero1", "Zero2", "Zero3"])
            # 将零值DataFrame与原始数据data进行水平拼接
            data = pd.concat([data, zeros_df], axis=1)
        
        # # 进行K-Means聚类,聚成
        # kmeans = KMeans(n_init = 10, n_clusters=category, random_state=0).fit(data.T) 
        # # 获取每个样本的聚类标签
        # labels = kmeans.labels_

        # # 根据标签将特征分组
        # groups = {}

        # for idx, label in enumerate(labels):
        #     feature = data.columns[idx]
        #     if label not in groups:
        #         groups[label] = [feature]
        #     else:
        #         groups[label].append(feature) 

        # total_sample = data.shape[0]
        # max_group_length = 0
        # for i in range(category):
        #     group = groups[i]
        #     group_length = len(group)
        #     #找出最大的组的长度
        #     if group_length > max_group_length:
        #         max_group_length = group_length
        # DATA = np.zeros((total_sample, category, max_group_length))
        # mask = np.ones((total_sample, category, max_group_length))
        # # DATA = [[] for _ in range(category)]
        # for i in range(category):
        #     group = groups[i]
        #     group_length = len(group)
        #     for j in range(group_length):
        #         feature = group[j]
        #         DATA[:, i, j] = data[feature].values
        #     # 将mask中的1改为0
        #     mask[:, i, group_length:] = 0
        # #DATA转化为tensor
        # DATA = torch.tensor(DATA, dtype=torch.float32)
        # #mask转化为tensor
        # mask = torch.tensor(mask, dtype=torch.float32)
        
        
        # tensor_list = [torch.tensor(l, dtype=torch.float32) for l in DATA]
        # DATA = torch.stack(tensor_list, dim=1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 划分训练集和测试集,训练集占80%,测试集占20%
        train_data, x_test, train_targets, y_test = train_test_split(
            data, targets, test_size=0.2
        )

        # 标签值减1
        train_targets = train_targets - 1
        y_test = y_test - 1
        # 转化为tensor格式
        # train_data = torch.tensor(train_data.values, dtype=torch.float32)
        # x_test = torch.tensor(x_test.values, dtype=torch.float32)
        # train_targets = torch.tensor(train_targets.values, dtype=torch.float32)
        # y_test = torch.tensor(y_test.values, dtype=torch.float32)


        # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        # train_targets = F.one_hot(train_targets.long(), num_classes=4) 
        # y_test = F.one_hot(y_test.long(), num_classes=4)

        # train_data = train_data.to(device) #size:Tensor(471, 490, 1)
        # x_test = x_test.to(device) #size:Tensor(118,  490, 1)
        # train_targets = train_targets.to(device)#size:Tensor(471, 4)
        # y_test = y_test.to(device) #size:Tensor(118, 4) 

        return train_data, x_test, train_targets, y_test
        # return DATA, np.array(targets)-1

# %%
