#%%
# 导入所需的包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import focal_loss
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
import transtab
#忽略警告
import warnings
warnings.filterwarnings('ignore')

# # 加载数据的同时去掉第一行,第一行是特征名称
# rawdata = pd.read_csv('..\\data\\new_data.csv', header=0)
# #取出第一列作为标签
# targets = rawdata.iloc[:,-1] 
# #取出后面的列作为特征
# data = rawdata.iloc[:,0:-1]

# # scaler = MinMaxScaler() # # 创建MinMaxScaler对象
# # normalized_data = scaler.fit_transform(data)# # 对data进行归一化
# # data = pd.DataFrame(normalized_data, columns=data.columns)# # 将归一化后的数据重新转换为DataFrame

# #为了提高多头注意力机制头的个数，补充四列0到data的后面
# # 创建一个包含4列零值的数组
# zeros = np.zeros((data.shape[0], 4))
# # 将零值数组转换为DataFrame
# zeros_df = pd.DataFrame(zeros, columns=['Zero1', 'Zero2', 'Zero3', 'Zero4'])
# # 将零值DataFrame与原始数据data进行水平拼接
# data = pd.concat([data, zeros_df], axis=1)
# #将4列零值转化为int
# data[['Zero1', 'Zero2', 'Zero3', 'Zero4']] = data[['Zero1', 'Zero2', 'Zero3', 'Zero4']].astype('int')
# columns = data.columns
# # 挑选出[231, 265, 266, 267, 280]列作为二值特征
# binary_columns = columns[[231, 265, 266, 267, 280]]
# #将二值特征转化为0-1
# data[binary_columns] = data[binary_columns].astype('bool').astype('int')
# # 划分训练集和测试集,训练集占80%,测试集占20%
# train_data, x_test, train_targets, y_test = train_test_split(data, targets, test_size=0.2)
# # 标签值减1
# train_targets = train_targets - 1 
# y_test = y_test - 1 
# # 转化为tensor格式
# train_data = torch.tensor(train_data.values, dtype=torch.float32)
# x_test = torch.tensor(x_test.values, dtype=torch.float32)
# train_targets = torch.tensor(train_targets.values, dtype=torch.float32)
# y_test = torch.tensor(y_test.values, dtype=torch.float32)
from read_data import Read_data
data = Read_data()
train_data, x_test, train_targets, y_test = data.Transformer_data_486()
#%%
#其他列作为数值特征
# numerical_columns = columns.drop(binary_columns)
num_class = 4
num_attention_head = 8
device='cuda:0'
model = transtab.build_classifier(
    # binary_columns=binary_columns, numerical_columns=numerical_columns, 
                                  num_class=num_class, num_attention_head = num_attention_head, 
                                  device=device)
# %%
# train_x = pd.DataFrame(train_data)
# train_y = pd.Series(train_targets)
# trainset = [(train_x, train_y)]
trainset = [(train_data, train_targets)]

# x_test = pd.DataFrame(x_test)
# y_test = pd.Series(y_test)
# testset = [(x_test, y_test)]
testset = [(x_test, y_test)]

training_arguments = {
    'num_epoch':30,
    'output_dir':'./checkpoint',
    'batch_size': 32,
    'lr': 0.001,
    }
transtab.train(model, trainset, **training_arguments)
# %%
model.load('./checkpoint')
x_test, y_test = testset[0]
y_test = pd.DataFrame(y_test).squeeze()
ypred = transtab.predict(clf = model, x_test = x_test, y_test=y_test)

# %%
ypred = np.argmax(ypred, axis=1)
#打印准确率,忽略零值警告
print("test accuracy: ", np.mean(ypred == y_test))
print(classification_report(y_test, ypred))
# 画混淆矩阵
cm = confusion_matrix(y_test, ypred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
#打印训练集准确率
train_pred = transtab.predict(clf = model, x_test = train_data, y_test=train_targets)
train_pred = np.argmax(train_pred, axis=1)
print("train accuracy: ", np.mean(train_pred == train_targets))
print(classification_report(train_targets, train_pred))

# %%
