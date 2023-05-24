#%%
import torch
import torch.nn as nn
from libraries.tab_transformer_pytorch.ft_transformer import FTTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# 加载数据的同时去掉第一行,第一行是特征名称
rawdata = pd.read_csv('data\\new_data.csv', header=0)
#取出第一列作为标签
targets = rawdata.iloc[:,-1] 
#取出后面的列作为特征
data = rawdata.iloc[:,0:-1]

# scaler = MinMaxScaler() # # 创建MinMaxScaler对象
# normalized_data = scaler.fit_transform(data)# # 对data进行归一化
# data = pd.DataFrame(normalized_data, columns=data.columns)# # 将归一化后的数据重新转换为DataFrame

#为了提高多头注意力机制头的个数，补充四列0到data的后面
# 创建一个包含4列零值的数组
zeros = np.zeros((data.shape[0], 4))
# 将零值数组转换为DataFrame
zeros_df = pd.DataFrame(zeros, columns=['Zero1', 'Zero2', 'Zero3', 'Zero4'])
# 将零值DataFrame与原始数据data进行水平拼接
data = pd.concat([data, zeros_df], axis=1)

# 划分训练集和测试集,训练集占80%,测试集占20%
train_data, x_test, train_targets, y_test = train_test_split(data, targets, test_size=0.2)
# 标签值减1
train_targets = train_targets - 1 
y_test = y_test - 1 
# 转化为tensor格式
train_data = torch.tensor(train_data.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
train_targets = torch.tensor(train_targets.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
train_targets = train_targets.view(-1, 1)  # Reshape the target tensor

#%%
model = FTTransformer(
    categories=[2, 2, 2, 2, 2],     # 5 discrete features with 2 categories each 
    num_continuous=285,          # 290 total features - 5 discrete features = 226 continuous features 
    dim=64, 
    # dim_out=1,
    depth=6, 
    heads=5,
)

# Train the model
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#取出train_data的[231, 265, 266, 267, 280]列作为离散特征
x_categ = train_data[:,[231, 265, 266, 267, 280]].long()
#去除train_data的[231, 265, 266, 267, 280]列作为连续特征
x_cont = train_data[:,[i for i in range(290) if i not in [231, 265, 266, 267, 280]]].float()
#转化为tensor格式
x_categ = torch.tensor(x_categ, dtype=torch.long)
x_cont = torch.tensor(x_cont, dtype=torch.float32)

#%%
for epoch in range(100):
    logits = model(x_categ, x_cont)
    loss = criterion(logits, train_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1 == 0:
        print(f'Epoch {epoch+1}: Loss {loss.item():.4f}')
        
        # Make predictions on train set
        with torch.no_grad():
            train_logits = model(x_categ, x_cont)
            train_preds = train_logits.round()
            
        # Calculate train accuracy
        train_acc = (train_preds == train_targets).float().mean()
        print(f'Train accuracy: {train_acc:.4f}')
#%%
# Make predictions 
with torch.no_grad():
  logits = model(x_categ, x_cont)
  predictions = logits.round()

# Evaluate accuracy
acc = (predictions == y_test).float().mean()
print(f'Accuracy: {acc}')
# %%
