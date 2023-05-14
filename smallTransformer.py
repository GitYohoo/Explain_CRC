#%%
import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from read_data import Read_data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# # 加载数据的同时去掉第一行，第一行是特征名称
data = pd.read_csv('D:\\Desktop\\298features.csv', header=0)
#取出第一列作为标签
targets = data.iloc[:,0]
#取出后面的298列作为特征
data = data.iloc[:,1:]

# 划分训练集和测试集
train_data, x_test, train_targets, y_test = train_test_split(data, targets, test_size=0.2)
train_targets = train_targets - 1
y_test = y_test - 1
# 转化为tensor格式
train_data = torch.tensor(train_data.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
train_targets = torch.tensor(train_targets.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        # 初始化一个Transformer编码器层对象，该层由多头自注意力和前馈神经网络组成
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        # 初始化一个由多个Transformer编码器层组成的Transformer编码器对象
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 添加一个Dropout层
        self.dropout = nn.Dropout(dropout)
        # 初始化一个全连接层，用于将Transformer编码器的输出映射到所需的输出维度
        self.fc = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        # Add time dimension
        # x = x.unsqueeze(0) 
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        # Take mean over time dimension
        # x = x.mean(dim=1)
        # Pass through dropout layer
        x = self.dropout(x) 
        # Pass through linear layer 
        x = self.fc(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 返回数据和其对应的目标值
        x = self.data[idx]
        y = self.targets[idx]
        # 将目标值转换为1维张量的类索引形式
        return x, y.flatten().long()
#%%
# 初始化Transformer模型，优化器和损失函数
model = Transformer(input_dim=train_data.shape[1], output_dim=4, hidden_dim=128, num_layers=2, num_heads=2, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction='none')

# 创建自定义数据集和数据加载器对象
train_dataset = CustomDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 进行100个训练周期
train_losses = []
train_accs = []

for epoch in range(200):
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader): 
        # Gradient reset
        optimizer.zero_grad()
        
        # Transfer data to float tensor
        data = data.to(torch.float32)  
        
        # Add time dimension 
        # data = data.unsqueeze(0)
        
        # Squeeze target to 1D
        target = target.squeeze() 
        
        # Calculate output and loss
        output = model(data)
        loss = criterion(output, target) 

        # Explicitly create gradients for loss
        # grad_output = torch.ones_like(output)
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        
        # Record loss and accuracy
        running_loss += loss.mean().item()  
        _, predicted = torch.max(output.data, 1)  
        total += target.size(0)  
        correct += (predicted == target).sum().item()
        
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # Print epoch statistics
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")
#%%
import matplotlib.pyplot as plt    
# Plot training curves
plt.plot(train_losses, label='Training Loss')
plt.plot(train_accs, label='Training Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
#%%
model.eval()  # set the model in evaluation mode to disable dropout
# x_test = x_test.unsqueeze(0)  # add a batch dimension to the test data
with torch.no_grad():
    output = model(x_test)
    pred = output.argmax(dim=1)  # get the predicted class for each sample
    accuracy = (pred == y_test).sum().item() / len(y_test)  # calculate the accuracy of the model on the test set
    print(f"Accuracy: {accuracy}")

# %%
 # 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

y_pred = pred.numpy()
y_true = y_test.numpy()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# %%
