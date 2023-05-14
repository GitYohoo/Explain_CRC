#%%
import torch  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from read_data import Read_data

train_data, x_test, train_targets, y_test, feature_names = Read_data.data()   

train_data = torch.tensor(train_data)
train_data = train_data.to(torch.float32)
x_test = torch.tensor(x_test)
x_test = x_test.to(torch.float32)
train_targets = torch.tensor(train_targets)
train_targets = train_targets.to(torch.float32)
y_test = torch.tensor(y_test)
y_test = y_test.to(torch.float32)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        # 初始化一个Transformer编码器层对象，该层由多头自注意力和前馈神经网络组成
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        # 初始化一个由多个Transformer编码器层组成的Transformer编码器对象
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # 初始化一个全连接层，用于将Transformer编码器的输出映射到所需的输出维度
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Add time dimension
        x = x.unsqueeze(0) 
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        # Take mean over time dimension
        x = x.mean(dim=1) 
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
        return x, y.squeeze().long()
#%%
# 初始化Transformer模型，优化器和损失函数
model = Transformer(input_dim=train_data.shape[1], output_dim=4, hidden_dim=128, num_layers=2, num_heads=2, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 创建自定义数据集和数据加载器对象
train_dataset = CustomDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 进行100个训练周期
for epoch in range(100):
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
        
        # Backpropagate error and update weights
        loss.backward()
        optimizer.step()

# model.eval()  # set the model in evaluation mode to disable dropout
# x_test = x_test.unsqueeze(0)  # add a batch dimension to the test data
# all_pred = []
# with torch.no_grad():
#     output = model(x_test)
#     pred = output.argmax(dim=1)  # get the predicted class for each sample
#     all_pred.append(pred)
#     accuracy = (pred == y_test).sum().item() / len(y_test)  # calculate the accuracy of the model on the test set
#     print(f"Accuracy: {accuracy}")

# %%
 # 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

y_pred = np.array(all_pred)
y_true = y_test.numpy()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# %%
