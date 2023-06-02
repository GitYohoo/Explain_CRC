#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import focal_loss
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from read_data import Read_data
data = Read_data()
DATA, targets = Read_data().Transformer_data(1)
#%%
# 定义Transformer分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, num_heads, dropout):
        super().__init__()

        # 定义Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 句子编码为平均向量
        x = self.classifier(x)
        return x

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.targets)

# 定义训练函数
def train(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5.0) 
        optimizer.step()
        total_loss += loss.item()
        total_correct += torch.sum(torch.argmax(output, dim=1) == y).item()
    accuracy = total_correct / len(dataloader.dataset)
    scheduler.step()
    return total_loss / len(dataloader), accuracy

# 定义测试函数
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_preds = []
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            total_preds += pred.tolist()
            total_correct += torch.sum(pred == y).item()

    accuracy = total_correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy, total_preds

# 准备数据
data = DATA
# 参数设置
input_dim = DATA.shape[2]  
output_dim = 4  
num_layers = 3
hidden_dim = 256
num_heads = 10
dropout = 0.1
batch_size = 64
learning_rate = 0.0001
num_epochs = 65


targets = torch.from_numpy(targets).long()  # 转换为LongTensor 
dataset = MyDataset(data, targets)

 
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)
train_data = train_data.to(device)
test_data = test_data.to(device)
train_targets = train_targets.to(device)
test_targets = test_targets.to(device)
train_dataset = MyDataset(train_data, train_targets)
test_dataset = MyDataset(test_data, test_targets)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = TransformerClassifier(input_dim, output_dim, num_layers, hidden_dim, num_heads, dropout).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100,
                                            num_training_steps=num_epochs * len(train_dataloader))
                                            
# 记录训练过程中的损失和准确率
train_losses = []
test_losses = []
trainaccuracies = []
testaccuracies = []

# 训练模型
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, scheduler)
    test_loss, test_acc , pred= test(model, test_dataloader, criterion)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, trainAccuracy={train_acc:.4f}, Test Loss={test_loss:.4f}, testAccuracy={test_acc:.4f}")
    
    # 记录损失和准确率
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    trainaccuracies.append(train_acc)
    testaccuracies.append(test_acc)

# 绘制训练过程曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.plot(trainaccuracies, label='trainAccuracy')
plt.plot(testaccuracies, label='testAccuracy')
plt.legend()
plt.show()

# %%
# # 绘制混淆矩阵
y_pred = np.array(pred)
y_true = test_targets.cpu().numpy()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true, y_pred, target_names=['AWNP', 'AWP', 'DWNP', 'DWP']))
# %%
