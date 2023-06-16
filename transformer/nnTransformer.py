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
import math
from sklearn.preprocessing import MinMaxScaler
from torch.nn import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import read_data
Data = read_data.Read_data()
train_data, x_test, train_targets, y_test = Data.Transformer_data_286(Normalization=True, zero=True, data2tensor=True)

# 定义自定义数据集类
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


# 定义Focal Loss损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


epochs = 180
# 初始化模型,优化器和损失函数
model = Transformer(
    d_model=64,
    num_encoder_layers=2,
    num_decoder_layers=0,
    nhead=8,
    dropout=0.5,
    batch_first=True,
)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(reduction="none")  # 交叉熵损失函数
# criterion = FocalLoss(gamma=2, alpha=1)
train_dataset = CustomDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 创建数据集和数据加载器
lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    anneal_strategy="linear",
)  # 定义学习率策略

model.to(device)

train_losses = []
train_accs = []
attention_weights = []
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data1, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度重置
        data1 = data1.to(torch.float32)  # 转换为浮点数tensor
        target = target.squeeze()  # 压缩目标值到1D
        output, attention_weights = model(data1, target)  # 计算输出和损失
        loss = criterion(output, target)
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        lr_scheduler.step()  # 根据学习率策略更新学习率
        running_loss += loss.mean().item()  # 记录损失和准确率
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        attention_weights.append(attention_weights)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # 打印周期统计信息
    print(
        f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}, Current lr: {lr_scheduler.get_last_lr()[0]}"
    )

# 绘制训练曲线
plt.plot(train_losses, label="Training Loss")
plt.plot(train_accs, label="Training Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
#%%
model.eval()  # 设置模型为评估模式以禁用dropout
with torch.no_grad():
    output, _ = model(x_test)
    pred = output.argmax(dim=1)  # 获取每个样本的预测类
    accuracy = (pred == y_test).sum().item() / len(y_test)  # 计算模型在测试集上的准确率
    print(f"Accuracy: {accuracy}")
# model.plot_attention_weights([w.cpu().numpy() for w in attention_weights])

#%%
# 预测x_test[sample]的类别
sample = 115
model.eval()  # 设置模型为评估模式以禁用dropout
with torch.no_grad():
    output, attention_weights = model(x_test[sample].unsqueeze(0))
    pred = output.argmax(dim=1)
    print(f"pred: {pred}")
    print(f"y_test[sample]: {y_test[sample]}")
# model.plot_attention_weights([w.cpu().numpy() for w in attention_weights])

#%%
# # 绘制混淆矩阵
y_pred = pred.cpu().numpy()
y_true = y_test.cpu().numpy()
# cm = confusion_matrix(y_true, y_pred)
# sns.heatmap(cm, annot=True, cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.show()
# print(confusion_matrix(y_true,y_pred))
# print(classification_report(y_true, y_pred, target_names=['AWNP', 'AWP', 'DWNP', 'DWP']))
# %%
from libraries.lime import lime_tabular

model = model.to("cpu")
# Create explainer
num_features = 10
class_names = ["AWNP", "AWP", "DWNP", "DWP"]
feature_names = data.columns.values.tolist()
# categorical_features = [231, 265, 266, 267, 280]
x_train = train_data.cpu().numpy()

explainer = lime_tabular.LimeTabularExplainer(
    x_train,
    discretize_continuous=True,
    discretizer="quartile",
    kernel_width=None,
    verbose=True,
    feature_names=feature_names,
    mode="classification",
    class_names=class_names,
    training_labels=y_true,
    feature_selection="lasso_path",
    categorical_features=None,
)

# Explain instance
x_test_np = x_test.cpu().numpy()
exp = explainer.explain_instance(
    x_test_np[sample],
    model.predict_proba,
    num_features=num_features,
    top_labels=1,
    num_samples=5000,
    distance_metric="euclidean",
    model_regressor=None,
)
# Show explanation
exp.show_in_notebook(show_table=True, show_all=False)
# %%
feature_names = data.columns.values.tolist()

avg_attn = []
for attn in attention_weights:
    # avg_attn.append(torch.sum(attn, dim=0))
    avg_attn.append(attn)

for layer_avg_attn in avg_attn:
    plt.plot(feature_names, layer_avg_attn.cpu().numpy())

    # 找出每层的最大值的前10个的索引
    topk = torch.topk(layer_avg_attn, k=10, dim=0)
    print(topk.indices)
    # 打印索引对应的特征名
    print([feature_names[i] for i in topk.indices])
plt.show()

# %%
