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
#%%
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
# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                       dim_feedforward=hidden_dim, dropout=dropout)# 初始化一个Transformer编码器层对象,该层由多头自注意力和前馈神经网络组成
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) # 初始化一个由多个Transformer编码器层组成的Transformer编码器对象       
        self.dropout = nn.Dropout(dropout) # 添加一个Dropout层,用于防止过拟合 
        self.fc = nn.Linear(input_dim, output_dim) # 初始化一个全连接层,用于将Transformer编码器的输出映射到所需的输出维度

    def forward(self, x):
        x = self.transformer_encoder(x)# 传入Transformer编码器
        x = self.dropout(x) # 传入Dropout层      
        x = self.fc(x)# 传入全连接层  
        attention_weights = self.calculate_attention_weights()# 计算注意力权重
        return x, attention_weights

    def calculate_attention_weights(self):
        # 从自注意力层检索注意力权重
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            attn_weights = layer.self_attn.in_proj_weight.data.detach()
            attention_weights.append(attn_weights)
        return attention_weights

    def predict_proba(self, x):        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)#如果x是numpy数组,则转换为tensor
        x = self.forward(x)[0]
        x = F.softmax(abs(x), dim=1)# 计算模型输出的概率分布
        x = x.detach().numpy()# 将张量转换为numpy数组
        return x

    def plot_attention_weights(self):
        count = 0 #层数
        fig, axes = plt.subplots(ncols=len(attention_weights), figsize=(20,10))
        for ax, attn_weights in zip(axes, attention_weights):
            im = ax.imshow(attn_weights)
            ax.set_title('Attention Weights Layer {}'.format(count))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            count += 1
    plt.show()

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
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

epochs=100
# 初始化模型,优化器和损失函数 
model = Transformer(input_dim=train_data.shape[1], output_dim=4, hidden_dim=256, num_layers=3, 
                                                    num_heads=10, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(reduction='none') # 交叉熵损失函数
# criterion = FocalLoss(gamma=2, alpha=0.25)
train_dataset = CustomDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 创建数据集和数据加载器
lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=0.001, 
                                            steps_per_epoch=len(train_loader),
                                            epochs=epochs,
                                            anneal_strategy='linear')# 定义学习率策略

train_losses = []
train_accs = []
attention_weights = [] 
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data1, target) in enumerate(train_loader):
        optimizer.zero_grad() # 梯度重置 
        data1 = data1.to(torch.float32)   # 转换为浮点数tensor 
        target = target.squeeze()  # 压缩目标值到1D 
        output, attention_weights = model(data1)# 计算输出和损失
        loss = criterion(output, target)  
        loss.backward(torch.ones_like(loss))  
        optimizer.step()  
        lr_scheduler.step()# 根据学习率策略更新学习率
        running_loss += loss.mean().item()# 记录损失和准确率
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()  
        attention_weights.append(attention_weights)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # 打印周期统计信息
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}, Current lr: {lr_scheduler.get_last_lr()[0]}")

# 绘制训练曲线 
plt.plot(train_losses, label='Training Loss')
plt.plot(train_accs, label='Training Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


model.eval()  # 设置模型为评估模式以禁用dropout
with torch.no_grad():
    output, attention_weights = model(x_test)
    pred = output.argmax(dim=1)  # 获取每个样本的预测类
    accuracy = (pred == y_test).sum().item() / len(y_test)  # 计算模型在测试集上的准确率
    print(f"Accuracy: {accuracy}")
model.plot_attention_weights()
# # 绘制混淆矩阵  
y_pred = pred.numpy()
y_true = y_test.numpy()  
# cm = confusion_matrix(y_true, y_pred)
# sns.heatmap(cm, annot=True, cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.show()
# print(classification_report(y_true, y_pred, target_names=['AWNP', 'AWP', 'DWNP', 'DWP']))
# %%
from libraries.lime import lime_tabular 

#创建解释器
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
#取data的第一行作为特征名称
feature_names = data.columns.values.tolist()
categorical_features = [231, 265, 266, 267, 280]
x_train = train_data.numpy()
explainer = lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=True,    #true是选择分位
                                                discretizer='quartile',
                                                kernel_width=None, verbose=True, feature_names=feature_names,
                                                mode='classification', class_names=class_names,
                                                training_labels=y_true,
                                                feature_selection='lasso_path',
                                                categorical_features=categorical_features)  
                                                # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation' 
                                                # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
# %%

# 将样本tensor转换为numpy数组,传入LIME
x_test_np = x_test.numpy()  
exp = explainer.explain_instance(x_test_np[8], model.predict_proba, num_features=num_features, top_labels=1,
                                 num_samples=10000, distance_metric='euclidean', model_regressor=None)
exp.show_in_notebook(show_table=True, show_all=False)  # 在notebook中显示解释结果
1#%%
# %%
