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
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
import math
import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据的同时去掉第一行,第一行是特征名称
rawdata = pd.read_csv("..\\data\\new_data.csv", header=0)
# 取出第一列作为标签
targets = rawdata.iloc[:, -1]
# 取出后面的列作为特征
data = rawdata.iloc[:, 0:-1]

scaler = MinMaxScaler() # # 创建MinMaxScaler对象
normalized_data = scaler.fit_transform(data)# # 对data进行归一化
data = pd.DataFrame(normalized_data, columns=data.columns)# # 将归一化后的数据重新转换为DataFrame

# 为了提高多头注意力机制头的个数，补充四列0到data的后面
# 创建一个包含4列零值的数组
zeros = np.zeros((data.shape[0], 4))
# 将零值数组转换为DataFrame
zeros_df = pd.DataFrame(zeros, columns=["Zero1", "Zero2", "Zero3", "Zero4"])
# 将零值DataFrame与原始数据data进行水平拼接
data = pd.concat([data, zeros_df], axis=1)
#%%
#统计数字1占targets的比例
print("数字1占比：", targets.value_counts()[1] / targets.shape[0])
#%%
# 划分训练集和测试集,训练集占80%,测试集占20%
train_data, x_test, train_targets, y_test = train_test_split(
    data, targets, test_size=0.2
)

# 标签值减1
train_targets = train_targets - 1
y_test = y_test - 1
# 转化为tensor格式
train_data = torch.tensor(train_data.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
train_targets = torch.tensor(train_targets.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_data = train_data.to(device)
x_test = x_test.to(device)
train_targets = train_targets.to(device)
y_test = y_test.to(device)

# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        )  # 初始化一个Transformer编码器层对象,该层由多头自注意力和前馈神经网络组成
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )  # 初始化一个由多个Transformer编码器层组成的Transformer编码器对象
        self.dropout = nn.Dropout(dropout)  # 添加一个Dropout层,用于防止过拟合
        self.fc = nn.Linear(
            input_dim, output_dim
        )  # 初始化一个全连接层,用于将Transformer编码器的输出映射到所需的输出维度
       
    def forward(self, x):
        x = self.transformer_encoder(x)  # 传入Transformer编码器
        x = self.dropout(x)  # 传入Dropout层
        attention_weights = self.calculate_attention_weights(x)  # 计算注意力权重
        x = self.fc(x)  # 传入全连接层
        return x

    def calculate_attention_weights(self, x):
        attn_weights = []
        for layer in self.transformer_encoder.layers:
            weight = layer.self_attn.in_proj_weight
            dim = layer.self_attn.embed_dim
            # 获取q,k,v的权重
            q_weight = weight[:dim, :]
            k_weight = weight[dim:2*dim, :] 
            v_weight = weight[2*dim:, :]  
            Q = x @ q_weight / math.sqrt(dim)
            K = x @ k_weight / math.sqrt(dim)
            V = x @ v_weight
            attn_output_weights = F.softmax(Q @ K.T / math.sqrt(dim), dim=1) @ V
            attn_weights.append(attn_output_weights)

        return attn_weights

    def predict_proba(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)  # 如果x是numpy数组,则转换为tensor
        x = self.forward(x)
        x = F.softmax(abs(x), dim=1)  # 计算模型输出的概率分布
        x = x.detach().numpy()  # 将张量转换为numpy数组
        return x

    def plot_attention_weights(self, attention_weights):
        count = 0
        #将attention_weights复制20行
        # attention_weights = np.repeat(attention_weights, 20, axis=0)
        fig, axes = plt.subplots(ncols=len(attention_weights), figsize=(20, 10))
        for ax, attn_weights in zip(axes, attention_weights):
            im = ax.imshow(attn_weights, origin="upper")
            ax.set_title("Attention Weights Layer {}".format(count))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.invert_yaxis()  # 反向显示y轴
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


epochs = 150
# 初始化模型,优化器和损失函数
model = Transformer(
    input_dim=train_data.shape[1],
    output_dim=4,
    hidden_dim=128,
    num_layers=2,
    num_heads=10,
    dropout=0.5,
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
        output = model(data1)  # 计算输出和损失
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
    output = model(x_test)
    pred = output.argmax(dim=1)  # 获取每个样本的预测类
    accuracy = (pred == y_test).sum().item() / len(y_test)  # 计算模型在测试集上的准确率
    print(f"Accuracy: {accuracy}")
# model.plot_attention_weights([w.cpu().numpy() for w in _])

#%%
# 预测x_test[sample]的类别
sample = 115
# model.eval()  # 设置模型为评估模式以禁用dropout
# with torch.no_grad():
#     output = model(x_test[sample].unsqueeze(0))
#     pred = output.argmax(dim=1)
#     print(f"pred: {pred}")
#     print(f"y_test[sample]: {y_test[sample]}")
# model.plot_attention_weights([w.cpu().numpy().T for w in attention_weights])

#%%
# # 绘制混淆矩阵
y_pred = pred.cpu().numpy()
y_true = y_test.cpu().numpy()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')

print(confusion_matrix(y_true,y_pred))
print(classification_report(y_true, y_pred, target_names=['AWNP', 'AWP', 'DWNP', 'DWP']))

#计算F1值
f1 = f1_score(y_true, y_pred, average='weighted')
print('F1 score:', f1)
#计算召回率
recall = recall_score(y_true, y_pred, average='weighted')
print('Recall:', recall)
#计算精确率
precision = precision_score(y_true, y_pred, average='weighted')
print('Precision:', precision)

#%%
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# #计算ROC曲线的值
# fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# #计算AUC的值
# roc_auc = auc(fpr, tpr)
# #绘制ROC曲线
# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# #设置x轴和y轴的取值范围
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# #设置x轴和y轴的标签
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# #设置标题
# plt.title('Receiver operating characteristic example')
# #显示图例
# plt.legend(loc="lower right")
# plt.show()

# %%
import sys
sys.path.append('..')
from libraries.lime import lime_tabular
sample = 114
model = model.to("cpu")
x_test = x_test.to("cpu")
# Create explainer
num_features = 10
class_names = ["AWNP", "AWP", "DWNP", "DWP"]
feature_names = data.columns.values.tolist()
categorical_features = [231, 265, 266, 267, 280]
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
    categorical_features=categorical_features,
)

# Explain instance
x_test_np = x_test.numpy()
exp = explainer.explain_instance(
    x_test_np[sample],
    model.predict_proba,
    num_features=num_features,
    top_labels=1,
    num_samples=10000,
    distance_metric="euclidean",
    model_regressor=None,
)
# Show explanation
# exp.show_in_notebook(show_table=True, show_all=False)
# %%
# feature_names = data.columns.values.tolist()

# avg_attn = []
# for attn in attention_weights:
#     # avg_attn.append(torch.sum(attn, dim=0))
#     avg_attn.append(attn.reshape(-1))

# for layer_avg_attn in avg_attn:
#     #画柱状图
#     layer_avg_attn = abs(layer_avg_attn) # 取绝对值
#     plt.bar(feature_names, layer_avg_attn.cpu().numpy())

#     # plt.plot(feature_names, layer_avg_attn.cpu().numpy())
    
#     # 找出每层的最大值的前10个的索引
#     topk = torch.topk(layer_avg_attn, k=10, dim=0)
#     print(topk.indices)
#     # 打印索引对应的特征名
#     print([feature_names[i] for i in topk.indices])
# plt.show()

# # %%用shap对模型进行
# import sys
# sys.path.append("..")
# import libraries.shap as shap
# shap.initjs()
# # select a set of background examples to take an expectation over
# background = x_train[np.random.choice(x_train.shape[0], 400, replace=False)]
# background = torch.from_numpy(background)

# exp = shap.DeepExplainer(model, background)
# shap_values = exp.shap_values(x_test[sample].unsqueeze(0))
# # shap.plots.waterfall(shap_values[0])

# # %%
# shap.summary_plot(shap_values, x_test[sample].unsqueeze(0), feature_names=feature_names, class_names=class_names)

# # %%
# from libraries.shap.plots import _waterfall
# a = exp.expected_value
# x = exp.expected_value.argmax()
# _waterfall.waterfall_legacy(x, shap_values[x].squeeze(0), feature_names=feature_names, max_display=10, show=True)
# %%统计结果
import sys
sys.path.append('..')
from Save_exp import save_exp
print('开始解释....')
# sample = [46]
sample = list(range(len(x_test_np)))
for i in sample:
    output = []
    # truelabel = target[i]
    # print("该样本的真实标签为", truelabel)
    # output.append("该样本的真实标签为"+str(truelabel))
    # predlabel = wsm.predict(x_train[i].reshape(1, -1))
    # predlabel = int(predlabel)
    # print("该样本的预测标签为", predlabel)
    # output.append("该样本的预测标签为"+str(predlabel))
    # if predlabel != truelabel:
    #     continue
    exp = explainer.explain_instance(
        x_test_np[i],
        model.predict_proba,
        num_features=num_features,
        top_labels=4,
        num_samples=5000,
        distance_metric="euclidean",
        model_regressor=None,
    )

    # exp.show_in_notebook(show_table=True, show_all=False)

    for label in range(4): 
        csv_path = r'D:\Desktop\Explain_CEC_Recording\exp_result\label_{}\\test_{}.csv'.format(label+1, i)
        html_path = r'D:\Desktop\Explain_CEC_Recording\exp_result\html\\test_{}.html'.format(i)
        save_exp(exp, i, output, label, csv_path, html_path)
# %%
import sys
sys.path.append('..')
from Count_exp import count_exp
org_path = r'D:\Desktop\Explain_CEC_Recording\exp_result\label_{}\\'
count_path = r'D:\Desktop\Explain_CEC_Recording\exp_result\\'
count_exp(org_path, count_path)
# %%
from Count_exp import sum_all
sum_path = r'D:\Desktop\Explain_CEC_Recording\exp_result\\'
sum_all(sum_path)

from Count_exp import quotation_marks
quotation_marks(sum_path)
# %%
