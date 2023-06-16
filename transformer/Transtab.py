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

import read_data
Data = read_data.Read_data()
# train_data, x_test, train_targets, y_test = Data.Transformer_data_486()
train_data, x_test, train_targets, y_test = Data.Transformer_data_286(Normalization=True, zero=True, data2tensor=False)
#%%
train_x = train_data
train_y = pd.Series(train_targets)
trainset = [(train_x, train_y)]

y_test = pd.Series(y_test)
testset = [(x_test, y_test)]
#%%
num_class = 4
num_attention_head = 8
# binary_columns=[231, 265, 266, 267, 280]
# numerical_columns=[i for i in range(286) if i not in binary_columns]
# hidden_dropout_prob=0.5
model = transtab.build_classifier(
    # numerical_columns=numerical_columns,binary_columns=binary_columns,
                                    num_class=num_class, num_attention_head = num_attention_head, 
                                    )
                                  
training_arguments = {
    'num_epoch':25,
    'valset': None,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':'./checkpoint',
    'batch_size': 64,
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
from sklearn.metrics import f1_score, recall_score, precision_score
#计算F1值
f1 = f1_score(y_test, ypred, average='weighted')
print('F1 score:', f1)
#计算召回率
recall = recall_score(y_test, ypred, average='weighted')
print('Recall:', recall)
#计算精确率
precision = precision_score(y_test, ypred, average='weighted')
print('Precision:', precision)

#打印训练集准确率
train_pred = transtab.predict(clf = model, x_test = train_data, y_test=train_targets)
train_pred = np.argmax(train_pred, axis=1)
print("train accuracy: ", np.mean(train_pred == train_targets))
print(classification_report(train_targets, train_pred))



# # %%
# import sys
# sys.path.append("..")
# from libraries.lime import lime_tabular
# sample = 10
# model = model.to("cpu")
# # Create explainer
# num_features = 10
# class_names = ["AWNP", "AWP", "DWNP", "DWP"]
# feature_names = data.columns.values.tolist()
# categorical_features = [231, 265, 266, 267, 280]
# x_train = np.array(train_data)

# class predict_proba():
#     def __init__(self, model, y_test.iloc[sample]):
#         self.model = model
#         self.y_test = y_test

#     def predict_proba(self, x):
#         #转化为dataframe
#         x = pd.DataFrame(x)
#         pred = transtab.predict(clf = self.model, x_test = x, y_test=self.y_test, return_loss=False)
#         return pred
    
# pred = predict_proba(model, y_test[sample])

# explainer = lime_tabular.LimeTabularExplainer(
#     x_train,
#     discretize_continuous=True,
#     discretizer="quartile",
#     kernel_width=None,
#     verbose=True,
#     feature_names=feature_names,
#     mode="classification",
#     class_names=class_names,
#     training_labels=train_y,
#     feature_selection="lasso_path",
#     categorical_features=categorical_features,
# )

# # Explain instance
# x_test_np = np.array(x_test)
# exp = explainer.explain_instance(
#     x_test_np[sample],
#     pred.predict_proba,
#     num_features=num_features,
#     top_labels=1,
#     num_samples=5000,
#     distance_metric="euclidean",
#     model_regressor=None,
# )
# # Show explanation
# exp.show_in_notebook(show_table=True, show_all=False)
# # %%
# pred.predict_proba(x_test_np[sample])