#%%
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from libraries.tab_transformer_pytorch.ft_transformer import FTTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import read_data
Data = read_data.Read_data()
train_data, x_test, train_targets, y_test = Data.Transformer_data_286(Normalization=False, zero=False, data2tensor=True)
train_targets = torch.zeros(train_targets.cpu().size(0), 4).scatter_(1, train_targets.cpu().unsqueeze(1).long(), 1).to(device) 

#%%
from torch.optim.lr_scheduler import OneCycleLR

model = FTTransformer(
    categories=[2, 2, 2, 2, 2],
    num_continuous=281,
    dim=128,
    dim_out=4,
    depth=2,
    heads=5,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

x_categ = train_data[:, [231, 265, 266, 267, 280]].long()
x_cont = train_data[:, [i for i in range(286) if i not in [231, 265, 266, 267, 280]]].float()

x_categ = torch.tensor(x_categ, dtype=torch.long, device=device)
x_cont = torch.tensor(x_cont, dtype=torch.float32, device=device)

train_out = []
total_epochs = 37
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_data), epochs=total_epochs)

for epoch in range(total_epochs):
    logits = model(x_categ, x_cont)
    loss = criterion(logits, train_targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    scheduler.step()  # 更新学习率
    
    if epoch % 1 == 0:
        with torch.no_grad():
            train_preds = torch.zeros(logits.cpu().size(0), 4).scatter_(1, logits.cpu().argmax(dim=1).view(-1, 1), 1).to(device)
            train_out.append(logits)
            
        train_acc = (train_preds == train_targets).float().mean()
        print(f'Epoch {epoch+1}: Loss {loss.item():.4f}   Train accuracy: {train_acc:.4f}')
#%%
with torch.no_grad():
    logits = model(x_categ, x_cont)
    predictions = torch.round(torch.sigmoid(logits)).long()

# Evaluate accuracy
acc = (predictions == y_test.long()).float().mean()
print(f'Accuracy: {acc.item()}')

# %%
