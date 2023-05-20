#%%
from models import models
from read_data import Read_data
from  joblib import dump, load
import pandas as pd
import numpy as np
from read_data import Read_data

data_reader = Read_data()
x_train, x_test, y_train, y_test, feature_names = data_reader.data()

# %%
clf = models(x_train, y_train, 2)
firstlayer_models = clf.Stacking(x_test, return_firstlayer_models=True) #一级模型
xtrain, xtest, secondlayer_model = clf.Stacking(x_test, return_first_labels=True) #二级模型
# %%
#将训练好的分类器保存到文件中
for j, model in enumerate(firstlayer_models):
    dump(model, f'jobmodels\\the{j}th_firstlayer_clf.joblib')
dump(secondlayer_model, 'jobmodels\\secondlayer_clf.joblib')
# %%
from Stacking_model import Stacking_model

wm = Stacking_model(firstlayer_models, secondlayer_model)
# mm.predict(xtest=x_test[1].reshape(1, -1))
dump(wm, f'jobmodels\\whole_stacking_clf.joblib')
#%%
stackingmodel = load(f'jobmodels\\whole_stacking_clf.joblib')
stackingmodel.predict(xtest=x_test[1].reshape(1, -1))
# %%
#将xtrain和xtest保存到csv文件中
xtrain = pd.DataFrame(xtrain)
xtest = pd.DataFrame(xtest)
xtrain.to_csv('data/xtrain.csv', index=False)
xtest.to_csv('data/xtest.csv', index=False)