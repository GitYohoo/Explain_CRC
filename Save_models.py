#%%
from models import models
from read_data import Read_data
from  joblib import dump, load
import pandas as pd
import numpy as np
x_train, x_test, y_train, y_test, feature_names = Read_data.data()

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
class Stacking_model():
    """
        将整个stacking模型封装起来
    """
    def __init__(self, firstlayer_models, secondlayer_model):
        self.firstlayer_models = firstlayer_models
        self.secondlayer_model = secondlayer_model

    def predict(self, xtest):
        labels = []
        for model in self.firstlayer_models:
            labels.append(model.predict(xtest))
        labels = np.array(labels).T
        return self.secondlayer_model.predict(labels)

wm = Stacking_model(firstlayer_models, secondlayer_model)
# mm.predict(xtest=x_test[1].reshape(1, -1))
dump(wm, f'jobmodels\\whole_stacking_clf.joblib')
#%%
# stackingmodel = load(f'jobmodels\\whole_stacking_clf.joblib')
# stackingmodel.predict(xtest=x_test[1].reshape(1, -1))
# %%
#将xtrain和xtest保存到csv文件中
xtrain = pd.DataFrame(xtrain)
xtest = pd.DataFrame(xtest)
xtrain.to_csv('data/xtrain.csv', index=False)
xtest.to_csv('data/xtest.csv', index=False)