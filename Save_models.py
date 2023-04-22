#%%
from models import models
from read_data import Read_data
from  joblib import dump, load
x_train, x_test, y_train, y_test, feature_names = Read_data.data()

# %%
clf = models(x_train, y_train, 2)
firstlayer_models = clf.Stacking(x_test, return_firstlayer_models=True) #一级模型
xtrain, xtest, secondlayer_model = clf.Stacking(x_test, return_first_labels=True) #二级模型
# %%
#将训练好的分类器保存到文件中
j = 0
for model in firstlayer_models:
    dump(model, 'jobmodels\\the{}th_firstlayer_clf.joblib'.format(j))
    test_predict = clf.Stacking(x_test) #二级模型输出结果标签
    proba = clf.proba_value(x_test) #二级模型输出概率
    j += 1

dump(secondlayer_model, 'jobmodels\\secondlayer_clf.joblib')
# %%
