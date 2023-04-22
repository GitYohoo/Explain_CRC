#%%
import models
from read_data import Read_data

x_train, x_test, y_train, y_test, feature_names = Read_data.data()

# %%
clf = models(x_train, y_train, 2)
