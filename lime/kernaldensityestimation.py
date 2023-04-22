#%%
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import cv2

class KernalDensityEstimation():
    """
        核密度聚类
    """
    def __init__(self, features, maxpiecewise=5):

        self.features = features 
        self.maxpicewise = maxpiecewise
        # self.plot_x = np.linspace(-5, 4, features.shape[0])[:, np.newaxis]
        
            

   # 定义一个函数来计算密度估计
    def compute_density_estimation(self, a, bandwidth, plot_x):
        # 使用高斯核
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(a)
        log_dens = kde.score_samples(plot_x)
        return log_dens

    # 定义一个函数来求局部极值
    def compute_local_extrema(self, log_dens): 
        mi, ma = argrelextrema(log_dens, np.less)[0], argrelextrema(log_dens , np.greater)[0]
        return mi, ma

    def statistical(self):
        
        amplitude = []
        quantile = []
        # 使用优化后的代码绘图
        for i in range(self.features.shape[1]):
            bindwidth = 0.4
            a = np.array(self.features)[:, i].reshape(-1, 1) 
            begin = np.min(a)
            end = np.max(a)
            plot_x = np.linspace(begin-2, end+2, self.features.shape[0])[:, np.newaxis]
            log_dens = self.compute_density_estimation(a, bindwidth, plot_x)
            mi, ma = self.compute_local_extrema(log_dens)
            while len(mi) == 0:
                bindwidth =  bindwidth/2 
                log_dens = self.compute_density_estimation(a, bindwidth, plot_x)
                mi, ma = self.compute_local_extrema(log_dens)
            while len(mi) > self.maxpicewise:
                bindwidth =  bindwidth*1.2 
                log_dens = self.compute_density_estimation(a, bindwidth, plot_x)
                mi, ma = self.compute_local_extrema(log_dens)
            # plt.figure(),
            # plt.fill(plot_x, a, fc='#AAAAFF', label='true_density')
            # plt.figure(),
            # plt.plot(plot_x, np.exp(log_dens), 'r', label='estimated_density')
            # for _ in range(a.shape[0]):
            #     plt.plot(a[:, 0], np.zeros(a.shape[0])-0.01, 'g*') 
            # plt.legend()
            # plt.plot(plot_x[ma], np.exp(log_dens)[ma], 'bo',
            #         plot_x[mi], np.exp(log_dens)[mi], 'ro')
            # plt.show()
            print("这是第",i,"个特征")
            print("Minima:", plot_x[mi])
            print("Maxima:", plot_x[ma])
            print("数据类型",type(plot_x[mi]))
            amplitude.append(plot_x[mi])

        for i in amplitude:
            qts = np.array(i, dtype=float)
            if len(qts)==1:
                qts= np.append(qts, qts)
            
            qts =np.squeeze(qts)
            quantile.append(qts)
            quantile = [np.unique(x) for x in quantile]
            
        return quantile
#%%
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import pandas as pd

df = pd.read_excel(r'X:\结直肠癌研究\数据\samplecolondata.xlsx',header=None,index_col=0)#low_memory=False
data=df.T
data = data.dropna(axis=1, how='any')
data=data.drop(['P_ID'],axis=1)
data['label']=pd.to_numeric(data['label'],errors='coerce')
data=pd.DataFrame(data,dtype=np.float64)

#feature=data.drop(['label', 'AGE', 'WEIGHT'],axis=1)
feature=data.drop(['label'],axis=1)

label=data['label'].values
zscore = preprocessing.StandardScaler()
# 标准化处理
x = pd.DataFrame(zscore.fit_transform(feature))
#feature=data.drop(['label', 'AGE', 'WEIGHT'],axis=1)
feature=data.drop(['label'],axis=1)

label=data['label'].values
KDE = KernalDensityEstimation(features=feature)
quantile = KDE.statistical()

# %%
