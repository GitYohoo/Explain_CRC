"""
Discretizers classes, to be used in lime_tabular
"""
import numpy as np
import sklearn
import sklearn.tree
import scipy
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class BaseDiscretizer():
    """
    Abstract class - Build a class that inherits from this class to implement
    a custom discretizer.
    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.
    """

    __metaclass__ = ABCMeta  # abstract class

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None,
                 data_stats=None):
        """Initializer
        Args:
            data: numpy 2d array
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. These features will not be discretized.
                Everything else will be considered continuous, and will be
                discretized.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            data_stats: must have 'means', 'stds', 'mins' and 'maxs', use this
                if you don't want these values to be computed from data
        """
        self.to_discretize = ([x for x in range(data.shape[1])
                               if x not in categorical_features]) #首先x从range(*)取值，判断是否在cate*_*ures中, 是则保留
        self.data_stats = data_stats #None
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}  #标准差
        self.mins = {}  #每一段上的最小值
        self.maxs = {}  #每一段上的最大值
        self.random_state = check_random_state(random_state)

        # To override when implementing a custom binning
        bins = self.bins(data, labels) #分位数
        bins = [np.unique(x) for x in bins] 

        # Read the stats from data_stats if exists
        if data_stats:
            self.means = self.data_stats.get("means")
            self.stds = self.data_stats.get("stds")
            self.mins = self.data_stats.get("mins")
            self.maxs = self.data_stats.get("maxs")

        for feature, qts in zip(self.to_discretize, bins): #qts：bins的每一行
            n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
            boundaries = np.min(data[:, feature]), np.max(data[:, feature])
            name = feature_names[feature]

            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <= %.2f' %
                                           (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))

            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x) #寻找x在qts中的位置
            discretized = self.lambdas[feature](data[:, feature]) #查找每一个样本的该特征属于哪一个段

            # If data stats are provided no need to compute the below set of details
            if data_stats:
                continue

            self.means[feature] = []
            self.stds[feature] = []
            for x in range(n_bins + 1):
                selection = data[discretized == x, feature]
                mean = 0 if len(selection) == 0 else np.mean(selection)
                self.means[feature].append(mean)
                std = 0 if len(selection) == 0 else np.std(selection)
                std += 0.00000000001
                self.stds[feature].append(std)
            self.mins[feature] = [boundaries[0]] + qts.tolist()
            self.maxs[feature] = qts.tolist() + [boundaries[1]]

    @abstractmethod
    def bins(self, data, labels):
        """
        To be overridden
        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """Discretizes the data.
        Args:
            data: numpy 2d or 1d array
        Returns:
            numpy array of same dimension, discretized.
        """
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature])) #把单一样本特征对应到分位区间
            else:
                ret[:, feature] = self.lambdas[feature](
                    ret[:, feature]).astype(int) #把数据对应到第几个分位数
        return ret

    def get_undiscretize_values(self, feature, values): #将 values 中的每一个值转换为原始值  
        mins = np.array(self.mins[feature])[values]
        maxs = np.array(self.maxs[feature])[values]

        means = np.array(self.means[feature])[values]
        stds = np.array(self.stds[feature])[values]
        minz = (mins - means) / stds
        maxz = (maxs - means) / stds
        min_max_unequal = (minz != maxz)

        ret = minz
        ret[np.where(min_max_unequal)] = scipy.stats.truncnorm.rvs(
            minz[min_max_unequal],
            maxz[min_max_unequal],
            loc=means[min_max_unequal],
            scale=stds[min_max_unequal],
            random_state=self.random_state
        )
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means:
            if len(data.shape) == 1:
                ret[feature] = self.get_undiscretize_values(
                    feature, ret[feature].astype(int).reshape(-1, 1)
                )
            else:
                ret[:, feature] = self.get_undiscretize_values(
                    feature, ret[:, feature].astype(int)
                )
        return ret


class StatsDiscretizer(BaseDiscretizer):
    """
        Class to be used to supply the data stats info when discretize_continuous is true
    """
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None,
                 data_stats=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state,
                                 data_stats=data_stats)

    def bins(self, data, labels):
        bins_from_stats = self.data_stats.get("bins")
        bins = []
        if bins_from_stats is not None:
            for feature in self.to_discretize:
                bins_from_stats_feature = bins_from_stats.get(feature)
                if bins_from_stats_feature is not None:
                    qts = np.array(bins_from_stats_feature)
                    bins.append(qts)
        return bins

class QuartileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
            bins.append(qts)
        return bins

class DecileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state) #计算min，max，means，生成“< = >”列表, 生成分位数

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature],
                                         [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            bins.append(qts)
        return bins

class EntropyDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if(labels is None):
            raise ValueError('Labels must be not None when using \
                             EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 8 bins so max_depth=3
            dt = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=3,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins

class DensityDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None, maxpicewise = 5):
        self.maxpicewise = maxpicewise
        BaseDiscretizer.__init__(self, data, categorical_features,
                            feature_names, labels=labels,
                            random_state=random_state)
        
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

    def bins(self, data, labels):
        amplitude = [] #统计极小值点
        quantile = [] #最终的分仓点
        # 使用优化后的代码绘图
        for i in self.to_discretize:
            bindwidth = 0.4
            a = np.array(data)[:, i].reshape(-1, 1) 
            begin = np.min(a)
            end = np.max(a)
            plot_x = np.linspace(begin-2, end+2,data.shape[0])[:, np.newaxis]
            log_dens = self.compute_density_estimation(a, bindwidth, plot_x)
            mi, ma = self.compute_local_extrema(log_dens)
            #将分仓数控制在2~5之间
            while len(mi) == 0:
                bindwidth =  bindwidth/2 
                if bindwidth < 0.0002:
                    print(i)
                log_dens = self.compute_density_estimation(a, bindwidth, plot_x)
                mi, ma = self.compute_local_extrema(log_dens)
            while len(mi) > self.maxpicewise:
                bindwidth =  bindwidth*1.2 
                log_dens = self.compute_density_estimation(a, bindwidth, plot_x)
                mi, ma = self.compute_local_extrema(log_dens)
            # for x in np.exp(log_dens)[mi]:
            #     if x < 0.05:
            #         mi = np.delete(mi, np.where(np.exp(log_dens)[mi] == x)) #删除小于0.05的极小值点
            # plt.figure()
            # plt.fill(plot_x, a, fc='#AAAAFF', label='true_density')
            fig, ax1 = plt.subplots()
            ax1.plot(plot_x, np.exp(log_dens), 'r', label='estimated_density')
            # for _ in range(a.shape[0]):
            #     ax1.plot(a[:, 0], np.zeros(a.shape[0])-0.01, 'g*') 
            ax1.legend()
            ax1.plot(plot_x[ma], np.exp(log_dens)[ma], 'bo',
                    plot_x[mi], np.exp(log_dens)[mi], 'ro')
            # 计算数据的四分位数
            ax2 = ax1.twinx()
            q1, q2, q3 = np.percentile(a, [25, 50, 75])
            # 绘制直方图
            ax2.hist(a, bins=50)
            # 在四分位数处绘制竖线
            ax2.axvline(q1, color='r', linestyle='--', label='Q1')
            ax2.axvline(q2, color='g', linestyle='--', label='Q2')
            ax2.axvline(q3, color='b', linestyle='--', label='Q3')
            # 显示图例
            
            # 显示图表
            plt.show()
            print("这是第",i,"个特征")
            print("Minima:", plot_x[mi])
            print("Maxima:", plot_x[ma])
            print("数据类型",type(plot_x[mi]))
            amplitude.append(plot_x[mi])  #统计极小值点

        for i in amplitude:
            qts = np.array(i, dtype=float)
            if len(qts)==1:
                qts= np.append(qts, qts)
            
            qts =np.squeeze(qts)
            quantile.append(qts)
            quantile = [np.unique(x) for x in quantile] #最终的分仓点
        
        return quantile  
