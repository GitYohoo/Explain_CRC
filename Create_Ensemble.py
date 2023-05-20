import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

random_state = 42
#各分类器进行实例化，注意初始状态要一致
class Create_ensemble(object):  # 集成器
    def __init__(self, n_splits, base_models):  # 初始化
        self.n_splits = n_splits
        self.base_models = base_models

    def predict(self, X, y, T, return_firstlayer_models=False):  # 该部分实现集成交叉验证
        X = np.array(X)  # 传入的是训练集
        y = np.array(y)  # 训练集的标签
        T = np.array(T)  # 测试集

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True,  # 交叉验证
                                     random_state=random_state).split(X, y))
        train_pred = np.zeros((X.shape[0], len(self.base_models)))
        test_pred_stack = np.zeros((T.shape[0], len(self.base_models)))
        firstlayer_models = [] #存储模型
        for i, clf in enumerate(self.base_models):
            # print(clf)
            test_col = 0
            test_pred = np.zeros((T.shape[0], self.n_splits))
            for j, (train_idx, valid_idx) in enumerate(folds):  # 集成交叉验证的实现

                X_train = X[train_idx]  # 训练集
                Y_train = y[train_idx]  # 训练集标签
                X_valid = X[valid_idx]  # 验证集
                Y_valid = y[valid_idx]  # 验证集标签

                clf.fit(X_train, Y_train)  # 训练模型
                valid_pred = clf.predict(X_valid)  # 验证建立的模型
                train_pred[valid_idx, i] = valid_pred
                test_pred[:, test_col] = clf.predict(T)  # 对测试集进行预测是一个m*5的矩
                test_col += 1
            firstlayer_models.append(clf)    

            tpred_weight = pd.DataFrame(test_pred)  # 找出预测标签当中出现次数最多的类别
            final_tpred_weight = tpred_weight.mode(axis=1)
            test_precict = final_tpred_weight[0]
            test_pred_stack[:,i]=np.array(test_precict)

            # pred = np.array(clf.predict(T))
            # result = np.concatenate((test_pred_stack[:, i].reshape(-1, 1), pred.reshape(-1, 1)), axis=1)
            # print("直接预测结果：", result)
            
        
        if return_firstlayer_models:
            return firstlayer_models
        return train_pred, test_pred_stack, clf#返回训练集预测标签，测试集预测标签, 模型
