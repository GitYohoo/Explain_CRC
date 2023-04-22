import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

random_state = 42

class Create_ensemble(object):
    def __init__(self, n_splits, base_models):
        self.n_splits = n_splits
        self.base_models = base_models

    def predict(self, X, y, T, return_firstlayer_models=False):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        # 使用StratifiedKFold进行交叉验证
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                     random_state=random_state).split(X, y))
        train_pred = np.zeros((X.shape[0], len(self.base_models)))
        test_pred_stack = np.zeros((T.shape[0], len(self.base_models)))
        firstlayer_models = []
        for clf in self.base_models:
            test_pred = np.zeros((T.shape[0], self.n_splits))
            # 使用zip函数同时迭代folds和self.base_models
            for (train_idx, valid_idx), clf in zip(folds, self.base_models):
                X_train = X[train_idx]
                Y_train = y[train_idx]
                X_valid = X[valid_idx]
                Y_valid = y[valid_idx]

                clf.fit(X_train, Y_train)
                valid_pred = clf.predict(X_valid)
                train_pred[valid_idx, i] = valid_pred
                test_pred[:, test_col] = clf.predict(T)
                test_col += 1
            firstlayer_models.append(clf)    

            # 使用argmax函数找到test_pred中每行中出现次数最多的元素的索引
            test_precict = np.argmax(np.bincount(test_pred), axis=1)
            test_pred_stack[:,i]=np.array(test_precict)

        if return_firstlayer_models:
            return firstlayer_models
        return train_pred, test_pred_stack, clf
