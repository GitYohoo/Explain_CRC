#%%
import numpy as np
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
    
    def proba_value(self, xtest):
        proba = []
        for model in self.firstlayer_models:
            proba.append(model.predict(xtest))
        proba = np.array(proba).T
        return self.secondlayer_model.predict_proba(proba)