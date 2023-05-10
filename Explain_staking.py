#%%
from libraries.lime import lime_tabular
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
from read_data import Read_data
from models import models
from  joblib import dump, load
from sklearn.tree import export_graphviz
from graphviz import Source
import libraries.shap as shap


x_train, x_test, y_train, y_test, feature_names1 = Read_data.data()
num_features = 10   #取前10个最重要的特征
class_names=['AWNP','AWP','DWNP','DWP'] #类别名称
feature_names = ['SVM', 'LR', 'XGB', 'RF'] #特征名称
clf = models(x_train, y_train, 2)
train, test, secondlayer_model = clf.Stacking(x_test, return_first_labels=True)

i = 48 # 选择第i个样本进行解释
truelabel = y_test[i]

class Explain_stacking(object):
    def __init__(self): 
        pass 
    
    def explain_secondlayer(self):
        #第二层解释        
        
        test_predict = secondlayer_model.predict(test)
        test_acc = accuracy_score(y_test, test_predict)
        print("这是第2个分类器")
        print("测试集准确率: {0:.3f}".format(test_acc))
        print(confusion_matrix(y_test,test_predict))
        print(classification_report(y_test,test_predict))
        #创建解释器
        explainer = lime_tabular.LimeTabularExplainer(train, discretize_continuous=True,    #true是选择分位
                                                        discretizer='quartile',
                                                        kernel_width=None, verbose=True, feature_names=feature_names,
                                                        mode='classification', class_names=class_names,
                                                        training_labels=y_train,
                                                        feature_selection='lasso_path')  
                                                        # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation'
                                                        # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
        print('开始第二层解释....')       
        print("该样本的真实标签为", truelabel)
        predlabel = secondlayer_model.predict(test[i].reshape(1, -1))
        print("该样本的预测标签为", predlabel)
        self.exp = explainer.explain_instance(test[i],
                                        secondlayer_model.predict_proba,num_features=num_features,
                                        top_labels=1, model_regressor=None, num_samples=10000) #model_regressor:简单模型

        exp_picture = self.exp.show_in_notebook(show_table=True, show_all=False)


    def explain_firstlayer(self):
        #第一层解释
        print('开始第一层解释....')
        local_exp_values = self.exp.local_exp[truelabel]
        #取出 local_exp_values中的第一列
        sortted_index = [i[0] for i in local_exp_values]
        #取出local_exp_values中的第二列
        sortted_contribution = [j[1] for j in local_exp_values]
        #将test的值按照sortted_index的顺序进行排序
        sortted_value = test[i][sortted_index]
        #将sortted_index中的特征名字提取出来
        sortted_models = [feature_names[i] for i in sortted_index]

        x = 0
        the_firstlayer_model_to_explian = []
        for value in sortted_value:
            if value == truelabel and sortted_contribution[x] > 0:
                print(sortted_models[x], "排名第", x + 1)
                the_firstlayer_model_to_explian.append(sortted_models[x])
            
            if value == truelabel and sortted_contribution[x] < 0:
                print("虽然", sortted_models[x], "与预测相同但是它的贡献值为负")

            if the_firstlayer_model_to_explian is None:
                print("没有模型预测与真实标签相同，拒绝解释")
                #抛出错误并退出
                raise SystemExit
            x = x + 1

        explainer = lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=True,    #true是选择分位
                                                        discretizer='quartile',
                                                        kernel_width=None, verbose=True, feature_names=feature_names1,
                                                        mode='classification', class_names=class_names,
                                                        training_labels=y_train,
                                                        feature_selection='lasso_path')  
                                                        # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation'
                                                        # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
        fitted_firstlayer_models = clf.Stacking(x_test, return_firstlayer_models=True)
        for firstlayer_model in the_firstlayer_model_to_explian:
            if firstlayer_model == 'SVM':
                firstlayer_model = fitted_firstlayer_models[0]
            if firstlayer_model == 'LR':
                firstlayer_model = fitted_firstlayer_models[1]
            if firstlayer_model == 'XGB':
                firstlayer_model = fitted_firstlayer_models[2]
            if firstlayer_model == 'RF':
                firstlayer_model = fitted_firstlayer_models[3] 
            
            print("正在解释的是", firstlayer_model)
                                                        
            first_exp =  explainer.explain_instance(x_test[i],
                                            firstlayer_model.predict_proba,num_features=num_features,
                                            top_labels=1, model_regressor=None, num_samples=10000) #model_regressor:简单模型
            first_exp_picture = first_exp.show_in_notebook(show_table=True, show_all=False)
        
        
    def explain_whole(self):
        #整体解释
        clf = load('jobmodels\\the2th_clf.joblib')
        train, test, model = clf.Stacking(x_test, return_first_labels=True)

        explainer = lime_tabular.LimeTabularExplainer(x_train, discretize_continuous=True,    #true是选择分位
                                                        discretizer='quartile',
                                                        kernel_width=None, verbose=True, feature_names=feature_names1,
                                                        mode='classification', class_names=class_names,
                                                        training_labels=y_train,
                                                        feature_selection='lasso_path')  
                                                        # 可选discretizer='quartile' 'decile' 'entropy', 'KernalDensityEstimation'
                                                        # 可选feature_selection='highest_weights' 'lasso_path' 'forward_selection'
        print('开始整体解释....')
        exp= explainer.explain_instance(x_test[i],
                                        clf.proba_value,num_features=num_features,
                                        top_labels=1, model_regressor=None, num_samples=10000) #model_regressor:简单模型

        exp_picture = exp.show_in_notebook(show_table=True, show_all=False)


    def explain_DT(self):
    #决策树决策过程
        dot_data = export_graphviz(secondlayer_model, out_file=None, feature_names=feature_names, 
                                    filled=True, rounded=True, special_characters=True,
                                    precision=2, class_names=class_names, impurity=False, 
                                    node_ids=True, proportion=False, label='none', 
                                    leaves_parallel=False, rotate=False)
        graph = Source(dot_data)
        # graph.view()
        path = secondlayer_model.decision_path(test[i].reshape(1, -1))
        # 获取样本在决策树上的节点列表
        nodes = path.indices.tolist()
        # 将节点列表转换为字符串格式
        node_list = ' -> '.join(map(str, nodes))

        # 在图像中添加文本标签以显示该样本的决策路径
        graph.render(view=False)
        graph.render(filename='DT', directory='./images/', format='png')
        graph.render('DT', view=False)

        # 打印该样本的决策路径和分类结果
        print('样本{}的决策路径：{}'.format(i, node_list))
        print('样本{}的分类结果：{}'.format(i, path))

    def explain_shap(self):
        shap.initjs()
        # explain the model's predictions using SHAP values
        explainer = shap.Explainer(secondlayer_model.predict,test, feature_names=feature_names) 
        shap_values = explainer(test)
        # plot the first prediction's explanation
        # for i in range(len(shap_values)):
        #     shap.plots.waterfall(shap_values[i])
        # shap.plots.force(shap_values)
        shap.plots.waterfall(shap_values[i])

# %%

if __name__ == '__main__':
    explain = Explain_stacking()
    explain.explain_secondlayer()
    explain.explain_firstlayer()
    explain.explain_whole()
    explain.explain_DT()
    explain.explain_shap()

# %%
