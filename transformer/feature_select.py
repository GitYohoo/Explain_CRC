import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# 读取数据
data = pd.read_csv('data.csv')
numeric_cols = ['col1', 'col2', 'col3']     # 数值特征列名
categorical_cols = ['col4', 'col5']         # 分类特征列名 
target = 'target'                          # 目标列名
# 定义Transformer
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())  # 标准化
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # one-hot编码
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# 定义选择特征的步骤
selector = SelectKBest(score_func=mutual_info_classif, k=5)

# 定义完整的Pipeline 
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("selector", selector)
])

# 训练Pipeline并选择特征
pipeline.fit(data, target)
selected_features = selector.get_support(indices=True)

# 打印选择的特征
print(selected_features)