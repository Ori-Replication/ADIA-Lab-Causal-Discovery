import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import multiprocessing

def get_device():
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA 可用，使用 GPU 进行训练。")
    else:
        device = 'cpu'
        print("CUDA 不可用，使用 CPU 进行训练。")
    return device

def get_optimal_num_workers():
    try:
        num_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        num_cores = 1  # 如果无法检测CPU核心数，默认设置为1
    num_workers = max(1, num_cores // 2)
    print(f"检测到的CPU核心数为{num_cores}，设置使用核心数为{num_workers}")
    return num_workers

def clean_feature_names(X):
    # 函数用于清理特征名称
    def clean_name(name):
        # 移除或替换特殊字符
        name = re.sub(r'[^\w\s-]', '_', name)
        # 确保名称不以数字开头
        if name and name[0].isdigit():
            name = 'f_' + name
        return name

    X.columns = [clean_name(col) for col in X.columns]
    return X

# 定义函数以自动检测和处理类别特征
def process_categorical_features(df, max_unique=10):
    """
    自动检测和处理数据框中的类别变量。

    参数：
    - df (pd.DataFrame): 输入的数据框。
    - max_unique (int): 判定为类别变量的最大唯一值数量。

    返回：
    - cat_features (list): 类别特征的列名列表。
    - df (pd.DataFrame): 经过编码后的数据框。
    """
    cat_cols = [col for col in df.columns if df[col].nunique() <= max_unique]
    print(f"识别到的类别特征: {cat_cols}")
    
    return cat_cols, df

device = get_device()
num_workers = get_optimal_num_workers()

# 读取数据
X_y_group_train = pd.read_csv('/hy-tmp/mid_data/X_y_group_train_updated_v12.2_piecewise.csv')

print("添加数值标签 y")
le = LabelEncoder()
X_y_group_train["y"] = le.fit_transform(X_y_group_train["label"])
# 重新排列列
X_y_group_train = X_y_group_train[["dataset", "variable"] + X_y_group_train.columns.drop(["dataset", "variable", "label", "y"]).tolist() + ["label", "y"]]

# 定义要删除的列
blacklist = [
    "ttest(v,X)", 
    "pvalue(ttest(v,X))<=0.05", 
    "ttest(v,Y)", 
    "pvalue(ttest(v,Y))<=0.05", 
    "ttest(X,Y)", 
    "pvalue(ttest(X,Y))<=0.05",
    "square_dimension", 
    "max(PPS(v,others))"
]
columns_to_drop = [col for col in blacklist if col in X_y_group_train.columns]
X_y_group_train = X_y_group_train.drop(columns=columns_to_drop)

# 处理数值列的缺失值
numeric_columns = X_y_group_train.select_dtypes(include=[np.number]).columns
X_y_group_train[numeric_columns] = X_y_group_train[numeric_columns].fillna(X_y_group_train[numeric_columns].mean())

# 清理特征名称
X_y_group_train = clean_feature_names(X_y_group_train)

print("提取 X_train, y_train 和 group")
# 分离数据集ID、特征和标签
group_train = X_y_group_train["dataset"]
X = X_y_group_train.drop(["variable", "dataset", "label", "y"], axis="columns")
y = X_y_group_train["y"]

# 处理类别特征
cat_features, X = process_categorical_features(X)
print(f"类别特征列名 (cat_features): {cat_features}")

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("y_train 唯一值:", np.unique(y_train))
print("y_test 唯一值:", np.unique(y_test))

# 计算类别权重（使用每个类别的逆频率）
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = list(class_weights)  # 转换为列表
print(f"类别权重: {class_weights}")

# 创建类别权重字典
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# 初始化 CatBoostClassifier
clf = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    class_weights=class_weights_dict,
    cat_features=cat_features,
    random_seed=42,
    eval_metric='BalancedAccuracy',
    verbose=100,
    early_stopping_rounds=50
)

# 训练模型
clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# 预测
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# 计算平衡准确率
train_score = balanced_accuracy_score(y_train, y_train_pred)
test_score = balanced_accuracy_score(y_test, y_test_pred)
print(f"训练集平衡准确率: {train_score:.6f}")
print(f"测试集平衡准确率: {test_score:.6f}")

# 打印分类报告
print("测试集分类报告:")
print(classification_report(y_test, y_test_pred))

# 获取特征重要性
feature_importances = clf.get_feature_importance()
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

# 显示前20个重要特征
print(feature_importance_df.head(20))

# 绘制特征重要性图
plt.figure(figsize=(10, 12))
plt.barh(feature_importance_df['feature'][:20][::-1], feature_importance_df['importance'][:20][::-1])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances in CatBoost Model')
plt.tight_layout()
plt.show()