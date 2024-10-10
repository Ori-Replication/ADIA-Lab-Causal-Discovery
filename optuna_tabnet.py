import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import multiprocessing
import re
import os
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
from joblib import dump, load

def get_device():
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
        if name[0].isdigit():
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
    - cat_idxs (list of int): 类别特征的索引。
    - cat_dims (list of int): 每个类别特征的模态数。
    - encoder_dict (dict): 存储每个类别特征的 LabelEncoder 实例。
    """
    cat_cols = [col for col in df.columns if df[col].nunique() <= max_unique]
    cat_dims = []
    cat_idxs = []
    encoder_dict = {}

    for col in cat_cols:
        print(f"处理类别特征: {col}，唯一值数量: {df[col].nunique()}")
        # 使用 LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)
        cat_dims.append(len(le.classes_))
        cat_idxs.append(df.columns.get_loc(col))

    return cat_idxs, cat_dims, df

device = get_device()
num_workers = get_optimal_num_workers()

X_y_group_train = pd.read_csv('/hy-tmp/mid_data/X_y_group_train_updated_v12.2_piecewise.csv')

print("Adding numeric labels y")
le = LabelEncoder()
X_y_group_train["y"] = le.fit_transform(X_y_group_train["label"])
# reordering columns:
X_y_group_train = X_y_group_train[["dataset", "variable"] + X_y_group_train.columns.drop(["dataset", "variable", "label", "y"]).tolist() + ["label", "y"]]

blacklist = ["ttest(v,X)", "pvalue(ttest(v,X))<=0.05", "ttest(v,Y)", "pvalue(ttest(v,Y))<=0.05", "ttest(X,Y)", "pvalue(ttest(X,Y))<=0.05"\
    "square_dimension", "max(PPS(v,others))"]
columns_to_drop = [col for col in blacklist if col in X_y_group_train.columns]
X_y_group_train = X_y_group_train.drop(columns=columns_to_drop)

numeric_columns = X_y_group_train.select_dtypes(include=[np.number]).columns
X_y_group_train[numeric_columns] = X_y_group_train[numeric_columns].fillna(X_y_group_train[numeric_columns].mean())

# display(X_y_group_train)

# 清理特征名称
X_y_group_train = clean_feature_names(X_y_group_train)

print("Extracting X_train, y_train, and group")
# 分离数据集ID、特征和标签
group_train = X_y_group_train["dataset"]
X = X_y_group_train.drop(["variable", "dataset", "label", "y"], axis="columns")
y = X_y_group_train["y"]

# 处理类别特征
cat_idxs, cat_dims, X = process_categorical_features(X)
print(f"类别特征索引 (cat_idxs): {cat_idxs}")
print(f"类别特征模态数 (cat_dims): {cat_dims}")

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("y_train 唯一值:", np.unique(y_train))
print("y_test 唯一值:", np.unique(y_test))

def train_model(params):
    # 初始化模型
    clf = TabNetClassifier(
        n_d=params['n_d&n_a'],                  # 决策层的宽度（小心过拟合）
        n_a=params['n_d&n_a'],                  # 注意力嵌入的宽度（一般与n_d一致）
        n_steps=params['n_steps'],          # 决策步骤数（3-10）
        gamma=params['gamma'],              # 特征重用系数（值接近 1 会减少层间的特征选择相关性，1.0-2.0）
        cat_idxs=cat_idxs,                  # 类别特征的索引列表
        cat_dims=cat_dims,                  # 每个类别特征的模态数（即类别数量）
        cat_emb_dim=1,                      # 类别特征的嵌入维度
        n_independent=params['n_independent'],      # 每个步骤中独立的 Gated Linear Units (GLU) 层的数量（1-5）
        n_shared=params['n_shared'],                # 每个步骤中共享的 GLU 层的数量（1-5）
        epsilon=1e-5,                       # 防止除以零的常数
        seed=42,                            # 随机种子
        momentum=params['momentum'],                # 批量归一化的动量参数（0.01-0.4）
        clip_value=params['clip_value'],            # 如果设置为浮点数，将梯度剪裁到该值
        lambda_sparse=params['lambda_sparse'],      # 额外的稀疏性损失系数，值越大，模型在特征选择上越稀疏（1e-3-1e-1）
        optimizer_fn=torch.optim.Adam,                      # 优化器
        optimizer_params=dict(lr=1e-2),                     # 优化器的参数
        scheduler_fn=torch.optim.lr_scheduler.StepLR,       # 学习率调度器
        scheduler_params=dict(step_size=15, gamma=0.5),     # 学习率调度器的参数
        mask_type=params['mask_type'],   # 特征选择的掩码类型（'sparsemax' 或 'entmax'）
        # grouped_features=None,         # 将特征分组，使模型在同一组内共享注意力。这在特征预处理生成相关或依赖特征时尤其有用，例如使用 TF-IDF 或 PCA。特征重要性在同组内将相同。
        verbose=1,                       # 是否打印训练过程中的信息（0 或 1）
        device_name=device,              # 使用 GPU
    )

    # 训练模型
    clf.fit(
        X_train=X_train.values,     # 训练集的特征矩阵（np.array）
        y_train=y_train.values,     # 训练集的目标标签（np.array，对于多分类任务，标签应为整数编码）
        eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)],   # 验证集列表
        eval_name=['train', 'valid'],                      # 验证集的名称
        eval_metric=['accuracy', 'balanced_accuracy'],     # 评估指标列表
        max_epochs=2000,            # 最大训练轮数
        loss_fn=,
        patience=10,                # 早停的耐心轮数
        batch_size=4096,            # 批量大小
        virtual_batch_size=1024,    # 用于 Ghost Batch Normalization 的虚拟批次大小（应能被 batch_size 整除）
        num_workers=8,    # 用于 torch.utils.data.DataLoader 的工作线程数
        drop_last=False,            # 是否在训练过程中丢弃最后一个不完整的批次
        callbacks=None,             # 回调函数列表
        compute_importance=False,   # 是否计算特征重要性
    )

    # 预测
    y_train_pred = clf.predict(X_train.values)
    y_test_pred = clf.predict(X_test.values)

    # 计算平衡准确率
    train_score = balanced_accuracy_score(y_train, y_train_pred)
    test_score = balanced_accuracy_score(y_test, y_test_pred)
    print(f"训练集平衡准确率: {train_score:.6f}")
    print(f"测试集平衡准确率: {test_score:.6f}")

    return test_score

def objective(trial):
    params = {
        'n_d&n_a': trial.suggest_int('n_d', 64, 128),
        'n_steps': trial.suggest_int('n_steps', 4, 12),
        'gamma': trial.suggest_float('gamma', 1.0, 1.8), 
        'n_independent': trial.suggest_int('n_independent', 1, 6), 
        'n_shared': trial.suggest_int('n_shared', 1, 6), 
        'momentum': trial.suggest_float('momentum', 0.01, 0.3),
        'clip_value': trial.suggest_float('clip_value', 1.01, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 0.0001, 0.001, log=True), 
        'mask_type': trial.suggest_categorical('mask_type', ["sparsemax", "entmax"])
    }
    test_score = train_model(params)  # 使用当前参数训练模型并计算平均验证集Metric
    return test_score

# 初始化调优器
study = optuna.create_study(direction='maximize')

# 执行调优
study.optimize(objective, n_trials=2)
print("Best hyperparameters:", study.best_params)
print("Best test score:", study.best_value)

# 存储模型参数
dump(study.best_params, os.path.join('/hy-tmp/params', f'tabnet_params_{study.best_value:.4f}.joblib'))