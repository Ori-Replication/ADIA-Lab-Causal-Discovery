import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import re
import os
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
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
    - cat_idxs (list of int): 类别特征的索引。
    - cat_dims (list of int): 每个类别特征的模态数。
    - df (pd.DataFrame): 经过编码后的数据框。
    """
    cat_cols = [col for col in df.columns if df[col].nunique() <= max_unique]
    cat_dims = []
    cat_idxs = []
    encoder_dict = {}

    for col in cat_cols:
        print(f"处理类别特征: {col}，唯一值数量: {df[col].nunique()}")
        # 使用 LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('NaN'))
        cat_dims.append(len(le.classes_))
        cat_idxs.append(df.columns.get_loc(col))

    return cat_idxs, cat_dims, df

# 定义多分类加权交叉熵损失函数
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        初始化加权交叉熵损失函数。

        参数：
        - class_weights (list或np.array或torch.Tensor): 每个类别的权重。
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        if class_weights is not None:
            # 使用 register_buffer 确保 class_weights 在模型保存和加载时被包含
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights = None

    def forward(self, y_pred, y_true):
        """
        前向传播计算损失。

        参数：
        - y_pred (torch.Tensor): 模型的预测输出，形状为 (batch_size, num_classes)。
        - y_true (torch.Tensor): 真实的标签，形状为 (batch_size,)。
        """
        if self.class_weights is not None:
            # 确保 class_weights 在 y_pred 的设备上
            return F.cross_entropy(y_pred, y_true, weight=self.class_weights.to(y_pred.device))
        else:
            return F.cross_entropy(y_pred, y_true)

device = get_device()
num_workers = get_optimal_num_workers()

# 读取数据
X_y_group_train = pd.read_csv('/hy-tmp/mid_data/X_y_group_train_updated_v12.2_piecewise.csv')

print("Adding numeric labels y")
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

def train_model(params):
    # 初始化自定义的加权交叉熵损失函数
    loss_fn = WeightedCrossEntropyLoss(class_weights=class_weights)

    # 初始化模型
    clf = TabNetClassifier(
        n_d=params['n_d&n_a'],                  # 决策层的宽度
        n_a=params['n_d&n_a'],                  # 注意力嵌入的宽度
        n_steps=params['n_steps'],              # 决策步骤数
        gamma=params['gamma'],                  # 特征重用系数
        cat_idxs=cat_idxs,                      # 类别特征的索引列表
        cat_dims=cat_dims,                      # 每个类别特征的模态数
        cat_emb_dim=1,                          # 类别特征的嵌入维度
        n_independent=params['n_independent'],  # 独立的 GLU 层的数量
        n_shared=params['n_shared'],            # 共享的 GLU 层的数量
        epsilon=1e-5,                           # 防止除以零的常数
        seed=42,                                # 随机种子
        momentum=params['momentum'],            # 批量归一化的动量参数
        clip_value=params['clip_value'],        # 梯度剪裁的值
        lambda_sparse=params['lambda_sparse'],  # 稀疏性损失系数
        optimizer_fn=torch.optim.Adam,          # 优化器
        optimizer_params=dict(lr=1e-2),         # 优化器的参数
        scheduler_fn=torch.optim.lr_scheduler.StepLR,  # 学习率调度器
        scheduler_params=dict(step_size=15, gamma=0.5),  # 学习率调度器的参数
        mask_type=params['mask_type'],          # 特征选择的掩码类型
        verbose=1,                              # 是否打印训练过程中的信息
        device_name=device,                     # 使用 GPU 或 CPU
    )

    # 训练模型
    clf.fit(
        X_train=X_train.values,     # 训练集的特征矩阵
        y_train=y_train.values,     # 训练集的目标标签
        eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)],  # 验证集列表
        eval_name=['train', 'valid'],                      # 验证集的名称
        eval_metric=['accuracy', 'balanced_accuracy'],     # 评估指标列表
        max_epochs=2000,            # 最大训练轮数
        loss_fn=loss_fn,            # 自定义的加权交叉熵损失函数
        patience=10,                # 早停的耐心轮数
        batch_size=4096,            # 批量大小
        virtual_batch_size=1024,    # Ghost Batch Normalization 的虚拟批次大小
        num_workers=num_workers,    # DataLoader 的工作线程数
        drop_last=False,            # 是否在训练过程中丢弃最后一个不完整的批次
        callbacks=None,             # 回调函数列表
        compute_importance=False    # 是否计算特征重要性
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
    test_score = train_model(params)  # 使用当前参数训练模型并计算验证集的 balanced_accuracy
    return test_score

# 初始化调优器
study = optuna.create_study(direction='maximize')

# 执行调优
study.optimize(objective, n_trials=40)  # 请根据需要增加 n_trials 的数量
print("Best hyperparameters:", study.best_params)
print("Best test score:", study.best_value)

# 存储模型参数
dump(study.best_params, os.path.join('/hy-tmp/params', f'tabnet_params_{study.best_value:.4f}.joblib'))