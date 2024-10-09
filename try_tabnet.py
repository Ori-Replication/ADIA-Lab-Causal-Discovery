import os
import sys
print(sys.path)
sys.path.append("/hy-tmp/A_dataloader")  # Only在云服务器中运行时
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import torch.nn as nn
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import mean_squared_error
import argparse

def scale_data(data, scaler=None, is_train=True):
    if is_train:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(data)
        return X_scaled

def MSE_RMSE_Loss(y_pred, y_true):
    mse = nn.functional.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse)
    return (mse + rmse) / 2

class MSE_RMSE_Metric(Metric):
    def __init__(self):
        self._name = "mse_rmse_mean"
        self._maximize = False
    def __call__(self, y_true, y_score):
        mse = mean_squared_error(y_true, y_score)
        rmse = np.sqrt(mse)
        return (mse + rmse) / 2

def Weighting_Metric(scores, weights):
    if len(scores) != len(weights):
        raise ValueError("The lengths of scores and weights must be the same.")
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("The sum of weights must not be zero.")
    weighted_metric = weighted_sum / total_weight
    return weighted_metric

def train_model(params):
    # 初始化模型
    model = TabNetRegressor(
        n_d=params['n_d'],
        n_a=params['n_a'],
        n_steps=params['n_steps'],
        gamma=params['gamma'],
        cat_idxs=[],  # 分类特征的索引列表
        cat_dims=[],  # 分类特征的模式数量
        cat_emb_dim=1,  # 每个分类特征的嵌入大小
        n_independent=params['n_independent'],
        n_shared=params['n_shared'],
        momentum=params['momentum'],
        lambda_sparse=params['lambda_sparse'],
        seed=params['seed'],
        clip_value=params['clip_value'],
        device_name='cuda',
        input_dim=X_train.shape[1],
        output_dim=1,
    )

    # KFold交叉验证
    # kf = KFold(n_splits=4, shuffle=True, random_state=42)
    kf = TimeSeriesSplit(n_splits=4, test_size=8640)
    val_folds_Metric = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold [{fold + 1}/4]")

        # 训练
        model.fit(
            X_train=X_train[train_idx],
            y_train=y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            eval_metric=["mse_rmse_mean", MSE_RMSE_Metric],
            max_epochs=params['num_epochs'],
            patience=params['patience'],
            batch_size=params['batch_size'],
            virtual_batch_size=params['virtual_batch_size'],
            loss_fn=MSE_RMSE_Loss,
        )

        # 验证集预测
        val_predictions = model.predict(X_train[val_idx]) * params['correction']
        # 计算损失
        val_Metric = (np.sqrt(mean_squared_error(y_train[val_idx], val_predictions.squeeze())) +
                    mean_squared_error(y_train[val_idx], val_predictions.squeeze())) / 2
        print(f"Validation MSE-RMSE-Mean: {val_Metric:.4f}")
        val_folds_Metric.append(val_Metric)

    # 平均损失
    # avg_val_Metric = np.mean(best_val_Metric_folds)
    # print(f"Average Validation MSE-RMSE-Mean across folds: {avg_val_Metric:.4f}")
    weighted_val_Metric = Weighting_Metric(val_folds_Metric, [1, 2, 3, 4])
    print(f"Weighted Validation MSE-RMSE-Mean across folds: {weighted_val_Metric:.4f}")

    return weighted_val_Metric

def objective(trial):
    params = {
        'n_d': trial.suggest_int('n_d', 36, 64),  # 决策预测层的宽度 n_d
        'n_a': trial.suggest_int('n_a', 36, 64),  # 每个掩码注意力嵌入的宽度 n_a
        'n_steps': trial.suggest_int('n_steps', 4, 10),  # 架构步数 n_steps
        'gamma': trial.suggest_float('gamma', 1.0, 1.8),  # 掩码中特征重用的系数 gamma
        'n_independent': trial.suggest_int('n_independent', 1, 6),  # 独立的GLU层数 n_independent
        'n_shared': trial.suggest_int('n_shared', 1, 6),  # 共享的GLU层数 n_shared
        'momentum': trial.suggest_float('momentum', 0.01, 0.3),  # 动量值 momentum
        'lambda_sparse': trial.suggest_float('lambda_sparse', 0.0001, 0.001, log=True),  # 稀疏正则化参数 lambda_sparse（以对数尺度）
        'seed': 42,  # 随机种子
        'clip_value': trial.suggest_float('clip_value', 1.01, 2.0),  # 梯度裁剪值 clip_value
        'num_epochs': 50,  # 训练的轮数，固定为 50
        'batch_size': trial.suggest_int('batch_size', 128, 1024),  # 批次大小 batch_size
        'patience': 5,  # 提前停止的耐心轮次，固定为 5
        'virtual_batch_size': trial.suggest_int('virtual_batch_size', 16, 64),  # 虚拟批次大小 virtual_batch_size
        'correction': trial.suggest_uniform('correction', 0.9, 0.99)
    }
    avg_val_rmse = train_model(params)  # 使用当前参数训练模型并计算平均验证集Metric
    return avg_val_rmse

if __name__ == '__main__':
    # 设置
    data_dir = f'data_A'
    model_dir = f'model/save'
    data_date = '0810'
    data_version = 'Time_Positional_Baseline'
    submit_date = '0818'
    submit_version = 'optuna_Tabnet'
    train = args.TrainOrTest

    # 加载数据
    drop_col = ['timestamp']
    X_train, X_test, y_train, X_train_drop_col, X_test_drop_col = Load_TrainTest(data_dir, data_date, data_version, drop_col)
    X_train, scaler = scale_data(X_train)
    y_train = y_train.values
    X_test = scale_data(X_test, scaler=scaler, is_train=False)

    if train == 'True':
        # 初始化调优器
        study = optuna.create_study(direction='minimize', sampler=TPESampler(multivariate=True))

        # 执行调优
        study.optimize(objective, n_trials=96)
        print("Best hyperparameters:", study.best_params)
        print("Best validation MSE-RMSE-Mean:", study.best_value)

        # 存储模型参数
        dump(study.best_params, os.path.join(model_dir, f'tabnet_params_{submit_date}_{data_version}.joblib'))

        # 重新训练模型
        tabnet_model = TabNetRegressor(
            n_d=study.best_params['n_d'],
            n_a=study.best_params['n_a'],
            n_steps=study.best_params['n_steps'],
            gamma=study.best_params['gamma'],
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            n_independent=study.best_params['n_independent'],
            n_shared=study.best_params['n_shared'],
            momentum=study.best_params['momentum'],
            lambda_sparse=study.best_params['lambda_sparse'],
            seed=42,
            clip_value=study.best_params['clip_value'],
            device_name='cuda',
            input_dim=X_train.shape[1],
            output_dim=1,
        )

        # 使用整个训练数据集重新训练模型
        tabnet_model.fit(
            X_train=X_train,
            y_train=y_train,
            max_epochs=5,
            patience=3,
            batch_size=study.best_params['batch_size'],
            virtual_batch_size=study.best_params['virtual_batch_size'],
            loss_fn=MSE_RMSE_Loss,
        )

        # 保存模型
        saved_filepath = tabnet_model.save_model(os.path.join(model_dir, f'tabnet_model_{submit_date}_{data_version}'))
        print(f"Model saved at {saved_filepath}")

    else:
        # 加载参数
        params_filepath = f'model/save/tabnet_params_{submit_date}_{data_version}.joblib'
        params = load(params_filepath)
        print(f'最优参数{params}')

        # 加载模型
        saved_filepath = f'model/save/tabnet_model_{submit_date}_{data_version}.zip'
        tabnet_model = TabNetRegressor()
        tabnet_model.load_model(saved_filepath)

        # 预测
        tabnet_pred = tabnet_model.predict(X_test)

        # 融合
        y_pred = 1 * tabnet_pred
        y_pred *= params['correction']

        # 提交
        sample_submit = pd.read_csv(os.path.join(data_dir, 'sample_submit.csv'))
        sample_submit["clearing price (CNY/MWh)"] = y_pred
        sample_submit.to_csv(f'submit/{submit_date}_{submit_version}.csv', index=False)