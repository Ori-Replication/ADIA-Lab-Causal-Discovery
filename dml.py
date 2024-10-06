def dml_estimate(data, Y_var, T_var, X_vars, n_splits=4, use_gpu=False):
    """
    使用双重机器学习估计T对Y的因果效应。
    返回：
    - result: 包含以下键的字典：
        - 'theta': 估计的因果效应。
        - 'se': 估计的标准误差。
    """
    # 从DataFrame中提取变量
    Y = data[Y_var].values
    T = data[T_var].values
    X = data[X_vars].values

    # 初始化残差
    Y_residuals = np.zeros_like(Y)
    T_residuals = np.zeros_like(T)

    # 设置交叉拟合
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # LightGBM参数
    params = {
        'objective': 'regression',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbosity': -1,
        'device_type': 'cuda' if use_gpu else 'cpu',
        'n_jobs': -1
    }

    # 如果使用GPU，加上其他GPU相关参数
    if use_gpu:
        # 根据最新的LightGBM文档，这些参数只在特定情况下需要
        # 这里只设置device_type为'gpu'，其余参数使用默认值
        params['max_bin'] = 63  # 建议在GPU模式下使用较小的max_bin值

    # 交叉拟合循环
    for train_index, test_index in kf.split(X):
        # 将数据拆分为训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        T_train, T_test = T[train_index], T[test_index]

        # 结果模型
        Y_model = LGBMRegressor(**params)
        Y_model.fit(X_train, Y_train)
        Y_pred = Y_model.predict(X_test)
        Y_residuals[test_index] = Y_test - Y_pred

        # 处理模型
        T_model = LGBMRegressor(**params)
        T_model.fit(X_train, T_train)
        T_pred = T_model.predict(X_test)
        T_residuals[test_index] = T_test - T_pred

    # 使用残差进行线性回归估计因果效应
    causal_model = LinearRegression(fit_intercept=False)
    causal_model.fit(T_residuals.reshape(-1, 1), Y_residuals)
    theta = causal_model.coef_[0]

    # 计算标准误差
    n = len(Y_residuals)
    residuals = Y_residuals - theta * T_residuals
    sigma2 = np.sum(residuals ** 2) / (n - 1)
    T_residuals_variance = np.var(T_residuals, ddof=1)
    se = np.sqrt(sigma2 / (n * T_residuals_variance))

    # 返回结果
    result = {
        'theta': theta,
        'se': se
    }
    return result

def double_machine_learning(dataset):
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # 判断v-X的因果效应，设置variables中的其他v和Y为控制变量
        Y_var = "X"
        T_var = variable
        X_vars = [var for var in dataset.columns.tolist() if var not in [Y_var, T_var]]
        result = dml_estimate(dataset, Y_var, T_var, X_vars, n_splits=4, use_gpu=False)

        df.append({
            "variable": variable,
            "v~X_DML_theta": result['theta'],
            # "v~X_DML_se": result['se']
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df