"""LGBM & DML use econml package"""
class LightGBMRegressorFixed(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 n_estimators=100, 
                 learning_rate=0.05, 
                 max_depth=-1,
                 num_leaves=31, 
                 subsample=0.8, 
                 colsample_bytree=0.8,
                 random_state=42,
                 **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None  # 在 fit 时初始化模型

    def fit(self, X, y, **kwargs):
        # 移除 'random_state' 参数以避免传递给 LightGBM 的 fit 方法
        kwargs = {k: v for k, v in kwargs.items() if k != 'random_state'}
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            **self.kwargs
        )
        self.model_.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise NotFittedError("This LightGBMRegressorFixed instance is not fitted yet.")
        return self.model_.predict(X)

    def get_params(self, deep=True):
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
def DML(T_, Y_, X_, data):
    """
    双重机器学习 (Double Machine Learning, DML) 函数，使用 LassoCV 作为第一阶段模型。
    参数:
        T_ (str): 处理变量的列名。
        Y_ (str): 结果变量的列名。
        X_ (list of str): 控制变量的列名。
        data (pd.DataFrame): 包含上述列的数据集。
    返回:
        amte (float): 平均处理效应 (Average Treatment Effect, ATE)。
        amte_inference (object): ATE 的推断结果。
        amte_interval (tuple): ATE 的置信区间。
    """
    # 提取处理变量、结果变量和控制变量
    T = data[T_].values.reshape(-1, 1)
    Y = data[Y_].values.reshape(-1, 1)
    X = data[X_].values

    # 定义 LassoCV 作为第一阶段模型
    model_t = LassoCV(cv=4, random_state=42, n_jobs=-1)
    model_y = LassoCV(cv=4, random_state=42, n_jobs=-1)

    # 初始化 CausalForestDML 使用自定义的 LassoCV 估计器
    model = CausalForestDML(
        model_t=model_t,
        model_y=model_y,
        n_jobs=-1,
        random_state=42,  # 确保随机性的一致性
        inference=True
    )
    model.fit(Y, T, X=X)

    # 计算平均处理效应 (ATE)
    amte = model.ate(X=X)
    # ATE 的置信区间
    amte_interval = model.ate_interval(X=X, alpha=0.05)
    # 推断结果（这里等同于置信区间）
    amte_inference = amte_interval

    return amte, amte_inference, amte_interval
def DML(T_, Y_, X_, data):
    # 设置处理变量、结果变量和控制变量
    T = data[T_].values.reshape(-1, 1)
    Y = data[Y_].values.reshape(-1, 1)
    X = data[X_].values

    # 定义自定义的 LightGBM 估计器
    model_t = LightGBMRegressorFixed()
    model_y = LightGBMRegressorFixed()

    # 初始化 CausalForestDML 使用自定义的估计器，并设置 random_state
    model = CausalForestDML(
        model_t=model_t,
        model_y=model_y,
        n_jobs=-1,
        random_state=42,  # 确保随机性的一致性
        inference=True
    )
    model.fit(Y, T, X=X)
    
    # 计算平均边际处理效应
    amte = model.ate(X=X)
    # 平均边际处理效应的推断结果
    amte_inference = model.ate_interval(X=X, alpha=0.05)
    # 平均边际处理效应的置信区间
    amte_interval = model.ate_interval(X=X, alpha=0.05)

    return amte, amte_inference, amte_interval

def double_machine_learning(dataset):
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # 判断v-X的因果效应，设置variables中的其他v和Y为控制变量
        amte, amte_inference, amte_interval = DML(variable, "X", ["Y"] + list(variables.drop(variable)), dataset)

        df.append({
            "variable": variable,
            "v~X_DML_AMTE": amte.item(),  
            # "v~X_DML_AMTE_stderr": amte_inference.stderr_mean.item(),
            # "v~X_DML_AMTE_pvalue": amte_inference.pvalue_mean.item(),
            # "v~X_DML_AMTE_lower": amte_interval[0].item(),
            # "v~X_DML_AMTE_upper": amte_interval[1].item()
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df




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
        'device_type': 'gpu' if use_gpu else 'cpu',
        'gpu_platform_id': 1,
        'gpu_device_id': 0,
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

"""ci pillai test use pgmpy package"""
def conditional_independence_tests(dataset):  # 太慢了，得三个小时
    """
    A mixed-data residualization based conditional independence test[1].
    Uses XGBoost estimator to compute LS residuals[2], and then does an association test (Pillai’s Trace) on the residuals.
    """
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    df = []
    for variable in variables:
        # v-X
        coef1, p_value1 = CITests.ci_pillai(X=variable, Y="X", Z=dataset.columns.drop(["X", variable]).tolist(), data=dataset, boolean=False)
        # v-Y
        coef2, p_value2 = CITests.ci_pillai(X=variable, Y="Y", Z=dataset.columns.drop(["Y", variable]).tolist(), data=dataset, boolean=False)
        # X-v
        coef3, p_value3 = CITests.ci_pillai(X="X", Y=variable, Z=dataset.columns.drop(["X", variable]).tolist(), data=dataset, boolean=False)
        # Y-v
        coef4, p_value4 = CITests.ci_pillai(X="Y", Y=variable, Z=dataset.columns.drop(["Y", variable]).tolist(), data=dataset, boolean=False)
        df.append({
            "variable": variable,
            "v~X_ci_pillai_coef": coef1,
            "v~X_ci_pillai_p_value": p_value1,
            "v~Y_ci_pillai_coef": coef2,
            "v~Y_ci_pillai_p_value": p_value2,
            "X~v_ci_pillai_coef": coef3,
            "X~v_ci_pillai_p_value": p_value3,
            "Y~v_ci_pillai_coef": coef4,
            "Y~v_ci_pillai_p_value": p_value4
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df


"""弱化版本"""
def run_dml(T_, Y_, X_, data):
    """
    双重机器学习 (Double Machine Learning, DML) 函数，使用 LassoCV 作为第一阶段模型。
    参数:
        T_ (str): 处理变量的列名。
        Y_ (str): 结果变量的列名。
        X_ (list of str): 控制变量的列名。
        data (pd.DataFrame): 包含上述列的数据集。

    返回:
        amte (float): 平均处理效应 (Average Treatment Effect, ATE)。
        amte_inference (object): ATE 的推断结果。
        amte_interval (tuple): ATE 的置信区间。
    """
    # 提取处理变量、结果变量和控制变量
    T = data[T_].values.reshape(-1, 1)
    Y = data[Y_].values.reshape(-1, 1)
    X = data[X_].values

    model_t = LassoCV(cv=4, random_state=42, n_jobs=-1)
    model_y = LassoCV(cv=4, random_state=42, n_jobs=-1)
    model_final = LinearRegression()

    # 初始化 DML 使用 econml.dml.DML 类
    model = DML(
        model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        discrete_treatment=False, 
        random_state=42,
        cv=4
    )
    model.fit(Y, T, X=X)

    # 计算平均处理效应 (ATE)
    amte = model.ate(X=X)
    # # 计算 ATE 的推断结果
    # amte_inference = model.ate_inference(X=X)
    # # 计算 ATE 的置信区间
    # amte_interval = model.ate_interval(X=X, alpha=0.05)

    return amte#, amte_inference, amte_interval

def result_from_dml(amte, variable): # , amte_inference, amte_interval, ):
    # 确保 amte 是标量，以使用 .item()
    if np.isscalar(amte):
        v_amte = amte
    elif isinstance(amte, np.ndarray) and amte.size == 1:
        v_amte = amte.item()
    else:
        raise ValueError(f"ATE (amte) for variable {variable} is not scalar.")

    # # 从推断结果中提取标准误和 p 值
    # summary = amte_inference.summary_frame()
    # v_amte_stderr = summary["stderr"].mean()
    # v_amte_pvalue = summary["pvalue"].mean()

    # # 提取置信区间
    # lower, upper = amte_interval

    # # 如果 lower 和 upper 是数组，提取第一个元素
    # if isinstance(lower, np.ndarray) and lower.size == 1:
    #     lower = lower.item()
    # if isinstance(upper, np.ndarray) and upper.size == 1:
    #     upper = upper.item()

    return v_amte# , v_amte_stderr, v_amte_pvalue, lower, upper

def double_machine_learning(dataset):
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # 判断v-X的因果效应，设置variables中的其他v和Y为控制变量
        amte, amte_inference, amte_interval = run_dml(variable, "X", ["Y"] + list(variables.drop(variable)), dataset)
        v_amte = result_from_dml(amte, variable)

        df.append({
            "variable": variable,
            "v~X_DML_AMTE": v_amte,  
            # "v~X_DML_AMTE_stderr": v_amte_stderr,
            # "v~X_DML_AMTE_pvalue": v_amte_pvalue,
            # "v~X_DML_AMTE_lower": lower,
            # "v~X_DML_AMTE_upper": upper
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df


def ttest(dataset, pvalue_threshold=0.05):
    """
    Given a dataset, this function computes the t-test between the
    values each variable v and X, Y. The t value and the result of the
    t-test with a given pvalue_threshold, are used to create features
    to describe/embed v, as well as the t-test result between the
    values of X and Y.
    """

    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        ttest_vX = ttest_rel(dataset[variable], dataset["X"])
        ttest_vY = ttest_rel(dataset[variable], dataset["Y"])

        df.append({
            "variable": variable,
            "ttest(v,X)": ttest_vX.statistic,
            f"pvalue(ttest(v,X))<={pvalue_threshold}": (ttest_vX.pvalue <= pvalue_threshold).astype(float),
            "ttest(v,Y)": ttest_vY.statistic,
            f"pvalue(ttest(v,Y))<={pvalue_threshold}": (ttest_vY.pvalue <= pvalue_threshold).astype(float),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    ttest_XY = ttest_rel(dataset["X"], dataset["Y"])
    df["ttest(X,Y)"] = ttest_XY.statistic
    df[f"pvalue(ttest(X,Y))<={pvalue_threshold}"] = (ttest_XY.pvalue <= pvalue_threshold).astype(float)

    # some the ttest returns NaN when the variance is 0, so we fill with 0:
    df.fillna(0, inplace=True)

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def custom_distance_correlation(dataset):
    """
    Compute distance correlation with custom distance metrics.
    
    Parameters:
    - dataset: pandas DataFrame
    
    Returns:
    - pandas DataFrame with distance correlation features
    """
    metric = 'chebyshev'  # You can change this to 'euclidean', 'manhattan', or 'minkowski'
    variables = dataset.columns.drop(["X", "Y"])
    
    def compute_distance_correlation(x, y):
        # Convert Series to numpy arrays and reshape
        x_array = x.values.reshape(-1, 1)
        y_array = y.values.reshape(-1, 1)
        
        x_dist = squareform(pdist(x_array, metric=metric))
        y_dist = squareform(pdist(y_array, metric=metric))
        return dcor.distance_correlation_sqr(x_dist, y_dist)
    
    df = []
    for variable in variables:
        df.append({
            "variable": variable,
            f"dcor_{metric}(v,X)": compute_distance_correlation(dataset[variable], dataset["X"]),
            f"dcor_{metric}(v,Y)": compute_distance_correlation(dataset[variable], dataset["Y"]),
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    df[f"dcor_{metric}(X,Y)"] = compute_distance_correlation(dataset["X"], dataset["Y"])
    
    # Reorder columns:
    df = df[["dataset", "variable"] + [colname for colname in df.columns if colname not in ["dataset", "variable"]]]
    
    return df



def entropy_features(dataset):
    """
    Calculate entropy and conditional entropy features for each variable.
    """
    variables = dataset.columns.drop(["X", "Y"])
    
    df = []
    for variable in variables:
        
        # Calculate conditional entropies
        joint_vX, _, _ = np.histogram2d(dataset[variable], dataset["X"], bins=20)
        joint_vY, _, _ = np.histogram2d(dataset[variable], dataset["Y"], bins=20)
        
        # Normalize the joint distributions
        joint_vX = joint_vX / joint_vX.sum()
        joint_vY = joint_vY / joint_vY.sum()
        
        # Calculate marginal distributions
        p_v_X = joint_vX.sum(axis=1)
        p_v_Y = joint_vY.sum(axis=1)
        
        # Calculate conditional entropies
        cond_entropy_X_v = np.sum(p_v_X * entropy(joint_vX.T, axis=0))
        cond_entropy_Y_v = np.sum(p_v_Y * entropy(joint_vY.T, axis=0))
        
        df.append({
            "variable": variable,
            "conditional_entropy(X|v)": cond_entropy_X_v,
            "conditional_entropy(Y|v)": cond_entropy_Y_v,
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]
    
    return df



def hilbert_schmidt_independence(dataset):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between variables.
    Features:
        - hsic(v, X)
        - hsic(v, Y)
        - hsic(X, Y)
    """
    variables = dataset.columns.drop(["X", "Y"])
    
    df = []
    for variable in variables:
        hsic_vX = hsic(dataset[variable].values, dataset["X"].values)
        hsic_vY = hsic(dataset[variable].values, dataset["Y"].values)
        df.append({
            "variable": variable,
            "hsic(v,X)": hsic_vX,
            "hsic(v,Y)": hsic_vY,
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    hsic_XY = hsic(dataset["X"].values, dataset["Y"].values)
    df["hsic(X,Y)"] = hsic_XY
    
    # Reorder columns
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]
    
    return df

def hsic(x, y, sigma=None):
    """
    Compute the HSIC between two variables x and y.
    x and y are numpy arrays of shape (n_samples,)
    """
    n = x.shape[0]
    x = x.reshape((n, -1))
    y = y.reshape((n, -1))
    
    if sigma is None:
        sigma = np.std(np.concatenate((x, y), axis=0))
        if sigma == 0:
            sigma = 1.0

    # Compute the Gram matrices using RBF kernel
    K = rbf_kernel(x, gamma=1.0/(2*sigma**2))
    L = rbf_kernel(y, gamma=1.0/(2*sigma**2))
    
    # Center the Gram matrices
    H = np.eye(n) - np.ones((n, n))/n
    Kc = H @ K @ H
    Lc = H @ L @ H
    
    # Compute HSIC
    hsic_value = (1/(n-1)**2) * np.trace(Kc @ Lc)
    return hsic_value



def statistical_tests(dataset):
    """
    Compute p-values from statistical tests for each variable with X and Y.
    """
    variables = dataset.columns.drop(["X", "Y"])
    
    df = []
    for variable in variables:
        # Chi-square test
        chi2_vX = chi2_contingency(pd.crosstab(dataset[variable], dataset["X"]))[1]
        chi2_vY = chi2_contingency(pd.crosstab(dataset[variable], dataset["Y"]))[1]
        
        # Kolmogorov-Smirnov test
        ks_vX = ks_2samp(dataset[variable], dataset["X"])[1]
        ks_vY = ks_2samp(dataset[variable], dataset["Y"])[1]
        
        df.append({
            "variable": variable,
            "pvalue(chi2_test(v,X))": chi2_vX,
            "pvalue(chi2_test(v,Y))": chi2_vY,
            "pvalue(ks_test(v,X))": ks_vX,
            "pvalue(ks_test(v,Y))": ks_vY,
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]
    
    return df

def cross_entropy_features(dataset):
    """
    Compute cross-entropy based features for each variable with X and Y.
    """
    variables = dataset.columns.drop(["X", "Y"])
    
    def compute_cross_entropy(p, q):
        # Ensure p and q are probability distributions
        p = softmax(p)
        q = softmax(q)
        return -np.sum(p * np.log(q + 1e-10))
    
    df = []
    for variable in variables:
        # Normalize the data to [0, 1] range
        scaler = MinMaxScaler()
        v_normalized = scaler.fit_transform(dataset[[variable]]).flatten()
        X_normalized = scaler.fit_transform(dataset[["X"]]).flatten()
        Y_normalized = scaler.fit_transform(dataset[["Y"]]).flatten()
        
        cross_entropy_vX = compute_cross_entropy(v_normalized, X_normalized)
        cross_entropy_vY = compute_cross_entropy(v_normalized, Y_normalized)
        
        # Compute cross-entropy with other variables
        other_vars = [col for col in variables]
        cross_entropies = []
        for other_var in other_vars:
            other_normalized = scaler.fit_transform(dataset[[other_var]]).flatten()
            cross_entropies.append(compute_cross_entropy(v_normalized, other_normalized))
        
        df.append({
            "variable": variable,
            "cross_entropy(v,X)": cross_entropy_vX,
            "cross_entropy(v,Y)": cross_entropy_vY,
            "max(cross_entropy(v,others))": max(cross_entropies),
            "min(cross_entropy(v,others))": min(cross_entropies),
            "mean(cross_entropy(v,others))": np.mean(cross_entropies),
            "std(cross_entropy(v,others))": np.std(cross_entropies),
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Compute cross-entropy between X and Y
    X_normalized = scaler.fit_transform(dataset[["X"]]).flatten()
    Y_normalized = scaler.fit_transform(dataset[["Y"]]).flatten()
    df["cross_entropy(X,Y)"] = compute_cross_entropy(X_normalized, Y_normalized)
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]
    
    return df



def multi_mutual_information(dataset):
    """
    Compute multi-variable mutual information for each variable with X and Y.
    """
    variables = dataset.columns.drop(["X", "Y"])
    
    def compute_multi_mi(X, Y, Z):
        # Discretize continuous variables
        kbd = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        X_disc = kbd.fit_transform(X.reshape(-1, 1))
        Y_disc = kbd.fit_transform(Y.reshape(-1, 1))
        Z_disc = kbd.fit_transform(Z.reshape(-1, 1))
        
        # Compute joint and marginal entropies
        XYZ = np.c_[X_disc, Y_disc, Z_disc]
        XY = np.c_[X_disc, Y_disc]
        XZ = np.c_[X_disc, Z_disc]
        YZ = np.c_[Y_disc, Z_disc]
        
        H_XYZ = mutual_info_regression(XYZ, XYZ[:, 0])[0]
        H_XY = mutual_info_regression(XY, XY[:, 0])[0]
        H_XZ = mutual_info_regression(XZ, XZ[:, 0])[0]
        H_YZ = mutual_info_regression(YZ, YZ[:, 0])[0]
        H_X = mutual_info_regression(X_disc, X_disc[:, 0])[0]
        H_Y = mutual_info_regression(Y_disc, Y_disc[:, 0])[0]
        H_Z = mutual_info_regression(Z_disc, Z_disc[:, 0])[0]
        
        # Compute multi-variable mutual information
        return H_X + H_Y + H_Z - H_XY - H_XZ - H_YZ + H_XYZ
    
    df = []
    for variable in variables:
        multi_mi = compute_multi_mi(dataset[variable].values, dataset["X"].values, dataset["Y"].values)
        
        df.append({
            "variable": variable,
            "multi_mutual_information(v,X,Y)": multi_mi,
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]
    
    return df



def hilbert_schmidt_independence_rff(dataset, n_components=100, random_state=None):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between variables
    using Random Fourier Features for efficiency.
    
    Parameters:
    - dataset: pandas DataFrame containing the data.
    - n_components: int, number of random Fourier features.
    - random_state: int or None, for reproducibility.
    
    Returns:
    - df: pandas DataFrame with HSIC features.
    """
    variables = dataset.columns.drop(["X", "Y"])
    rng = check_random_state(random_state)
    
    # Prepare the random weights and biases for RFF
    sigma = np.std(dataset.values)
    if sigma == 0:
        sigma = 1.0
    gamma = 1.0 / (2 * sigma ** 2)
    
    def compute_rff(X):
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1)
        W = rng.normal(scale=np.sqrt(2 * gamma), size=(X.shape[1], n_components))
        b = rng.uniform(0, 2 * np.pi, size=n_components)
        Z = np.sqrt(2.0 / n_components) * np.cos(X @ W + b)
        return Z

    # Compute RFF for X and Y once
    Z_X = compute_rff(dataset["X"].values)
    Z_Y = compute_rff(dataset["Y"].values)
    
    # Center the RFF features
    Z_X -= Z_X.mean(axis=0)
    Z_Y -= Z_Y.mean(axis=0)
    
    df = []
    for variable in variables:
        Z_v = compute_rff(dataset[variable].values)
        Z_v -= Z_v.mean(axis=0)
        
        # Compute HSIC approximations
        hsic_vX = np.sum(Z_v * Z_X) / (dataset.shape[0] - 1)
        hsic_vY = np.sum(Z_v * Z_Y) / (dataset.shape[0] - 1)
        
        df.append({
            "variable": variable,
            "hsic(v,X)": hsic_vX,
            "hsic(v,Y)": hsic_vY,
        })
    
    df = pd.DataFrame(df)
    df["dataset"] = dataset.name
    
    # Compute HSIC between X and Y
    hsic_XY = np.sum(Z_X * Z_Y) / (dataset.shape[0] - 1)
    df["hsic(X,Y)"] = hsic_XY
    
    # Reorder columns
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]
    
    return df

def affine_invariant_distance_correlation(dataset):
    """
    Given a dataset, we compute the affine-invariant distance correlation-based features for each
    variable, which are the affine-invariant distance correlation between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs of affine-invariant distance correlations.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        tmp = []
        # Compute affine-invariant distance correlation between 'variable' and all other variables (excluding itself)
        other_variables = dataset.columns.drop([variable])
        for other_var in other_variables:
            corr = dcor.distance_correlation_af_inv_sqr(dataset[variable], dataset[other_var])
            tmp.append(corr)
        tmp = pd.Series(tmp)  # Convert tmp to a Pandas Series

        distance_correlation_v_X = dcor.distance_correlation_af_inv_sqr(dataset[variable], dataset["X"])
        distance_correlation_v_Y = dcor.distance_correlation_af_inv_sqr(dataset[variable], dataset["Y"])
        distance_correlation_X_Y = dcor.distance_correlation_af_inv_sqr(dataset["X"], dataset["Y"])
        
        df.append({
            "variable": variable,
            "af_dcor(v,X)": distance_correlation_v_X,
            "af_dcor(v,Y)": distance_correlation_v_Y,
            "max(af_dcor(v, others))": tmp.max(),
            "min(af_dcor(v, others))": tmp.min(),
            "mean(af_dcor(v, others))": tmp.mean(),
            "std(af_dcor(v, others))": tmp.std(),
            "25%(af_dcor(v, others))": tmp.quantile(0.25),
            "75%(af_dcor(v, others))": tmp.quantile(0.75),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["af_dcor(X,Y)"] = distance_correlation_X_Y

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df



def energy_distance_features(dataset):
    """
    Given a dataset, we compute the energy distance-based features for each
    variable, which are the energy distance between that variable with X and Y,
    as well as summary statistics (max, min, mean, std) of all pairs of energy distances.
    """
    variables = dataset.columns.drop(["X", "Y"])

    df = []
    for variable in variables:
        # tmp = []
        # # Compute energy distance between 'variable' and all other variables (excluding itself)
        # other_variables = dataset.columns.drop([variable])
        # for other_var in other_variables:
        #     energy_dist = dcor.energy_distance(dataset[variable], dataset[other_var])
        #     tmp.append(energy_dist)
        # tmp = pd.Series(tmp)  # Convert tmp to a Pandas Series

        energy_distance_v_X = dcor.energy_distance(dataset[variable], dataset["X"])
        energy_distance_v_Y = dcor.energy_distance(dataset[variable], dataset["Y"])
        energy_distance_X_Y = dcor.energy_distance(dataset["X"], dataset["Y"])
        
        df.append({
            "variable": variable,
            "energy_dist(v,X)": energy_distance_v_X,
            "energy_dist(v,Y)": energy_distance_v_Y,
            # "max(energy_dist(v, others))": tmp.max(),
            # "min(energy_dist(v, others))": tmp.min(),
            # "mean(energy_dist(v, others))": tmp.mean(),
            # "std(energy_dist(v, others))": tmp.std(),
            # "25%(energy_dist(v, others))": tmp.quantile(0.25),
            # "75%(energy_dist(v, others))": tmp.quantile(0.75),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    df["energy_dist(X,Y)"] = energy_distance_X_Y

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df
def cluster_features(dataset):
    variables = dataset.columns.drop(["X", "Y"]).tolist()
    scaler = StandardScaler()
    d_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    results = []
    eps_values = [0.3]  # , 0.5, 0.7
    
    for variable in variables:
        cluster_counts = []
        noise_counts = []
        avg_cluster_sizes = []
        density_variations = []
        feature_importances = []
        silhouette_scores = []
        
        for eps in eps_values:
            cluster_df = d_scaled[[variable, "X", "Y"]].copy()
            dbscan = DBSCAN(eps=eps, min_samples=5)
            cluster_df["cluster"] = dbscan.fit_predict(cluster_df)
            
            # 基本统计
            cluster_count = len(set(cluster_df["cluster"])) - (1 if -1 in cluster_df["cluster"] else 0)
            noise_count = (cluster_df["cluster"] == -1).sum()
            cluster_counts.append(cluster_count)
            noise_counts.append(noise_count)
            
            # 密度分析
            cluster_sizes = cluster_df[cluster_df["cluster"] != -1]["cluster"].value_counts()
            avg_cluster_size = cluster_sizes.mean() if not cluster_sizes.empty else 0
            density_variation = cluster_sizes.std() / avg_cluster_size if avg_cluster_size > 0 else 0
            avg_cluster_sizes.append(avg_cluster_size)
            density_variations.append(density_variation)
            
            # 特征重要性
            if cluster_count > 1:  # 确保有多个簇
                feature_importance = abs(np.corrcoef(cluster_df[variable], cluster_df["cluster"]))[0, 1]
            else:
                feature_importance = 0
            feature_importances.append(feature_importance)
            
            # 轮廓系数
            non_noise_mask = cluster_df["cluster"] != -1
            if len(set(cluster_df.loc[non_noise_mask, "cluster"])) > 1:
                sil_score = silhouette_score(cluster_df.loc[non_noise_mask, [variable, "X", "Y"]], 
                                             cluster_df.loc[non_noise_mask, "cluster"], 
                                             metric="euclidean")
            else:
                sil_score = 0
            silhouette_scores.append(sil_score)
        
        result = {
            "variable": variable
        }
        for i, eps in enumerate(eps_values):
            result.update({
                f"cluster_count_{eps}": cluster_counts[i],             # 0.4730-0.4736
                # f"noise_count_{eps}": noise_counts[i],                 # 0.4736-0.4740
                # f"avg_cluster_size_{eps}": avg_cluster_sizes[i],     # 0.4740-0.4735
                # f"density_variation_{eps}": density_variations[i],     # 0.4740-0.4741
                # f"feature_importance_{eps}": feature_importances[i], # 0.4741-0.4736
                # f"silhouette_score_{eps}": silhouette_scores[i]      # 0.4741-0.4723
            })
        results.append(result)

    df = pd.DataFrame(results)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

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
        'device_type': 'gpu' if use_gpu else 'cpu',
        'gpu_platform_id': 1,
        'gpu_device_id': 0,
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

def PC_estimate(dataset, mcv, significance, isprint):
    dim = dataset.shape[1]
    if mcv == 'dim-1':
        mcv = dim - 1
    else:
        mcv = int(mcv)

    # 使用estimate方法学习DAG结构
    est = PC(dataset)
    estimated_model = est.estimate(
        variant='parallel',
        ci_test='pearsonr',
        max_cond_vars=mcv,  # 减少最大条件变量数
        return_type='dag',
        significance_level=significance,
        n_jobs=-1,  # 使用所有可用的CPU核心
    )

    # 打印估计的边
    if isprint:
        print("Estimated edges:")
        print(estimated_model.edges())

    # 将估计的模型转换为邻接矩阵格式
    adj_matrix = np.zeros((dim, dim), dtype=int)
    for edge in estimated_model.edges():
        i = dataset.columns.get_loc(edge[0])
        j = dataset.columns.get_loc(edge[1])
        adj_matrix[i, j] = 1

    # 打印估计的邻接矩阵
    if isprint:
        print("\nEstimated adjacency matrix:")
        print(adj_matrix)

    return adj_matrix


def ExactSearch_merge_feature(dataset):
    def Squared_col(dataset, variables):
        for var in variables:
            dataset[var] = dataset[var] ** 2
        return dataset

    def Cos_col(dataset, variables):
        for var in variables:
            dataset[var] = np.cos(dataset[var])
        return dataset

    def Sin_col(dataset, variables):
        for var in variables:
            dataset[var] = np.sin(dataset[var])
        return dataset

    def merge_dags(dag_dfs):
        if not dag_dfs:
            return pd.DataFrame()  # 如果列表为空，返回一个空的 DataFrame
        
        # 初始化一个与输入的 DataFrame 形状相同的全零 DataFrame
        merged_dag = pd.DataFrame(0, index=dag_dfs[0].index, columns=dag_dfs[0].columns)
        
        # 遍历每个 DAG 数据框，进行逻辑或操作
        for dag_df in dag_dfs:
            merged_dag |= dag_df  # 使用位运算符进行逻辑或操作
            
        return merged_dag
    
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    Raw_dataset = dataset.copy()
    Squared_dataset = Squared_col(dataset, variables)
    Cos_dataset = Cos_col(dataset, variables)
    Sin_dataset = Sin_col(dataset, variables)
    datasets = [Raw_dataset, Squared_dataset, Cos_dataset, Sin_dataset]
    estimate_adj_df_dags = [ExactSearch_estimate(d).astype(int) for d in datasets]
    estimate_adj_df_dag = merge_dags(estimate_adj_df_dags)

    df = []
    for variable in variables:
        # 检查变量与'X'和'Y'之间的边
        v_to_X = estimate_adj_df_dag.loc[variable, 'X']
        X_to_v = estimate_adj_df_dag.loc['X', variable]
        v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
        Y_to_v = estimate_adj_df_dag.loc['Y', variable]
        X_to_Y = estimate_adj_df_dag.loc['X', 'Y']

        df.append({
            "variable": variable,
            "ExactSearch-merge(v,X)": v_to_X,
            "ExactSearch-merge(X,v)": X_to_v,
            "ExactSearch-merge(v,Y)": v_to_Y,
            "ExactSearch-merge(Y,v)": Y_to_v,
            "ExactSearch-merge(X,Y)": X_to_Y
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def ExactSearch_nonlinear_feature(dataset):
    def Squared_col(dataset, variables):
        for var in variables:
            dataset[var] = dataset[var] ** 2
        return dataset

    def Cos_col(dataset, variables):
        for var in variables:
            dataset[var] = np.cos(dataset[var])
        return dataset

    def Sin_col(dataset, variables):
        for var in variables:
            dataset[var] = np.sin(dataset[var])
        return dataset
    
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    Squared_dataset = Squared_col(dataset, variables)
    Cos_dataset = Cos_col(dataset, variables)
    Sin_dataset = Sin_col(dataset, variables)

    nonlinears = ['sq', 'cos', 'sin']
    datasets = [Squared_dataset, Cos_dataset, Sin_dataset]
    estimate_adj_df_dags = [ExactSearch_estimate(d).astype(int) for d in datasets]

    df = []
    for variable in variables:
        result = {"variable": variable}
        for nonlinear, estimate_adj_df_dag in zip(nonlinears, estimate_adj_df_dags):
            # 检查变量与'X'和'Y'之间的边
            v_to_X = estimate_adj_df_dag.loc[variable, 'X']
            X_to_v = estimate_adj_df_dag.loc['X', variable]
            v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
            Y_to_v = estimate_adj_df_dag.loc['Y', variable]
            result[f"ExactSearch(v-{nonlinear},X)"] = v_to_X
            result[f"ExactSearch(X,v-{nonlinear})"] = X_to_v
            result[f"ExactSearch(v-{nonlinear},Y)"] = v_to_Y
            result[f"ExactSearch(Y,v-{nonlinear})"] = Y_to_v
        df.append(result)

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

def FCI_estimate(dataset, alpha=0.05, indep_test='fisherz', kernel=None, 
               depth=-1, max_path_length=-1, verbose=False, show_progress=False):
    """
    使用FCI算法进行因果发现，并应用先验知识。

    参数:
    - dataset (pd.DataFrame): 输入的数据框，包含'X'、'Y'和其他协变量。
    - alpha (float): 显著性水平，默认值为0.05。
    - indep_test (str): 独立性检验方法，默认值为'fisherz'。
    - kernel (str): 核函数类型，默认值为'linear'。
    - verbose (bool): 是否打印详细输出，默认值为False。
    - show_progress (bool): 是否显示算法进度，默认值为False。

    返回:
    - adj_df (pd.DataFrame): 因果图的邻接矩阵，格式为pivot_table。
    """
    
    # 1. 将数据框转换为numpy.ndarray
    data = dataset.values

    # 检验相关系数是否奇异，如果存在多重共线性，对存在多重共线性的变量添加随机扰动
    data = handle_multicollinearity(data)
    
    # 2. 定义先验知识：'X' → 'Y'
    # 创建GraphNode对象
    try:
        node_X = GraphNode('X')
        node_Y = GraphNode('Y')
    except Exception as e:
        raise ValueError("确保数据框中包含名为'X'和'Y'的列。") from e
    
    # 初始化BackgroundKnowledge对象并添加先验知识
    bk = BackgroundKnowledge().add_required_by_node(node_X, node_Y)

    # 3. 配置核参数
    if indep_test == 'kci':
        if kernel is None:
            kernel = 'linear'
        if kernel == 'linear':
            kernel_kwargs = {
                'kernelX': 'Linear', 
                'kernelY': 'Linear', 
                'kernelZ': 'Linear', 
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'polynomial':
            kernel_kwargs = {
                'kernelX': 'Polynomial', 
                'kernelY': 'Polynomial', 
                'kernelZ': 'Polynomial', 
                'polyd': 3,               # 多项式次数设置为3
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'gaussian':
            kernel_kwargs = {
                'kernelX': 'Gaussian', 
                'kernelY': 'Gaussian', 
                'kernelZ': 'Gaussian', 
                'est_width': 'empirical', # 使用经验宽度
                'approx': True,           # 使用伽玛近似
                'nullss': 1000,          # 原假设下模拟的样本量
            }
        elif kernel == 'mix':
            kernel_kwargs = {
                'kernelX': 'Polynomial', 
                'kernelY': 'Polynomial', 
                'kernelZ': 'Gaussian',     # Z使用高斯核
                'polyd': 3,                # 多项式次数设置为3
                'est_width': 'median',     # Z的高斯核带宽使用中位数技巧
                'approx': True,            # 使用伽玛近似
                'nullss': 1000,           # 原假设下模拟的样本量
            }
        else:
            raise ValueError(f'Unknown kernel: {kernel}')
    else:
        kernel_kwargs = {}

    # 4. 运行FCI算法，传入先验知识
    try:
        g, edges = fci(data, 
                alpha=alpha, 
                independence_test_method=indep_test, 
                depth=depth,
                max_path_length=max_path_length,
                background_knowledge=bk, 
                verbose=verbose, 
                show_progress=show_progress,
                **kernel_kwargs
        )

        # 5. 提取邻接矩阵
        adj_matrix = g.graph
    except Exception as e:
        adj_matrix = np.zeros((data.shape[1], data.shape[1]))
    
    # 6. 将邻接矩阵转换为pandas DataFrame，并设置行列索引为原数据框的列名
    adj_df = pd.DataFrame(adj_matrix, index=dataset.columns, columns=dataset.columns)
    
    return adj_df

def FCI_feature(dataset):
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    estimate_adj_df_bidirectional = FCI_estimate(dataset)  # PAG
    estimate_adj_df_dag = estimate_adj_df_bidirectional.astype('int')

    df = []
    for variable in variables:
        # 检查变量与'X'和'Y'之间的边
        v_to_X = estimate_adj_df_dag.loc[variable, 'X']
        X_to_v = estimate_adj_df_dag.loc['X', variable]
        v_to_Y = estimate_adj_df_dag.loc[variable, 'Y']
        Y_to_v = estimate_adj_df_dag.loc['Y', variable]
        X_to_Y = estimate_adj_df_dag.loc['X', 'Y']

        df.append({
            "variable": variable,
            "FCI(v,X)": v_to_X,
            "FCI(X,v)": X_to_v,
            "FCI(v,Y)": v_to_Y,
            "FCI(Y,v)": Y_to_v,
            "FCI(X,Y)": X_to_Y
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df

# 旧代码，提升约一个点

from sklearn.preprocessing import PolynomialFeatures

# 提取特征矩阵
X = X_y_group_train.drop(['variable', 'dataset', 'label', 'y'], axis=1)

# 使用随机森林获取特征重要性
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 13)
model.fit(X_train, y_train)
importances = model.feature_importances_

# 将特征重要性与特征名称对应
feature_importance = pd.Series(importances, index=X.columns)
# 选择重要性排名前10的特征
top_features = feature_importance.nlargest(10).index.tolist()

# 仅对这些特征生成交互项
X_top = X[top_features]

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions_top = poly.fit_transform(X_top)
feature_names_top = poly.get_feature_names_out(input_features=top_features)
X_interactions_top_df = pd.DataFrame(X_interactions_top, columns=feature_names_top)

# 将新特征与原始特征合并
X_train_new = pd.concat([X.reset_index(drop=True), X_interactions_top_df], axis=1)

"""PPS"""
def PPS_feature(dataset):
    variables = dataset.columns.drop(["X", "Y"]).tolist()

    matrix_df = pps_matrix(dataset)
    pivot_df = pd.pivot_table(matrix_df, index='x', columns='y', values='ppscore')

    df = []
    for variable in variables:
        df.append({
            "variable": variable,
            "PPS(v,X)": pivot_df.loc[variable, 'X'],
            "PPS(X,v)": pivot_df.loc['X', variable],
            "PPS(v,Y)": pivot_df.loc[variable, 'Y'],
            "PPS(Y,v)": pivot_df.loc['Y', variable],
            "PPS(X,Y)": pivot_df.loc['X', 'Y'],
            "max(PPS(v,others))": pivot_df.loc[variable, variables].max(),
            "mean(PPS(v,others))": pivot_df.loc[variable, variables].mean(),
        })

    df = pd.DataFrame(df)
    df["dataset"] = dataset.name

    # Reorder columns:
    df = df[["dataset"] + [colname for colname in df.columns if colname != "dataset"]]

    return df
