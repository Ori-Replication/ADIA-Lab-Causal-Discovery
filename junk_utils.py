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