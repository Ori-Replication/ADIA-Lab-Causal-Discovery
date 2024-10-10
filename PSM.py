##################################################################
###################  Utility/Helper Functions  ###################
##################################################################
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def get_propensity_scores(model, data, verbose = False):
    '''
    基于指定的逻辑回归模型计算倾向评分

    Parameters
    ----------
    model : string
        a model specification in the form Y ~ X1 + X2 + ... + Xn
    data : Pandas DataFrame
        the data used to calculate propensity scores
    verbose : boolean
        verbosity of the model output

    Returns
    -------
    An array of propensity scores.
    '''
    glm_binom = smf.glm(formula = model, data = data, family = sm.families.Binomial())
    result = glm_binom.fit()
    if verbose:
        print(result.summary)
    return result.fittedvalues

def flatten_match_ids(df):
    '''
    Converts a Pandas DataFrame of matched IDs into a list of those IDs.

    Parameters
    ----------
    df : Pandas Dataframe
        a dataframe consisting of 1 column of treated/case IDs and n columns
        of respective control(s) matched

    Returns
    -------
    A list of treated case and matched control IDs.
    '''
    master_list = []
    master_list.append(df[df.columns[0]].tolist())
    for i in range(1, df.shape[1]):
        master_list.append(df[df.columns[i]].tolist())
    master_list = [item for sublist in master_list for item in sublist]
    return master_list

def get_matched_data(match_ids, raw_data):
    '''
    Subsets the raw data to include data only from the treated cases and
    their respective matched control(s).

    Parameters
    ----------
    match_ids : Pandas DataFrame
        a dataframe of treated case IDs and matched control(s)
    raw_data: Pandas DataFrame
        a dataframe of all of the raw data

    Returns
    -------
    A dataframe containing data only from treated cases and matched control(s).
    '''
    match_ids = flatten_match_ids(match_ids)
    matched_data = raw_data[raw_data.index.isin(match_ids)]
    return matched_data

def evaluate_KL_divergence(y_t1, y_t0, num_bins=30, epsilon=1e-10):
    """
    计算 y_t1 和 y_t0 表示的概率分布之间的 KL 散度。
    
    参数:
    - y_t1: numpy 数组，表示处理组的数据，范围在 [-1, 1]
    - y_t0: numpy 数组，表示对照组的数据，范围在 [-1, 1]
    - num_bins: int，直方图的分箱数量
    - epsilon: float，为避免零概率而添加的平滑项
    
    返回:
    - KL 散度的值
    """
    # 设置固定的分箱边界，从 -1 到 1
    bins = np.linspace(-1, 1, num_bins + 1)
    
    # 计算直方图
    hist_t1, _ = np.histogram(y_t1, bins=bins, density=True)
    hist_t0, _ = np.histogram(y_t0, bins=bins, density=True)
    
    # 将密度转换为概率分布
    bin_width = bins[1] - bins[0]
    p = hist_t1 * bin_width
    q = hist_t0 * bin_width
    
    # 添加平滑项以避免零概率
    p += epsilon
    q += epsilon
    
    # 重新归一化
    p /= p.sum()
    q /= q.sum()
    
    # 计算 KL 散度
    KL = stats.entropy(p, q)
    
    return KL

####################################################
###################  Base Class  ###################
####################################################

class PSMatch(object):
    '''
    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to be used for propensity score matching.
    model : string
        The model specification for calculating propensity scores; in the format T ~ X1 + X2 + ... + Xn
    k : string
        The number of controls to be matched to each treated case.
    '''

    def __init__(self, dataset, Y_col, T_col, k):
        self.dataset = dataset.rename(columns={col: f'col_{col}' for col in dataset.columns})
        self.Y_col = f'col_{Y_col}'
        self.T_col = f'col_{T_col}'
        self.X_col = [f'col_{col}' for col in dataset.columns if col not in [Y_col, T_col]]
        self.model = f"{self.T_col} ~ {' + '.join(self.X_col)}"
        self.k = int(k)

    def prepare_data(self, **kwargs):
        '''
        计算倾向得分

        Returns
        -------
        A Pandas DataFrame containing raw data plus a column of propensity scores.
        '''
        df = self.dataset.copy()
        # 生成ID列
        df['ID'] = range(1, len(df) + 1)
        df = df.set_index('ID')
        # 将T_col从连续变量处理成0/1变量，按照85%和50%划分
        up_threshold = df[self.T_col].quantile(0.85)
        down_threshold = df[self.T_col].quantile(0.50)
        df['CASE'] = df[self.T_col].apply(lambda x: '处理组' if x >= up_threshold else ('控制组' if x < down_threshold else '剔除'))
        df = df[df['CASE'] != '剔除']
        # 更新 T_col 的值
        df[self.T_col] = df['CASE'].apply(lambda x: int(1) if x == '处理组' else int(0))
        # 计算倾向得分
        print("\nCalculating propensity scores ...", end = " ")
        propensity_scores = get_propensity_scores(model = self.model, data = df, verbose = False)
        print("Preparing data ...", end = " ")
        df["PROPENSITY"] = propensity_scores
        self.df = df

    def match(self, caliper = None, **kwargs):
        '''
        执行倾向得分匹配

        Returns
        -------
        matches : Pandas DataFrame
            the Match object attribute describing which control IDs are matched
            to a particular treatment case.
        matched_data: Pandas DataFrame
            the Match object attribute containing the raw data for only treatment
            cases and their matched controls.
        '''
        # 检查df是否已经初始化
        if not hasattr(self, 'df'):
            raise AttributeError("%s does not have a 'df' attribute." % (self))

        # 获取倾向得分
        groups = self.df[f'{self.T_col}']
        propensity = self.df['PROPENSITY']
        # 处理组
        g_t1 = groups[groups==1].index
        p_t1 = propensity[groups==1]
        # 控制组
        g_t0 = groups[groups==0].index
        p_t0 = propensity[groups==0]

        # 随机打乱T=1组的索引ID
        m_order = list(np.random.permutation(g_t1))
        matches = {}
        k = int(self.k)

        # 基于倾向得分差异进行匹配
        print("\nMatching [" + str(k) + "] controls to each case ... ", end = " ")
        for m in m_order:
            # 计算得分差异
            difference = abs(p_t1[m]-p_t0)
            difference_array = np.array(difference)
            # 选取k个最小的得分差异
            k_smallest = np.partition(difference_array, k)[:k].tolist()
            if caliper:  # 如果设置了阈值
                caliper = float(caliper)
                keep_diffs = [i for i in k_smallest if i <= caliper]
                keep_ids = difference[difference.isin(keep_diffs)].index.tolist()
            else:
                keep_ids = difference.nsmallest(k).index.tolist()

            # 如果匹配数大于 k，随机选择 k 个
            if len(keep_ids) > k:
                selected_ids = list(np.random.choice(keep_ids, k, replace=False))
                matches[m] = selected_ids
            elif len(keep_ids) < k:
                matches[m] = keep_ids.copy()
                while len(matches[m]) < k:
                    matches[m].append("NA")
            else:
                matches[m] = keep_ids.copy()

            # 只 drop 掉最小差异的控制组数据
            # 首先，确保当前匹配不包含 "NA"
            valid_matches = [ctrl for ctrl in matches[m] if ctrl != "NA"]
            if valid_matches:
                # 计算当前匹配中每个控制组的差异
                current_differences = difference[valid_matches]
                # 找到差异最小的控制组 ID
                best_control = current_differences.idxmin()
                # 从控制组池中删除该控制组
                g_t0 = g_t0.drop(best_control)
                p_t0 = p_t0.drop(best_control)

        # 将匹配结果转换为DataFrame
        matches = pd.DataFrame.from_dict(matches, orient="index")
        matches = matches.reset_index()
        column_names = {}
        column_names["index"] = "ID"
        for i in range(k):
            column_names[i] = str("CONTROL_MATCH_" + str(i+1))
        matches = matches.rename(columns = column_names)

        # 根据匹配结果获取匹配数据
        matched_data = get_matched_data(matches, self.df)
        self.matches = matches
        self.matched_data = matched_data

    def evaluate(self, **kwargs):
        '''
        Conducts chi-square tests to verify statistically that the cases/controls
        are well-matched on the variables of interest.
        '''
        # 检查是否已进行匹配
        if not hasattr(self, 'matches'):
            raise AttributeError("%s does not have a 'matches' attribute." % (self))
        if not hasattr(self, 'matched_data'):
            raise AttributeError("%s does not have a 'matched_data' attribute." % (self))

        matched_data = self.matched_data
        y_t1 = matched_data[matched_data[f'{self.T_col}'] == 1][f'{self.Y_col}'].values
        y_t0 = matched_data[matched_data[f'{self.T_col}'] == 0][f'{self.Y_col}'].values

        # 确保y_t1和y_t0的长度相同，如果不等，剪短为相同长度
        if len(y_t1) > len(y_t0):
            # 修改为随机抽取相同长度的样本
            y_t1 = np.random.choice(y_t1, len(y_t0), replace=False)
        elif len(y_t0) > len(y_t1):
            y_t0 = np.random.choice(y_t0, len(y_t1), replace=False)

        # 确保 y_t1 和 y_t0 在 [-1, 1] 范围内
        y_t1 = np.clip(y_t1, -1, 1)
        y_t0 = np.clip(y_t0, -1, 1)
    
        # 计算 KL 散度
        KL_divergence = evaluate_KL_divergence(y_t1, y_t0)

        return KL_divergence

    def run(self, **kwargs):
        self.prepare_data()
        self.match()
        KL = self.evaluate()
        return KL
    

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    X_train = pd.read_pickle('./data/X_train.pickle')
    y_train = pd.read_pickle('./data/y_train.pickle')
    print(len(X_train), len(y_train))

    # Example usage
    A = X_train['00000'].copy()
    B = y_train['00000'].copy()

    match = PSMatch(A, 'X', '0', 3)
    KL = match.run()
    print(KL)