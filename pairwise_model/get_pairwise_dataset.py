import pandas as pd
import pickle
from tqdm import tqdm

def DAG_to_biDAG(dag):
    """
    将单向DAG转换为双向DAG。
    
    参数:
        dag (pd.DataFrame): 输入的单向DAG，行和列均为节点标识，[i,j]=1表示i->j，0表示无关系。
    
    返回:
        pd.DataFrame: 转换后的双向DAG，[i,j]=1表示i->j，[j,i]=-1表示j->i，0表示无关系。
    """
    # 创建DAG的副本，避免修改原始数据
    bidag = dag.copy()
    
    # 找到所有存在因果关系的边
    edges = dag.stack()[dag.stack() == 1].index.tolist()
    
    # 为每条边设置双向关系
    for i, j in edges:
        bidag.at[i, j] = 1   # 保持原有的因果关系
        bidag.at[j, i] = -1  # 设置反向关系为-1
    
    return bidag

def get_CEPCDataset(X_dict, y_dict=None):
    """
    将X_dict和y_dict转换为CEPC格式的数据框。

    参数:
    - X_dict (dict): 键为数据集名称，值为包含'X'、'Y'和其他变量的DataFrame。
    - y_dict (dict or None): 键为数据集名称，值为表示因果关系的DAG DataFrame。如果为None，则仅生成变量对，不包含因果方向。

    返回:
    - 如果y_dict不为None:
        - df (pd.DataFrame): 包含'A', 'B', 'key', 'variable'的DataFrame。
        - targets (pd.Series): 'Target'列。
    - 如果y_dict为None:
        - df (pd.DataFrame): 包含'A', 'B', 'key', 'variable'的DataFrame。
    """
    df = []
    if y_dict != None:
        for key in tqdm(X_dict.keys()):
            dataset = X_dict[key]
            dag = y_dict[key]
            bidag = DAG_to_biDAG(dag)

            parents = dataset.columns.tolist()
            children = dataset.columns.tolist()
            for parent in parents:
                for child in children:
                    df.append([dataset[parent].values, dataset[child].values, bidag.loc[parent, child], key, parent, child])
       
        df = pd.DataFrame(df, columns=['A', 'B', 'Target', 'dataset', 'A_parent', 'B_child'])
        df.index = ['pair' + str(i+1) for i in range(len(df))]
        return df[['A', 'B', 'dataset', 'A_parent', 'B_child']], df[['Target']]
    
    else:
        df = []
        for key in tqdm(X_dict.keys()):
            dataset = X_dict[key]

            variables = dataset.columns.drop(["X", "Y"]).tolist()
            for variable in variables:
                df.append([dataset[variable].values, dataset["X"].values, key, variable, 'X'])
                df.append([dataset['X'].values, dataset[variable].values, key, 'X', variable])
                df.append([dataset[variable].values, dataset["Y"].values, key, variable, 'Y'])
                df.append([dataset['Y'].values, dataset[variable].values, key, 'Y', variable])

        df = pd.DataFrame(df, columns=['A', 'B', 'dataset', 'A_parent', 'B_child'])
        df.index = ['pair' + str(i+1) for i in range(len(df))]
        return df[['A', 'B', 'dataset', 'A_parent', 'B_child']]
    
if __name__ == "__main__":
    X_train = pd.read_pickle('../data/X_train.pickle')
    y_train = pd.read_pickle('../data/y_train.pickle')
    print(len(X_train), len(y_train))

    CEPCDataset, CEPCTarget = get_CEPCDataset(X_train, y_train)
    print(CEPCDataset.shape, CEPCTarget.shape)

    with open('../mid_data/CEPCDataset.pkl', 'wb') as f:
        pickle.dump(CEPCDataset, f)

    with open('../mid_data/CEPCTarget.pkl', 'wb') as f:
        pickle.dump(CEPCTarget, f)
    print('Done!')