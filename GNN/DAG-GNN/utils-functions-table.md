| 函数名 | 中文简称 | 输入 | 输出 | 功能详述 |
|--------|----------|------|------|----------|
| simulate_random_dag | 随机DAG生成 | d: 节点数, degree: 期望节点度, graph_type: 图类型, w_range: 权重范围 | nx.DiGraph | 生成具有指定节点数和期望度的随机有向无环图(DAG) |
| simulate_sem | SEM模拟 | G: 加权DAG, n: 样本数, x_dims: 维度, sem_type: SEM类型, linear_type: 线性类型, noise_scale: 噪声尺度 | np.ndarray | 从指定类型的结构方程模型(SEM)中模拟样本 |
| simulate_population_sample | 总体样本模拟 | W: 邻接矩阵, Omega: 噪声协方差矩阵 | np.ndarray | 模拟匹配总体最小二乘的数据矩阵X |
| count_accuracy | 精度计算 | G_true: 真实图, G: 预测图, G_und: 预测无向边 | tuple | 计算B或CPDAG B + B_und的FDR、TPR、FPR |
| my_softmax | 自定义softmax | input: 输入张量, axis: 轴 | Tensor | 在指定轴上应用softmax函数 |
| binary_concrete | 二元具体化 | logits: 对数几率, tau: 温度, hard: 是否硬化, eps: 极小值 | Variable | 应用二元具体化relaxation |
| binary_concrete_sample | 二元具体化采样 | logits: 对数几率, tau: 温度, eps: 极小值 | Tensor | 从二元具体化分布中采样 |
| sample_logistic | 逻辑分布采样 | shape: 形状, eps: 极小值 | Tensor | 从逻辑分布中采样 |
| sample_gumbel | Gumbel分布采样 | shape: 形状, eps: 极小值 | Tensor | 从Gumbel(0, 1)分布中采样 |
| gumbel_softmax_sample | Gumbel-Softmax采样 | logits: 对数几率, tau: 温度, eps: 极小值 | Tensor | 从Gumbel-Softmax分布中抽取样本 |
| gumbel_softmax | Gumbel-Softmax | logits: 对数几率, tau: 温度, hard: 是否离散化, eps: 极小值 | Variable | 从Gumbel-Softmax分布中采样并可选择离散化 |
| gauss_sample_z | 高斯采样Z | logits: 对数几率, zsize: Z维度 | Tensor | 从高斯分布中采样Z |
| gauss_sample_z_new | 新高斯采样Z | logits: 对数几率, zsize: Z维度 | Tensor | 从高斯分布中采样Z的新版本 |
| binary_accuracy | 二分类准确率 | output: 输出, labels: 标签 | float | 计算二分类问题的准确率 |
| list_files | 文件列表 | directory: 目录, extension: 扩展名 | generator | 列出指定目录下具有特定扩展名的文件 |
| read_BNrep | 读取BN数据 | args: 参数 | tuple | 从BN存储库加载结果 |
| load_data_discrete | 加载离散数据 | args: 参数, batch_size: 批量大小, suffix: 后缀, debug: 调试模式 | tuple | 加载离散数据并创建数据加载器 |
| load_data | 加载数据 | args: 参数, batch_size: 批量大小, suffix: 后缀, debug: 调试模式 | tuple | 加载数据并创建数据加载器 |
| to_2d_idx | 转换为2D索引 | idx: 一维索引, num_cols: 列数 | tuple | 将一维索引转换为二维索引 |
| encode_onehot | one-hot编码 | labels: 标签 | np.ndarray | 将标签转换为one-hot编码 |
| get_triu_indices | 获取上三角索引 | num_nodes: 节点数 | Tensor | 获取上三角矩阵的线性索引 |
| get_tril_indices | 获取下三角索引 | num_nodes: 节点数 | Tensor | 获取下三角矩阵的线性索引 |
| get_offdiag_indices | 获取非对角索引 | num_nodes: 节点数 | Tensor | 获取非对角元素的线性索引 |
| get_triu_offdiag_indices | 获取上三角非对角索引 | num_nodes: 节点数 | Tensor | 获取上三角非对角元素的线性索引 |
| get_tril_offdiag_indices | 获取下三角非对角索引 | num_nodes: 节点数 | Tensor | 获取下三角非对角元素的线性索引 |
| get_minimum_distance | 获取最小距离 | data: 数据 | Tensor | 计算数据点之间的最小距离 |
| get_buckets | 获取桶 | dist: 距离, num_buckets: 桶数 | tuple | 将距离划分为指定数量的桶 |
| get_correct_per_bucket | 获取每个桶的正确数 | bucket_idx: 桶索引, pred: 预测, target: 目标 | list | 计算每个桶中的正确预测数 |
| get_correct_per_bucket_ | 获取每个桶的正确数(改进版) | bucket_idx: 桶索引, pred: 预测, target: 目标 | list | 计算每个桶中的正确预测数(改进版) |
| kl_categorical | 分类KL散度 | preds: 预测, log_prior: 先验对数, num_atoms: 原子数, eps: 极小值 | Tensor | 计算分类分布的KL散度 |
| kl_gaussian | 高斯KL散度 | preds: 预测, zsize: Z维度 | Tensor | 计算高斯分布的KL散度 |
| kl_gaussian_sem | SEM高斯KL散度 | preds: 预测 | Tensor | 计算SEM中高斯分布的KL散度 |
| kl_categorical_uniform | 均匀分类KL散度 | preds: 预测, num_atoms: 原子数, num_edge_types: 边类型数, add_const: 是否添加常数, eps: 极小值 | Tensor | 计算相对于均匀分布的分类KL散度 |
| nll_catogrical | 分类负对数似然 | preds: 预测, target: 目标, add_const: 是否添加常数 | Tensor | 计算离散变量的负对数似然 |
| nll_gaussian | 高斯负对数似然 | preds: 预测, target: 目标, variance: 方差, add_const: 是否添加常数 | Tensor | 计算高斯分布的负对数似然 |
| normalize_adj | 邻接矩阵归一化 | adj: 邻接矩阵 | Tensor | 对称归一化邻接矩阵 |
| preprocess_adj | 邻接矩阵预处理 | adj: 邻接矩阵 | Tensor | 预处理邻接矩阵 |
| preprocess_adj_new | 新邻接矩阵预处理 | adj: 邻接矩阵 | Tensor | 新版本的邻接矩阵预处理 |
| preprocess_adj_new1 | 新邻接矩阵预处理1 | adj: 邻接矩阵 | Tensor | 另一种新版本的邻接矩阵预处理 |
| isnan | 检查NaN | x: 输入 | Tensor | 检查输入是否为NaN |
| my_normalize | 自定义归一化 | z: 输入 | Tensor | 对输入进行自定义归一化 |
| sparse_to_tuple | 稀疏矩阵转元组 | sparse_mx: 稀疏矩阵 | tuple | 将稀疏矩阵转换为元组表示 |
| matrix_poly | 矩阵多项式 | matrix: 矩阵, d: 度 | Tensor | 计算矩阵的d次多项式 |
| A_connect_loss | A连接损失 | A: 邻接矩阵, tol: 容差, z: 辅助变量 | Tensor | 计算确保A至少连接到另一个父节点的损失 |
| A_positive_loss | A正值损失 | A: 邻接矩阵, z_positive: 正值辅助变量 | Tensor | 计算确保每个A_ij > 0的损失 |
| compute_BiCScore | 计算BIC分数 | G: 图, D: 数据 | float | 计算贝叶斯信息准则(BIC)分数 |
| compute_local_BiCScore | 计算局部BIC分数 | np_data: 数据, target: 目标, parents: 父节点 | float | 计算局部贝叶斯信息准则(BIC)分数 |
