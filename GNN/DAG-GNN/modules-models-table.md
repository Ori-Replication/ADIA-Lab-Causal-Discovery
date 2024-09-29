| 模型名 | 中文简称 | 属性 | 方法 | 功能详述 |
|--------|----------|------|------|----------|
| MLPEncoder | MLP编码器 | adj_A, factor, Wa, fc1, fc2, dropout_prob, batch_size, z, z_positive | init_weights, forward | 多层感知机编码器,用于将输入编码为隐藏表示和邻接矩阵 |
| MLPDEncoder | 离散MLP编码器 | adj_A, factor, Wa, fc1, fc2, embed, dropout_prob, alpha, batch_size, z, z_positive | init_weights, forward | 用于离散数据的多层感知机编码器,包含嵌入层 |
| SEMEncoder | SEM编码器 | factor, adj_A, dropout_prob, batch_size | init_weights, forward | 结构方程模型(SEM)编码器,直接学习邻接矩阵 |
| MLPDDecoder | 离散MLP解码器 | bn0, out_fc1, out_fc2, out_fc3, bn1, batch_size, data_variable_size, dropout_prob | init_weights, forward | 用于离散数据的多层感知机解码器(旧版,不推荐使用) |
| MLPDiscreteDecoder | 离散MLP解码器 | bn0, out_fc1, out_fc2, out_fc3, bn1, batch_size, data_variable_size, softmax, dropout_prob | init_weights, forward | 用于离散数据的多层感知机解码器,输出经过softmax |
| MLPDecoder | MLP解码器 | out_fc1, out_fc2, batch_size, data_variable_size, dropout_prob | init_weights, forward | 通用多层感知机解码器 |
| SEMDecoder | SEM解码器 | batch_size, data_variable_size, dropout_prob | forward | 结构方程模型(SEM)解码器,直接使用学习的邻接矩阵重构输入 |
