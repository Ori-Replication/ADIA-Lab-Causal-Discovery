https://hub.crunchdao.com/competitions/causality-discovery/submit/via/notebook?projectName=default

## 特征工程
### Baseline
#### pearson_correlation

**Output**

- for v in variables：

  - corr(v,X)

  - corr(v,Y)

  - max(corr(v, others))

  - min(corr(v, others))

  - mean(corr(v, others))

  - std(corr(v, others))


- corr(X,Y)

**Note**

- **关系强度**：皮尔逊相关系数提供了变量之间线性关系的强度和方向，范围从-1到1。
- **直观解释**：相关系数易于理解和解释，为初步的数据探索提供了直观的指标。
- **标准化**：相关系数不受原始数据单位的影响，便于不同尺度变量间的比较。
- **缺陷**：相关不等于因果，线性、缺乏方向性。

> **Pearson相关系数**
>
> 对于变量 $X$ 和 $Y$ ，Pearson相关系数 $r$ 的计算公式为：
> $$
> r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}} = \frac{\sigma_{xy}}{\sigma_x \sigma_y}
> $$
> 其中： $\bar{x}$ 和 $\bar{y}$ 分别是 $X$ 和 $Y$ 的平均值；$\sigma_{xy}$ 是 $X$ 和 $Y$ 的协方差，$\sigma_x$ 是 $X$ 的标准差，$\sigma_y$ 是 $Y$ 的标准差。

#### ttest

**Output**

- for v in variables：

  - ttest(v,X)

  - pvalue(ttest(v,X))<=0.05：T or F

  - ttest(v,Y)

  - pvalue(ttest(v,Y))<=0.05：T or F


- ttest(X,Y)
- pvalue(ttest(X,Y))<=0.05：T or F

**Note**

- **差异检测**：配对t检验可以帮助我们识别变量之间是否存在显著的系统性差异。这种差异可能暗示了潜在的因果关系，但并不能直接证明因果性。
- **统计显著性**：t检验提供了p值，帮助我们判断观察到的差异是否具有统计显著性，这是建立因果关系的必要（但非充分）条件。
- **缺陷**：相关不等于因果，线性、正态假设、缺乏方向性。

> **配对t检验**
>
> 对于两个配对样本 $X$ 和 $Y$ ，t-统计量的计算公式为：
> $$
> t = \frac{\bar{d}}{\frac{s_d}{\sqrt{n}}}
> $$
> 其中：$\bar{d}$ 是差值的平均数（$\bar{d} = \frac{1}{n}\sum_{i=1}^n (X_i - Y_i)$），$s_d$ 是差值的标准差，$n$ 是样本数量。

#### mutual_information

**Output**

- for v in variables：
  - MI(v,X)
  - MI(v,Y)
  - max(MI(v, others))
  - min(MI(v, others))
  - mean(MI(v, others))
  - std(MI(v, others))
- MI(X,Y)

**Note**

- **非线性关系**：互信息可以捕捉变量之间的非线性关系，这是对皮尔逊相关系数的重要补充。
- **独立性检验**：互信息为0当且仅当两个变量统计独立，这提供了一种检测变量间依赖关系的方法。
- **缺陷**：缺乏方向性、计算复杂度较高、对样本量敏感（小样本可能导致过高估计）。

> **互信息** 
>
> 对于连续随机变量 $X$ 和 $Y$ ，互信息的计算公式为：
> $$
> I(X;Y) = \int_Y \int_X p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right) dx dy
> $$
> 其中：$p(x,y)$ 是 $X$ 和 $Y$ 的联合概率分布，$p(x)$ 和 $p(y)$ 分别是 $X$ 和 $Y$ 的边缘概率分布。

### 0917

#### dimension_feature

**Output**

- for v in variables：
  - dimension
  - square_dimension

#### spearman_correlation

**Output**

- for v in variables：

  - spearman_corr(v,X)

  - spearman_corr(v,Y)

- spearman_corr(X,Y)

**Note**

> **斯皮尔曼相关系数**
>
> 对于变量 $X$ 和 $Y$，斯皮尔曼相关系数 $ρ$ 的计算公式为：
> $$
> \rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}
> $$
> 其中：$d_i$ 是第 $i$ 个观察值在 $X$ 和 $Y$ 中的秩次差，$n$ 是样本量。

#### kendall_correlation

**Output**

- for v in variables：
  - kendall_corr(v,X)
  - kendall_corr(v,Y)
- kendall_corr(X,Y)

**Note**

> **肯德尔相关系数**
>
> 对于变量 $X$ 和 $Y$，肯德尔相关系数 $τ$ 的计算公式为：
> $$
> \tau = \frac{2(n_c - n_d)}{n(n-1)}
> $$
> 其中：$n_c$ 是一致对的数量，$n_d$ 是不一致对的数量，$n$ 是样本量。

#### distance_correlation

**Output**

- for v in variables：
  - dcor(v,X)
  - dcor(v,Y)
- dcor(X,Y)

**Note**

- **独立性度量**：距离相关系数为0当且仅当两个变量是独立的，这克服了皮尔逊相关系数的弱点。
- **非线性关系检测**：能够检测任意类型的依赖关系，包括非线性和非单调关系。

> **距离相关系数**
>
> 对于变量 $X$ 和 $Y$，距离相关系数的DC样本估计值为：
> $$
> dcorr(x,y) = \frac{dcov(x,y)}{\sqrt{dcov(x,x)dcov(y,y)}}
> $$
> 其中：$dcov^2(x,y) = Ŝ_1 + Ŝ_2 - 2Ŝ_3$。

### 0918

#### linear_regression_feature_v2

#### conditional_mutual_information

## Causal Learn

### 检验方法
#### Fisher's Z 检验（fisherz）
1. **相关系数计算**: 对于两个变量 $X$ 和 $Y$，在给定条件变量集 $S$ 下，计算其偏相关系数 $r_{XY⋅S}$。
2. **Fisher Z 变换**: 将偏相关系数 $r$ 转换为 Fisher's Z 统计量：$Z=\frac{1}{2}ln⁡(\frac{1+r}{1−r})$
3. **检验统计量**: 统计量 $Z$ 服从标准正态分布。根据 $Z$ 值计算其对应的$p$值，用于判断 $X$ 和 $Y$ 在条件集 $S$ 下是否独立。
#### 核方法独立性检验（kci）
1. **条件独立性的定义**: 在统计学中，给定一个条件集 $Z$，如果变量 $X$ 和 $Y$ 在 $Z$ 下独立，记作 $X \perp Y | Z$，则表明在控制了 $Z$ 的影响后， $X$ 和 $Y$ 之间不存在统计上的关联。
2. **核方法**: 通过将数据映射到高维的特征空间（通常是希尔伯特空间）来捕捉数据的非线性结构。
   - 高斯核（Gaussian Kernel）: $k(x,y) = \exp \left( -\frac{\|x-y\|^2}{2\sigma^2} \right)$
   - 线性核（Linear Kernel）: $k(x,y) = x^\top y$
   - 多项式核（Polynomial Kernel）: $k(x,y) = (\gamma x^\top y + c)^d$
3. KCI检验基于**条件交叉协方差**（Conditional Cross Covariance）的概念，该方法通过将变量 $X$ 、 $Y$ 和 $Z$ 映射到希尔伯特空间中的特征映射，然后评估 $X$ 和 $Y$ 在给定 $Z$ 的条件下是否独立。具体步骤如下：
   - **特征映射**: 使用核函数将原始数据映射到高维特征空间。
   - **条件交叉协方差**: 计算 $X$ 和 $Y$ 在给定 $Z$ 的条件下的交叉协方差。
   - **统计量计算**: 基于交叉协方差计算检验统计量，该统计量衡量 $X$ 和 $Y$ 在给定 $Z$ 下的依赖程度。
   - **显著性检验**: 通过置换检验或渐近分布方法计算 $p$ 值，以决定是否拒绝独立性假设。

4. 设有随机变量 $X$, $Y$ 和 $Z$，KCI检验的目标是检验 $X \perp Y | Z$。假设我们有 $n$ 个观测样本 $\{(x_i, y_i, z_i)\}_{i=1}^n$。

   - 核矩阵构建:

     - $K: X$ 的核矩阵, $K_{ij} = k_X(x_i, x_j)$

     - $L: Y$ 的核矩阵, $L_{ij} = k_Y(y_i, y_j)$

     - $M: Z$ 的核矩阵, $M_{ij} = k_Z(z_i, z_j)$

   - 中心化:
     对核矩阵进行中心化处理，去除均值的影响:

   $$
   \tilde{K} = HKH, \quad \tilde{L} = HLH, \quad \tilde{M} = HMH
   $$

   ​       其中 $H = I_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n^\top$ 为中心化矩阵。

   - 条件交互差:
     通过线性回归将 $X$ 和 $Y$ 对 $Z$ 进行回归，得到残差:

   $$
   \hat{K} = \tilde{K} - \tilde{M}(\tilde{M})^+\tilde{K}
   $$

   $$
   \hat{L} = \tilde{L} - \tilde{M}(\tilde{M})^+\tilde{L}
   $$

   ​        其中 $(\tilde{M})^+$ 表示 $\tilde{M}$ 的伪逆。

   - 检验统计量:
     计算 $\hat{K}$ 和 $\hat{L}$ 的Trace乘积作为检验统计量:

   $$
   \text{Statistic} = \text{Trace}(\hat{K}\hat{L})
   $$

   ​       该统计量在 $X \perp Y | Z$ 的原假设下趋近于零，统计量越大，表明 $X$ 和 $Y$ 在 $Z$ 条件下越不独立。

   - p值计算:
     通过置换检验或使用渐近分布，计算统计量对应的 p 值，以决定是否拒绝独立性假设。

### Constrain-Based

#### PC

- 显著性水平：0.05

