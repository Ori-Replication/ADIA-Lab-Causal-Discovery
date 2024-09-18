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

> **距离相关系数**