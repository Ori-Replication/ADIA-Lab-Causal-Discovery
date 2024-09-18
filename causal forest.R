# 设置随机种子以确保结果可重复
set.seed(1)

# 加载必要的包
library(grf)
library(MASS)  # 包含Boston数据集
library(ggplot2)

# 加载数据
data(Boston)

# 准备数据
Y <- Boston$medv  # 因变量：房价中位数
W <- as.numeric(Boston$rm > median(Boston$rm))  # 处理变量：房间数是否高于中位数
X <- Boston[, !(names(Boston) %in% c("medv", "rm"))]  # 其他变量作为控制变量
X <- as.matrix(X)  # 确保X是一个矩阵

# 构建回归森林
Y.forest <- regression_forest(X, Y)
Y.hat <- predict(Y.forest)$predictions
W.forest <- regression_forest(X, W)
W.hat <- predict(W.forest)$predictions

# 构建初始因果森林
cf.raw <- causal_forest(X, Y, W, Y.hat = Y.hat, W.hat = W.hat)

# 变量重要性
varimp <- variable_importance(cf.raw)
selected.idx <- which(varimp > mean(varimp))

# 构建最终因果森林
cf <- causal_forest(X[, selected.idx, drop = FALSE], Y, W,
                    Y.hat = Y.hat, W.hat = W.hat,
                    tune.parameters = "all")

# 预测个体处理效应
tau.hat <- predict(cf)$predictions

# 估计平均处理效应
ATE <- average_treatment_effect(cf)
CI <- paste("95% CI for the ATE:", round(ATE[1], 3),
            "+/-", round(qnorm(0.975) * ATE[2], 3))
print(CI)

# 绘制CATE直方图
hist(tau.hat, xlab = "Estimated CATE", main = "Distribution of Estimated Causal Effects")

# 查看变量重要性
var_importance <- variable_importance(cf)
sorted_importance <- sort(var_importance, decreasing = TRUE)
print(sorted_importance)

# 异质性测试
test_calibration(cf)

# 可视化处理效应与一个重要特征的关系
important_feature <- names(X)[which.max(var_importance)]
plot_data <- data.frame(tau = tau.hat, feature = X[, important_feature])
ggplot(plot_data, aes(x = feature, y = tau)) +
  geom_point() +
  geom_smooth(method = "loess") +
  labs(x = important_feature, y = "Estimated Treatment Effect",
       title = paste("Treatment Effect vs", important_feature))
