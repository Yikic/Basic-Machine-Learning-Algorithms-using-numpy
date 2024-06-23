from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np

# 加载数据集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 转换数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义参数
params = {
    'objective': 'reg:squarederror',  # 回归问题
    'max_depth': 6,                   # 树的最大深度
    'eta': 0.1,                       # 学习率
    'eval_metric': 'rmse'             # 评价指标
}

# 训练模型
num_round = 100  # 迭代次数
bst = xgb.train(params, dtrain, num_round)

# 进行预测
preds = bst.predict(dtest)

plt.plot(np.sort(y_test), 'o')
plt.plot(np.sort(preds), 'x')
plt.show()

# 计算均方误差
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse}")

