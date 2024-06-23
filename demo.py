from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from tree_series import xgboost
# import dataset
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split


data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.3, random_state=42)

xgb = xgboost.Xgboost(X_train, y_train, epsilon=0.3, max_trees=100, max_depth=6, eta=0.1)
xgb.fit()
pred = xgb.predict(X_test)
plt.plot(np.sort(y_test), 'o')
plt.plot(np.sort(pred), 'x')
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"RMSE: {rmse}")


