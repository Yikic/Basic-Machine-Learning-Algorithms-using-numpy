import numpy as np
from utils import DecisionTree


class RandomForest:
    """
    Notes:
        the type of args is the same as class DecisionTree's

    Examples:
        rf = utils.RandomForest(x_train,y_train,number_of_trees=6, max_features=2)
        rf.fit()
    """
    def __init__(self, x, y, number_of_trees=6, max_depth=2, max_features=3):
        self.x = x
        self.y = y
        self.m, self.n = x.shape
        self.max_depth = max_depth
        self.max_features = max_features
        self.number_trees = number_of_trees

    def bootstrap(self):
        index = np.random.randint(self.m, size=self.m)
        x = self.x[index]
        y = self.y[index]
        return x, y

    def fit(self):
        for i in range(self.number_trees):
            x, y = self.bootstrap()
            dt = DecisionTree(x, y, max_feature=self.max_features, if_init=True)
            root_index = np.arange(0, dt.m)
            print("tree {}:".format(i+1))
            dt.tree_recursive(root_index, 'Root', self.max_depth)

