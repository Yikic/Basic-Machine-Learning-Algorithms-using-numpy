import numpy as np
import random
random.seed(42)


class TreeNode:
    def __init__(self, feature, value, left=None, right=None, w=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.w = w


class Xgboost:
    
    def __init__(self, x, y, epsilon=0.3, Lambda=1.0, gamma=0, max_depth=6, eta=0.1, max_trees=100, max_bin=32):
        self.x = x
        self.x_sorted = np.argsort(x, axis=0)
        self.epsilon = epsilon  # useful in choosing bins
        self.y = y
        self.true_y = y
        self.first_score = y.mean()
        self.init_w = y.mean()
        self.g_h = np.zeros((len(y), 2))
        self.com_g_h(self.first_score, np.arange(len(y)))
        self.Lambda = Lambda
        self.gamma = gamma
        self.max_depth = max_depth
        self.eta = eta  # learning rate
        self.max_trees = max_trees
        self.max_bin = max_bin
        self.trees = {f'tree{i}': None for i in range(self.max_trees)}

    def fit(self):
        depth = 0
        tree = 0
        while tree < self.max_trees:
            print(f"Tree {tree}:")
            idx, features = self.boostrap()
            self.trees[f'tree{tree}'] = self.split(idx, features, depth, 'Root', self.init_w)
            print('\n')
            self.y = self.true_y - (self.internal_predict(self.x, tree) + self.first_score)
            self.init_w = self.y.mean()
            self.com_g_h(self.init_w, np.arange(len(self.y)))
            tree += 1

    def split(self, idx, features, depth, direction, score):
        if depth >= self.max_depth or len(np.unique(idx)) <= 2:
            print(f"{'|   ' * depth}{direction}: {score}")
            return score
        split_out = self.com_l_w(idx, features)
        if not split_out:
            print(f"{'|   ' * depth}{direction}: {score}")
            return score
        feature_best, split_value, left_point, right_point, w_left, w_right = split_out
        w_left = score + w_left
        w_right = score + w_right
        print(f"{'|   ' * depth}Feature {feature_best} <= {split_value}")  # left node represent feature <= split_value
        node = TreeNode(feature_best, split_value, w=self.init_w)
        self.com_g_h(w_left, left_point)
        self.com_g_h(w_right, right_point)
        node.left = self.split(left_point, features, depth + 1, 'Left', w_left)
        node.right = self.split(right_point, features, depth + 1, 'Right', w_right)
        return node

    def boostrap(self):
        if self.x.ndim == 1:
            n = 1
            m = len(self.x)
        else:
            m, n = self.x.shape
        index = np.random.randint(m, size=m)
        features = np.random.choice(n, size=int(np.sqrt(n)), replace=False)
        return index, features

    def com_g_h(self, w, points):
        g = (-2 * (self.y[points] - w)).reshape(-1, 1)
        h = 2 * np.ones_like(g)
        self.g_h[points] = np.c_[g, h]
    
    def com_l_w(self, idx, features):
        total_g_h = np.sum(self.g_h[idx], axis=0)
        l_prev = total_g_h[0] ** 2 / (total_g_h[1] + self.Lambda)
        l_best = None
        unique_idx, counts = np.unique(idx, return_counts=True)

        for feature in features:
            num = 0
            has_nan = False
            if np.isnan(self.x[idx, feature]).any():
                nan_idx = list(np.where(np.isnan(self.x[idx, feature]))[0])
                real_idx = [i for i in idx if i not in nan_idx]
                has_nan = True
            else:
                real_idx = idx
            m = len(real_idx)
            bin_num = np.max([1, np.min([self.max_bin, int(self.epsilon * m)])])
            left_g_h = np.zeros(total_g_h.shape)
            left_point = []
            value_max = np.max(self.x[real_idx, feature])

            for i in self.x_sorted[:, feature]:
                if num <= bin_num - 1:
                    if i in real_idx:
                        count = counts[np.where(unique_idx == i)][0]
                        left_g_h += count * self.g_h[i]
                        left_point.extend(count * [i])
                        num += count
                elif (i in real_idx and i not in left_point and self.x[i, feature] == self.x[left_point[-1], feature]
                      and self.x[i, feature] != value_max):
                    count = counts[np.where(unique_idx == i)][0]
                    left_g_h += count * self.g_h[i]
                    left_point.extend(count * [i])
                    num += count
                else:
                    right_g_h = total_g_h - left_g_h
                    l_split = 1/2 * (left_g_h[0] ** 2 / (left_g_h[1] + self.Lambda) +
                                     right_g_h[0] ** 2 / (right_g_h[1] + self.Lambda) -
                                     l_prev) - self.gamma

                    if has_nan:
                        nan_g_h = np.sum(self.g_h[nan_idx], axis=0)
                        add_to_left_left_g_h = nan_g_h + left_g_h
                        add_to_left_right_g_h = total_g_h - add_to_left_left_g_h
                        l_left_split = 1 / 2 * (add_to_left_left_g_h[0] ** 2 / (add_to_left_left_g_h[1] + self.Lambda) +
                                                add_to_left_right_g_h[0] ** 2 / (add_to_left_right_g_h[1] + self.Lambda) -
                                                l_prev) - self.gamma
                        if l_left_split > l_split:
                            l_split = l_left_split
                            left_g_h = add_to_left_left_g_h.copy()
                            right_g_h = add_to_left_right_g_h.copy()
                            left_point.extend(nan_idx)

                    if l_best is None or l_split > l_best:
                        l_best = l_split
                        feature_best = feature
                        best_left_point = left_point.copy()
                        best_right_point = [i for i in idx if i not in best_left_point]
                        best_w_left = - left_g_h[0] / (left_g_h[1] + self.Lambda)
                        best_w_right = - right_g_h[0] / (right_g_h[1] + self.Lambda)
                        best_split_value = self.x[best_left_point[-1], feature]
                    bin_num += bin_num
                    if bin_num >= m:
                        break

        if l_best <= 1e-5:
            return None

        return feature_best, best_split_value, best_left_point, best_right_point, best_w_left, best_w_right

    def _predict(self, x, tree):
        node = self.trees[f'tree{tree}']
        init_y = node.w
        while not isinstance(node, float):
            if x[node.feature] <= node.value:
                node = node.left
            else:
                node = node.right
        return node - init_y

    def single_tree_predict(self, x, tree):
        if x.ndim == 1:
            return self._predict(x, tree)
        else:
            return np.array([self._predict(i, tree) for i in x])

    def predict(self, x):
        for i in range(self.max_trees):
            if i == 0:
                pred = self.single_tree_predict(x, i) * self.eta
            else:
                pred += self.single_tree_predict(x, i) * self.eta
        return pred + self.first_score

    def internal_predict(self, x, tree):
        if tree == 0:
            pred = self.single_tree_predict(x, tree) * self.eta
            return pred
        for i in range(tree+1):
            if i == 0:
                pred = self.single_tree_predict(x, i) * self.eta
            else:
                pred += self.single_tree_predict(x, i) * self.eta
        return pred


