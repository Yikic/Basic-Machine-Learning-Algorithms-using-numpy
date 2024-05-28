import random
import numpy as np
import utils
from utils import rbf_kernel_matrix, poly_kernel


np.random.seed(42)


class SVR:
    """
    Notes:
        - optimization using smo

    Args:
        - U is penalty factor
        - epsilon is hard margin error
        - func can choose rbf_kernel_matrix or poly_kernel
        - gamma is the std of Gaussian
        - tol is kkt condition tolerance

    Examples:
        standard_x = utils.StandardScaler()
        standard_y = utils.StandardScaler()

        scaler_x = standard_x.fit_transform(x)
        scaler_y = standard_y.fit_transform(y)

        x_train = scaler_x[:500]
        y_train = scaler_y[:500]
        x_test = scaler_x[500:]
        y_test = scaler_y[500:]

        best_gamma = 1
        best_mse = 1000
        gamma_list = np.arange(1, 10, 0.5)
        for gamma in gamma_list:
            svr_test = svr.SVR(x_train,y_train,U=1e3,max_iter=10, tol=1, epsilon=0.1, gamma=gamma)
            svr_test.fit()
            pred = svr_test.predict(x_test)
            mse = utils.mse(pred, y_test)
            print('mse:', mse)
            if mse <= best_mse:
                best_gamma = gamma
        print(best_gamma)

        svr = svr.SVR(x_train,y_train,U=1e3,max_iter=100, tol=1, epsilon=0.1, gamma=10)
        svr.fit()
        pred = svr.predict(x_test)
        time = np.arange(0,len(pred))
        plt.plot(time,pred, c='r')
        plt.scatter(time,y_test, c='b')
        plt.show()

    Reference:
        [1] 'Schölkopf and Smola (2002). "Learning with Kernels."
            <https://mcube.lab.nycu.edu.tw/~cfung/docs/books/scholkopf2002learning_with_kernels.pdf>'_
        [2] 'Drucker, H., Burges, C. J., Kaufman, L., Smola, A., & Vapnik, V. (1996). "Support vector
            regression machines."
            <https://proceedings.neurips.cc/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf>'_
    """
    def __init__(self, x, y, func=rbf_kernel_matrix, epsilon=0.1, U=1e3, max_iter=10, gamma=1, degree=5, tol=1):
        self.x = x
        self.y = y
        self.func = func
        self.m, self.n = self.x.shape
        self.epsilon = epsilon
        self.U = U
        self.max_iter = max_iter
        self.gamma = gamma
        self.degree = degree

        if self.func == rbf_kernel_matrix:
            self.arg = self.gamma
        else:
            self.arg = self.degree

        self.alpha = np.zeros(self.m)
        self.alpha_star = np.zeros(self.m)
        self.K = self.func(self.x, radius=self.arg)
        self.f = np.zeros(self.m)
        self.b = 0
        self.tol = tol
        self.cache = np.zeros(self.m)

    def take_step(self, i, j):
        if i == j:
            return 0
        r = self.alpha[i] - self.alpha_star[i] + self.alpha[j] - self.alpha_star[j]
        L = np.max((r - self.U, -self.U))
        H = np.min((r + self.U, self.U))
        if L == H:
            return 0
        l = np.min((r, 0))
        h = np.max((r, 0))
        X = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

        if X > 0:
            beta0 = self.alpha[i] - self.alpha_star[i] + (self.f[i] - self.y[i] - self.f[j] + self.y[j]) / X
            betap = beta0 - 2 * self.epsilon / X
            betan = beta0 + 2 * self.epsilon / X
            beta = np.max((np.min((beta0, h)), l))
            if beta == h:
                beta = np.max((np.min((betap, H)), h))
            elif beta == l:
                beta = np.max((np.min((betan, l)), L))
        elif self.f[i] - self.y[i] - self.f[j] + self.y[j] < 0:
            beta = h
            if self.f[i] - self.y[i] - self.f[j] + self.y[j] + 2 * self.epsilon < 0:
                beta = H
        else:
            beta = l
            if self.f[i] - self.y[i] - self.f[j] + self.y[j] - 2 * self.epsilon > 0:
                beta = L

        if np.abs(beta - self.alpha[i] + self.alpha_star[i]) < self.epsilon * (self.epsilon * np.abs(beta) +
                                                                               self.alpha[i] + self.alpha_star[i]):
            return 0

        self.alpha[i] = np.max((beta, 0))
        self.alpha_star[i] = np.max((-beta, 0))
        self.alpha[j] = np.max((0, r - beta))
        self.alpha_star[j] = np.max((0, beta - r))

        # update b
        non_bound = ((self.alpha > 0) & (self.alpha < self.U)) | (self.alpha == 0)
        non_bound_star = ((self.alpha_star > 0) & (self.alpha_star < self.U)) | (self.alpha_star == self.U)
        if sum(non_bound_star) == 0:
            e_hi = np.min(self.f[non_bound] - self.y[non_bound] + self.epsilon)
        else:
            e_hi = np.min((np.min(self.f[non_bound] - self.y[non_bound] + self.epsilon),
                           np.min(self.f[non_bound_star] - self.y[non_bound_star] - self.epsilon)))

        non_bound = ((self.alpha > 0) & (self.alpha < self.U)) | (self.alpha == self.U)
        non_bound_star = ((self.alpha_star > 0) & (self.alpha_star < self.U)) | (self.alpha_star == 0)
        if sum(non_bound) == 0:
            e_lo = np.max(self.f[non_bound_star] - self.y[non_bound_star] - self.epsilon)
        else:
            e_lo = np.max((np.max(self.f[non_bound] - self.y[non_bound] + self.epsilon),
                           np.max(self.f[non_bound_star] - self.y[non_bound_star] - self.epsilon)))

        b_hi = self.b - e_hi
        b_lo = self.b - e_lo
        self.b = (b_lo + b_hi) / 2

        # update fx and error cache
        self.f = (self.alpha_star - self.alpha) @ self.K + self.b
        self.cache = self.f - self.y
        return 1

    def examine_example(self, i):
        kkt_i = (int(self.alpha[i] > 0) * np.max((0, self.f[i] - self.y[i] + self.epsilon)) + int(self.alpha_star[i]) *
                np.max((0, self.y[i] + self.epsilon - self.f[i])) + int(self.U - self.alpha[i]) * np.max(
                (0, self.y[i] - self.epsilon - self.f[i])) + int(self.U - self.alpha_star[i]) * np.max(
                (0, self.f[i] - self.y[i] - self.epsilon)))

        if kkt_i > self.tol:
            count, indices = self.non_bound()
            if count > 1:
                j = self.select_j(i, indices)
                if self.take_step(i, j):
                    return 1

            random.shuffle(indices)
            for j in indices:
                if self.take_step(i, j):
                    return 1

            sequence = [x for x in range(self.m) if x not in indices]
            random.shuffle(sequence)
            for j in sequence:
                if self.take_step(i, j):
                    return 1
        return 0

    def non_bound(self):
        count = 0
        indices = []
        for i in range(self.m):
            if 0 < self.alpha[i] < self.U:
                count += 1
                indices.append(i)
        for i in range(self.m):
            if 0 < self.alpha_star[i] < self.U:
                count += 1
                indices.append(i)
        return count, indices

    def select_j(self, i, indices):
        Ei = self.cache[i]
        temp_j = np.argmax(abs(self.cache[indices]-Ei))
        j = indices[temp_j]
        return j

    def fit(self):
        ExamineAll = 1
        NumChanged = 0
        it = 0
        while (NumChanged > 0 or ExamineAll == 1) & (it < self.max_iter):
            NumChanged = 0
            if ExamineAll == 1:
                for i in range(self.m):
                    NumChanged += self.examine_example(i)
            else:
                _, indices = self.non_bound()
                for i in indices:
                    NumChanged += self.examine_example(i)

            if ExamineAll == 1:
                ExamineAll = 0
            elif NumChanged == 0:
                ExamineAll = 1
                it += 1
                print('current iteration:', it)
                print('training error:', utils.mse(self.f, self.y))
        return

    def predict(self, x):
        K = self.func(x, predict=True, ori_x=self.x, radius=self.arg)
        preds = K @ (self.alpha_star - self.alpha) + self.b
        return preds