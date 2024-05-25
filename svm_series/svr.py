import numpy as np
import utils
from utils import rbf_kernel_matrix, poly_kernel
from scipy.optimize import minimize

np.random.seed(42)

class SVR:
    """
    Args:
        - U is penalty factor
        - epsilon is hard margin error
        - func can choose rbf_kernel_matrix or poly_kernel
    """
    def __init__(self, x, y, func=rbf_kernel_matrix, epsilon=0.1, U=5, max_iter=50):
        self.x = x
        self.y = y
        self.func = func
        self.m, self.n = self.x.shape
        self.c = np.zeros(2*self.m)
        self.epsilon = np.ones(self.m) * epsilon
        self.U = U
        self.init_xy()
        self.max_iter = max_iter

    def init_xy(self):
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)

    def step(self):
        K, _ = self.func(self.x)
        temp = np.concatenate((K, -K), axis=1)
        Q = np.concatenate((temp, -temp), axis=0)
        self.c[:self.m] = self.epsilon - self.y
        self.c[self.m:] = self.epsilon + self.y

        # objective func
        def objective(beta):
            return 0.5 * np.dot(beta.T, np.dot(Q, beta)) + np.dot(self.c, beta)

        # equality constraint
        def eq_constraint(beta):
            return np.sum(beta[:self.m]) - np.sum(beta[self.m:])

        eq_constraints = {'type': 'eq', 'fun': eq_constraint}
        bounds = [(0, self.U)] * (2 * self.m)
        beta0 = np.zeros(2 * self.m)    # initialize multipliers beta

        options = {
            'maxiter': self.max_iter,
            'disp': True
        }

        def callback(beta):
            obj_value = objective(beta)
            print(f'objective function value: {obj_value}')

        result = minimize(objective, beta0, method='SLSQP', bounds=bounds, constraints=eq_constraints,
                          options=options, callback=callback)

        if result.success:
            print("optimal multipliers：", result.x)
            print("minimum objective value：", result.fun)
        else:
            print("optimization failed：", result.message)

        beta = result.x
        beta_between = beta[:self.m] - beta[self.m:]
        index = np.where((1e-8 < beta) & (beta < self.U))[0]
        b = []
        for id in index:
            if id <= self.m - 1:
                t = self.y[id] - self.epsilon - beta_between @ K[:, id]
                b.append(t)
            else:
                t = self.y[id - self.m] + self.epsilon - beta_between @ K[:, id - self.m]
                b.append(t)
        b = np.array(b)
        b = np.mean(b)
        pred = beta_between @ K + b
        return pred