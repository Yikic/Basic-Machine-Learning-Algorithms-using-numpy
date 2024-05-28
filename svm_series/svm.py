import random
import re
from string import punctuation
import numpy as np
from utils import NaiveBayes, rbf_kernel_matrix


class SVMHardMargin(NaiveBayes):
    """
    Notes:
        this version is only for spam classification

    Examples:
        svmhard = utils.SVMHardMargin(radius=0.1)
        svmhard.fit(train_messages,train_labels)
        print(np.mean(svmhard.predict(test_messages)==test_labels))
    """
    def __init__(self,radius=0.5):
        super().__init__()
        self.square = None
        self.b = 0
        self.alpha = None
        self.y = None
        self.x = None
        self.radius = radius

    def count(self,x):
        """
        Notes:
            use for spam classification
        """
        x_matrix = np.zeros((len(x),self.k))
        i = 0
        for sentence in x:
            word_list = []
            words_count = np.zeros(self.k)
            re_punc = re.sub(f"[{re.escape(punctuation)}]", "", sentence).split()
            word_lower = [word.lower() for word in re_punc]
            for word in word_lower:
                if word in self.voca:
                    word_list.append(word)
            words, counts = np.unique(np.array(word_list), return_counts=True)
            for word, count in zip(words, counts):
                if word in self.voca:
                    idx = self.voca.index(word)
                    words_count[idx] = count
            x_matrix[i,:] = words_count
            i+=1
        return x_matrix

    def fit(self,x,y):
        self.creat_voca(x)
        x = self.count(x)
        m, n = x.shape
        y = 2 * y - 1
        square = np.sum(x*x,axis=1)
        gram = x @ x.T
        K = np.exp(- (square.reshape(-1,1)+square.reshape(1,-1)-2*gram) / (2 * self.radius**2))
        alpha = np.zeros(m)
        for i in np.arange(1,m,2):
            C = - np.concatenate([alpha[:i-1], alpha[i+1:]]) @ np.concatenate([y[:i-1], y[i+1:]])
            if K[i - 1, i - 1] + K[i, i] - 2 * K[i, i - 1] == 0:
                continue
            alpha[i] = ((K[i - 1, i - 1] - K[i - 1, i]) * y[i] * C + 1 - y[i - 1] * y[i]) / (
                        K[i - 1, i - 1] + K[i, i] - 2 * K[i, i - 1])
            if alpha[i] < C * y[i]:
                alpha[i] = C * y[i]
            alpha[i - 1] = (C - alpha[i] * y[i]) * y[i - 1]
            if m % 2 == 1:
                alpha[-1] = - y[-1] * alpha[:-1] @ y[:-1]
        preds = K @ (alpha * y)
        b = - 1 / 2 * (min(preds[preds > 0]) + max(preds[preds < 0]))
        self.x = x
        self.y = y
        self.alpha = alpha
        self.b = b
        self.square = square

    def predict(self,x, plot=False):
        if plot:
            K, _ = rbf_kernel_matrix(x, predict=True, ori_x=self.x[:,:2], radius=self.radius)
        elif type(x[0]) == np.ndarray:
            K, _ = rbf_kernel_matrix(x, predict=True, ori_x=self.x, radius=self.radius)
        else:
            x = self.count(x)
            K, _ = rbf_kernel_matrix(x,predict=True, ori_x=self.x, radius=self.radius)
        preds = K @ (self.alpha * self.y)
        output = (1 + np.sign(preds+self.b)) // 2
        return output


class SVMSoftMargin(SVMHardMargin):
    """
     Args:
         C: penalty factor, control the penalty of errors
         tol: tolerance, allow small errors in KKT condition
         radius: std of rbf kernel
         max_iter: maximum iteration before all satisfy KKT
                condition

     Notes:
         * labels are 0 or 1

     Examples:
        tol_pack = [1e-3, 1e-5, 1e-2]
        C_pack = [1, 0.1, 10, 0.001]
        radius_pack = np.arange(3,5,0.1)
        best_score = 0
        best_param = []
        for tol in tol_pack:
            for C in C_pack:
                for radius in radius_pack:
                    svmhard = utils.SVMSoftMargin(train_messages,train_labels,tol=tol,C=C,radius=radius, max_iter=1)
                    svmhard.fit(train_messages,train_labels)
                    score = np.mean(svmhard.predict(val_messages)==val_labels)
                    print('valid:',score)
                    if score > best_score:
                        best_param = [tol, C, radius]
                        best_score = score
        print(best_param)
        svmhard_best = utils.SVMSoftMargin(train_messages,train_labels,tol=best_param[0], C=best_param[1], radius=best_param[2])
        svmhard_best.fit(train_messages, train_labels)
        utils.plot(svmsoft_best,test_messages,test_labels,if_svm=True)
        print('test precision:', np.mean(svmhard_best.predict(test_messages)==test_labels))
     """

    def __init__(self,x, y, C=1, tol=1e-3, radius=3.2, max_iter=10):
        super().__init__(radius=radius)
        self.x = x
        self.y = y * 2 - 1
        self.m = x.shape[0]
        self.tol = tol
        self.C = C
        self.alpha = np.zeros(self.m)
        self.K, self.square = rbf_kernel_matrix(self.x, radius=self.radius)
        self.maxiter = max_iter
        self.E = np.zeros(self.m)

    def objFun(self, a, i2):
        self.alpha[i2] = a
        matrix = (self.y * self.alpha).reshape(-1,1) * self.K * (self.y * self.alpha)
        return sum(self.alpha) - 1/2 * np.sum(matrix)

    def compute_error(self, i):
        return sum(self.alpha*self.y*self.K[i,:])+self.b-self.y[i]

    def non_bound(self):
        count = 0
        indices = []
        m, n = self.x.shape
        for i in range(m):
            if 0 < self.alpha[i] < self.C:
                count += 1
                indices.append(i)
        return count, indices

    def select_j(self,i, indices):
        Ei = self.E[i]
        temp_j = np.argmax(abs(self.E[indices]-Ei))
        j = indices[temp_j]
        return j

    def examineExample(self,i2):
        E2 = self.E[i2]
        if (self.y[i2] * E2 < -self.tol and self.alpha[i2] == 0) or (
                abs(self.y[i2] * E2) > self.tol and self.C > self.alpha[i2] > 0) or (
                self.y[i2] * E2 > self.tol and self.alpha[i2] == self.C):

            count, indices = self.non_bound()

            if count > 1:
                i1 = self.select_j(i2, indices)
                if self.takeStep(i1, i2):
                    return 1

            for i1 in indices:
                if self.takeStep(i1, i2):
                    return 1

            sequence = list(range(self.m))
            random.shuffle(sequence)
            for i1 in sequence:
                if self.takeStep(i1, i2):
                    return 1
        return 0

    def takeStep(self, i1, i2):
        if i1 == i2:
            return 0

        E1 = self.E[i1]
        E2 = self.E[i2]
        alpha_i1_old = self.alpha[i1].copy()
        alpha_i2_old = self.alpha[i2].copy()

        if self.y[i1] != self.y[i2]:
            L = max(0, alpha_i2_old - alpha_i1_old)
            H = min(self.C, self.C + alpha_i2_old - alpha_i1_old)
        else:
            L = max(0, alpha_i2_old + alpha_i1_old - self.C)
            H = min(self.C, alpha_i2_old + alpha_i1_old)
        if L == H:
            return 0

        eta = 2 * self.K[i1, i2] - self.K[i1, i1] - self.K[i2, i2]
        if eta < 0:
            self.alpha[i2] -= self.y[i2] * (E1 - E2) / eta
            self.alpha[i2] = max(L, min(H, self.alpha[i2]))
        else:
            Lobj = self.objFun(L, i2)
            Hobj = self.objFun(H, i2)
            if Lobj > Hobj + self.tol:
                self.alpha[i2] = L
            elif Lobj < Hobj - self.tol:
                self.alpha[i2] = H
            else:
                self.alpha[i2] = alpha_i2_old

        if self.alpha[i2] < 1e-8:
            self.alpha[i2] = 0
        elif self.alpha[i2] > self.C - 1e-8:
            self.alpha[i2] = self.C

        if abs(self.alpha[i2] - alpha_i2_old) < self.tol * (self.alpha[i2] + alpha_i2_old + self.tol):
            return 0

        self.alpha[i1] += self.y[i1] * self.y[i2] * (alpha_i2_old - self.alpha[i2])

        b_old = self.b
        b1 = b_old - E1 - self.y[i1] * (self.alpha[i1] - alpha_i1_old) * self.K[i1, i1] - \
             self.y[i2] * (self.alpha[i2] - alpha_i2_old) * self.K[i1, i2]
        b2 = b_old - E2 - self.y[i1] * (self.alpha[i1] - alpha_i1_old) * self.K[i1, i2] - \
             self.y[i2] * (self.alpha[i2] - alpha_i2_old) * self.K[i2, i2]
        if 0 < self.alpha[i1] < self.C:
            self.b = b1
        elif 0 < self.alpha[i2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        self.E = self.E + self.y[i1] * (self.alpha[i1] - alpha_i1_old) * self.K[i1, :] + \
             self.y[i2] * (self.alpha[i2] - alpha_i2_old) * self.K[i2, :] + self.b - b_old

        return 1

    def fit(self, x, y):
        numChanged = 0
        examineAll = 1
        iter = 0

        E = []
        for i in range(self.m):
            E.append(self.compute_error(i))
        self.E = np.array(E)

        while (numChanged > 0 or examineAll) and iter < self.maxiter:
            numChanged = 0
            if examineAll:
                for i2 in range(self.m):
                    numChanged += self.examineExample(i2)
            else:
                for i2 in self.non_bound()[1]:
                    numChanged += self.examineExample(i2)

            if examineAll:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
                iter += 1
                pred = self.K @ (self.alpha * self.y) + self.b
                print(f'epoch:{iter}, precision:{np.mean(np.sign(pred)==self.y)}')

