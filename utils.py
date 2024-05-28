import os
import random
import re
from string import punctuation
import matplotlib.pyplot as plt
import numpy as np
import scipy

np.random.seed(42)


def qp_lc(h, c, I, e=None):
    """
    Notes:
        - use active set method to solve quadratic programming-linear constraint problem
        - if there is equality, remember to initialize x to be satisfied, if you have equality more than
          1, change line 10
        - only for positive definite matrix

    :param e: augmented matrix
    :param i: augmented matrix, must be smaller or equal than b
    """

    m = len(c)
    x = np.zeros(m)  # must satisfy equality, remember to change it

    if e is not None:
        # assume there is only one equality
        A = np.concatenate((I[:,:-1], np.expand_dims(e[:-1],axis=0)), axis=0)
        b = np.concatenate((I[:,-1],np.expand_dims(e[-1],axis=0)),axis=0)
        e_len = 1
    else:
        A = I[:, :-1]
        b = I[:, -1]
        e_len = 0
    w = []
    # assume x=0 satisfy all equality
    for i in range(A.shape[0]):
        if A[i] @ x == b[i]:
            w.append(i)
    A_w = A[w]
    b_w = b[w]
    StopFlag = 0
    k = 0

    while not StopFlag:
        # solve QP-SUB to get d
        g = h @ x + c
        d = - np.linalg.inv(h) @ (np.eye(m) - A_w.T @ np.linalg.inv((A_w @ np.linalg.inv(h) @ A_w.T)+1e-8*np.eye(A_w.shape[0]))
                                  @ A_w @ np.linalg.inv(h)) @ g
        if np.all(d <= 1e-8):
            Lambda = - np.linalg.inv((A_w @ np.linalg.inv(h) @ A_w.T)) @ (A_w @ np.linalg.inv(h) @ c + b_w)
            if e_len == 0:
                Lambda_q = np.min(Lambda)
                q = np.argmin(Lambda)
            else:
                Lambda_q = np.min(Lambda[:-e_len])
                q = np.argmin(Lambda[:-e_len])
            if Lambda_q >= 0:
                StopFlag = 1
            else:
                A_w = np.concatenate((A_w[:q],A_w[q+1:]), axis=0)
                b_w = np.concatenate((b_w[:q],b_w[q+1:]), axis=0)
        else:
            index = np.where(A@d > 1e-8)[0]
            temp = np.min((b[index] - A[index] @ x) / (A[index] @ d))
            p = np.argmin((b[index] - A[index] @ x) / (A[index] @ d))
            p = index[p]
            alpha = np.min((temp, 1))
            x = x + alpha * d
            if temp <= 1:
                A_w = np.concatenate((np.expand_dims(A[p],axis=0), A_w), axis=0)
                b_w = np.concatenate((np.expand_dims(b[p], axis=0), b_w), axis=0)

        k += 1

    return x


# use for language process
with open('stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()
    stopwords_set = set(stopwords)


def one_hot_encoding(y):
    cats = np.unique(y)
    i = 0
    one_hot_y = np.zeros((len(y), len(cats)))
    for cat in cats:
        t = np.zeros(len(cats))
        t[i] = 1
        one_hot_y[y==cat] = t
        i += 1
    return one_hot_y


def plot_loss(loss):
    x = np.arange(0,len(loss)) + 1
    plt.plot(x,loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def k_fold(x,y,k):
    """
    Examples:
        folds = k_fold(train_matrix,train_labels,10)
        for (x_train, y_train, x_val, y_val) in folds:
            training process..
    """
    m = len(y)
    n = m // k
    folds = []

    data = np.c_[x, y]
    np.random.shuffle(data)

    for i in range(k):
        start = i * n
        end = start + n if i != k - 1 else m

        val_set = data[start:end]
        train_set = np.vstack((data[:start], data[end:]))

        x_train, y_train = train_set[:, :-1], train_set[:, -1]
        x_val, y_val = val_set[:, :-1], val_set[:, -1]

        folds.append((x_train, y_train, x_val, y_val))

    return folds


def create_voca(x):
    """

    Args:
        x: a list of sentences

    Returns: a vocabulary which contains the word that appear in the list(at least 5 times)

    """

    words = []
    voca = []
    for sentence in x:
        re_punc = re.sub(f"[{re.escape(punctuation)}]", "", sentence).split()
        word_lower = [word.lower() for word in re_punc]
        words.extend(word_lower)

    words, counts = np.unique(np.array(words), return_counts=True)
    for word, count in zip(words, counts):
        if count >= 5:
            voca.append(word)
            # dict[word] = count

    return voca


def voca_frequency(x, voca):
    """

    Args:
        x: a list of sentences
        voca: vocabulary return form function create_voca

    Returns: a matrix of the frequency of each word in voca

    """

    k = len(voca)
    x_matrix = np.zeros((len(x),k))
    i = 0
    for sentence in x:
        word_list = []
        words_count = np.zeros(k)
        re_punc = re.sub(f"[{re.escape(punctuation)}]", "", sentence).split()
        word_lower = [word.lower() for word in re_punc]
        for word in word_lower:
            if word in voca:
                word_list.append(word)
        words, counts = np.unique(np.array(word_list), return_counts=True)
        for word, count in zip(words, counts):
            if word in voca:
                idx = voca.index(word)
                words_count[idx] = count
        x_matrix[i,:] = words_count
        i+=1
    return x_matrix


def plot(model,x,y, if_svm=False):
    """

    Notes:
         * would only take the first and second feature of your sample
         * only apply to binary classification
         * plot from 3 dim to 2 dim

    """

    # x = model.count(x)
    x1,x2 = np.meshgrid(np.linspace(x[:,0].min()-1,x[:,0].max()+1,20),np.linspace(x[:,1].min()-1,x[:,1].max()+1,20))
    if if_svm:
        z = model.predict(np.c_[x1.ravel(),x2.ravel()], plot=True)
    else:
        z = model.predict(np.c_[x1.ravel(), x2.ravel()])
    z = z.reshape(x1.shape)

    plt.figure(figsize=(12,8))
    plt.contourf(x1,x2,z, levels=[-float('inf'),0,float('inf')],colors=['orange','cyan'])
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


def rbf_kernel(xi,x,sigma=1):
    return np.exp(-(xi-x)@(xi-x)/(2*sigma**2))


def rbf_kernel_matrix(x, predict=False, ori_x=None, radius=0.5):
    square = np.sum(x*x,axis=1)
    if predict:
        if ori_x is None:
            raise ValueError('in predict mode, you have to give ori_x')
        ori_square=np.sum(ori_x*ori_x,axis=1)
        gram = x @ ori_x.T
        K = np.exp(- (square.reshape(-1,1)+ori_square.reshape(1,-1)-2*gram) / (2 * radius**2))
    else:
        gram = x @ x.T
        K = np.exp(- (square.reshape(-1, 1) + square.reshape(1, -1) - 2 * gram) / (2 * radius ** 2))
    return K


def poly_kernel(x,p):
    temp = x @ x.T
    K = (temp + np.ones(temp.shape))**p
    return K


def sign(x):
    if x >= 0:
        x = 1
    else:
        x = 0
    return x


def add_intercept(x):
    m,n = x.shape
    x_new = np.zeros((m,n+1))
    x_new[:,0] = 1
    x_new[:,1:] = x
    return x_new


def load_data(path, intercept=False, inverse=False):
    # data and label are in the same file
    data = np.loadtxt(path,delimiter=',',skiprows=1)
    if inverse:
        x = data[:, 1:]
        y = data[:, 0]
    else:
        x = data[:,:-1]
        y = data[:,-1]
    if intercept:
        x = add_intercept(x)
    return x,y


def matrix_inv(h):
    """
    singular matrix is also OK
    """
    if np.linalg.det(h) == 0:
        h_inv = np.linalg.pinv(h)
    else:
        h_inv = np.linalg.inv(h)
    return h_inv


def sigmoid(x,theta, if_single_input=False):
    """
    x is m * n, theta is n * 1, the dimension will be reduced
    """
    if if_single_input:
        return 1/(1+np.exp(-x))
    else:
        return 1/(1+np.exp(-x@theta))


class LogisticRegression:
    """
    (Newton's Method)
    Examples:
        lr = utils.LogisticRegression()
        lr.fit(x_train,y_train)
        lr.predict(x_valid)
        lr.precision(x_valid,y_valid)
        lr.plot(x_valid,y_valid)
        plt.show()
    """
    def __init__(self):
        self.theta=None

    def fit(self,x,y):
        """
        use Newton's Method
        epsilon = 1e-5
        """
        def hessian(x,theta):
            w=np.expand_dims(sigmoid(x,theta),1)@np.expand_dims((1-sigmoid(x,theta)),1).T
            w=np.diag(np.diag(w))
            h=-x.T@w@x
            return h

        def gradient(x,y,theta):
            return x.T@(y-sigmoid(x,theta))

        m,n = x.shape
        epsilon = 1e-5
        theta = np.zeros(n)
        theta_next = theta - matrix_inv(hessian(x,theta))@gradient(x,y,theta)
        while np.linalg.norm(theta_next-theta,1) >= epsilon:
            theta = theta_next
            # though is maximum problem, it's still minus, because hessian would be minus
            theta_next = theta - matrix_inv(hessian(x, theta)) @ gradient(x, y, theta)
        self.theta = theta_next

    def predict(self,x):
        return np.round(sigmoid(x,self.theta))

    def precision(self, x, y):
        print(np.mean(self.predict(x) == y))

    def plot(self,x,y):
        """
        plot 2-dimension hyperplane
        """
        bi_y = np.unique(y)
        plt.plot(x[y == bi_y[0], -2], x[y == bi_y[0], -1], 'bx', linewidth=2)
        plt.plot(x[y == bi_y[1], -2], x[y == bi_y[1], -1], 'go', linewidth=2)

        margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.2
        margin2 = (max(x[:, -1]) - min(x[:, -1])) * 0.2
        x1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.01)
        x2 = -(self.theta[0]+self.theta[1]*x1)/self.theta[2]

        plt.plot(x1, x2, c='red', linewidth=2)
        plt.xlim(x[:, -2].min() - margin1, x[:, -2].max() + margin1)
        plt.ylim(x[:, -1].min() - margin2, x[:, -1].max() + margin2)
        plt.xlabel('x1')
        plt.ylabel('x2')


class GDA(LogisticRegression):
    """
    the same usage as LogisticRegression
    """

    def __init__(self):
        super().__init__()
        self.sigma = None
        self.miu0 = None
        self.miu1 = None
        self.fai = None
        self.N0 = None
        self.N1 = None

    def fit(self,x,y):
        """
        don't give intercept
        """
        x_new = x[:,1:]
        x = x_new
        m,n = x.shape

        self.N1 = sum(y)
        self.N0 = sum(1-y)
        self.fai = self.N1/m
        self.miu1 = x.T @ y/self.N1
        self.miu0 = x.T @ (1-y)/self.N0
        s1 = (x[y==1]-self.miu1).T @ (x[y==1]-self.miu1)
        s0 = (x[y==0]-self.miu0).T @ (x[y==0]-self.miu0)
        self.sigma = (s1+s0)/m

        self.theta = np.zeros(n+1)
        self.theta[1:] = matrix_inv(self.sigma)@(self.miu1-self.miu0)
        self.theta[0] = -1/2*(self.miu1+self.miu0)@self.theta[1:]-(self.N1-self.N0)/2*\
                        np.log(2*np.pi)-np.log((1-self.fai)/self.fai)


class LocallyWeightedLinearRegression():
    """
    Examples:
        llr = utils.LocallyWeightedLinearRegression()
        llr.fit(x_train,y_train)
        llr.predict(x_valid)
        llr.MSE(y_valid)
        llr.plot(x_valid,y_valid)
        plt.show()
    """

    def __init__(self):
        self.y = None
        self.x = None
        self.pred = None

    def fit(self,x,y):
        self.x = x
        self.y = y

    def predict(self,x,tau=0.5):
        l, n = x.shape
        self.pred = np.zeros(l)

        for i in range(l):
            w = np.exp(-np.linalg.norm(self.x - x[i],2,1)/(2*tau**2))
            w = np.diag(w)
            theta = matrix_inv(self.x.T@w@self.x)@(self.x.T@w@self.y)
            self.pred[i] = x[i] @ theta

    def MSE(self,y):
        print(np.linalg.norm(self.pred-y,2))

    def plot(self,x,y):
        plt.scatter(x[:,-1],y,marker='x',c='b',linewidths=2,label='validation set')
        plt.scatter(x[:,-1],self.pred,marker='o',c='r',linewidths=2,label='prediction')
        plt.legend()
        plt.show()


class NaiveBayes():
    """
    Notes:
        * y can be any pair str, the only thing you have to change is in line 236,237 and 260,261

    Args:
        y: str of binary class
        x: sentences

    Examples:
        nb = utils.NaiveBayes()
        nb.fit(x_train,y_train)
        y_pred = nb.predict(x_valid)
        print('precision:', np.mean(y_pred==y_valid))
    """

    def __init__(self):
        self.fai = None
        self.fai_neg = None
        self.fai_pos = None
        self.email_len = []
        self.voca = []
        self.dict = {}
        self.k = int # the category of words

    def count(self,x):
        words = []
        words_count = np.zeros(self.k)
        for sentence in x:
            word_list = []
            re_punc = re.sub(f"[{re.escape(punctuation)}]", "", sentence).split()
            word_lower = [word.lower() for word in re_punc]
            for word in word_lower:
                if word in self.voca:
                    word_list.append(word)
            words.extend(word_list)

        words, counts = np.unique(np.array(words), return_counts=True)
        for word, count in zip(words, counts):
            if word in self.voca:
                idx = self.voca.index(word)
                words_count[idx] = count
        return words_count

    def creat_voca(self,x):
        words = []
        for sentence in x:
            re_punc = re.sub(f"[{re.escape(punctuation)}]", "", sentence).split()
            word_lower = [word.lower() for word in re_punc]
            words.extend(word_lower)

        words, counts = np.unique(np.array(words), return_counts=True)
        for word, count in zip(words, counts):
            if count >= 5:
                self.voca.append(word)
                self.dict[word] = count
        self.k = len(self.voca)

        for sentence in x:
            word_list = []
            re_punc = re.sub(f"[{re.escape(punctuation)}]", "", sentence).split()
            word_lower = [word.lower() for word in re_punc]
            for word in word_lower:
                if word in self.voca:
                    word_list.append(word)
            self.email_len.append(len(word_list))

    def fit(self,x,y):
        # y_new = np.zeros(len(y))
        # y_new[y=='ham'] = 0
        # y_new[y=='spam'] = 1  #define spam as positive
        # y = y_new
        self.creat_voca(x)

        pos_de = y@self.email_len + self.k
        neg_de = (1-y)@self.email_len + self.k
        x_pos = np.array(x)[y==1]
        x_neg = np.array(x)[y==0]

        self.fai_pos = (self.count(x_pos)+1)/pos_de
        self.fai_neg = (self.count(x_neg)+1)/neg_de
        self.fai = sum(y)/len(y)

    def predict(self,x):
        pred = np.zeros(len(x))
        i = 0
        for sentence in x:
            x_count = self.count(sentence)
            pred[i] = np.round(1/(1+np.exp(x_count@(np.log(self.fai_neg)-np.log(self.fai_pos))+np.log((1-self.fai)/self.fai))))
            i += 1

        # pred_new = np.zeros(len(pred))
        # pred_new = np.array(pred_new, dtype=object)
        # pred_new[pred==1] = 'spam'
        # pred_new[pred==0] = 'ham'
        # pred = pred_new

        return pred


class Perceptron:
    """
    Notes:
        * this is a perceptron with kernel trick
        * activation: sign(x)

    Examples:
        per = Perceptron()
        kernel = rbf_kernel
        for x,y in zip(train_x,train_y):
            per.update_state(kernel,x,y)
        total = 0
        for xi in test_x:
            total += per.predict(xi) == test_y
        print('precision:',np.mean(total))
        per.plot(test_x,test_y)
        plt.show()
    """

    def __init__(self):
        self.kernel = None
        self.state = []

    def update_state(self,kernel,x_i,y_i,learning_rate=0.5):
        self.kernel = kernel
        beta_i = learning_rate*(y_i-sign(sum(beta * self.kernel(x,x_i) for beta,x in self.state)))
        self.state.append((beta_i,x_i))

    def predict(self,x_i):
        return sign(sum(beta * self.kernel(x,x_i) for beta,x in self.state))

    def plot(self,x,y):
        x1,x2 = np.meshgrid(np.linspace(-10,10,20),np.linspace(-10,10,20))
        z = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                z[i, j] = self.predict([x1[i, j], x2[i, j]])

        plt.figure(figsize=(12,8))
        plt.contourf(x1,x2,z, levels=[-float('inf'),0,float('inf')],colors=['orange','cyan'])
        x_1 = x[y==1]
        x_0 = x[y==0]
        plt.scatter(x_1[:,-1],x_1[:,-2],marker='x',c='r')
        plt.scatter(x_0[:,-1],x_0[:,-2],marker='o',c='g')


class DecisionTree:
    """
    Args:
        both str and int are OK, if it's str, set if_init=True
        max_feature: only consider a maximum number of features in
                    a node, it's useful in random forest

    Todo:
        * add a predict function
        * AdaBoost

    Examples:
        df = pd.read_excel('../data/dt_train.xlsx')
        data = df.to_numpy()
        x_train = data[:,1:-1] #skip id column
        y_train = data[:,-1]

        dt = utils.DecisionTree(x_train,y_train,if_init=True)
        root_index = np.arange(0, dt.m)
        dt.tree_recursive(root_index, 'Root', 2)
    """
    def __init__(self, x, y, max_feature=None, if_init=True):
        self.x = x
        self.y = y
        self.ls = []
        self.m, self.n = self.x.shape
        if max_feature is None:
            self.max_feature = self.n
        else:
            self.max_feature = max_feature
        if if_init:
            self.init_xy()
        self.tree = []

    def init_xy(self):
        self.x = np.array(self.x)
        for i in range(self.n):
            dict = {}
            if type(self.x[:,i][0]) == str:
                strs = np.unique(self.x[:,i])
                for j in range(len(strs)):
                    self.x[:,i][self.x[:,i] == strs[j]] = j
                    dict[j] = strs[j]
            self.ls.append(dict)

        dict = {}
        if type(self.y[0]) == str:
            strs = np.unique(self.y)
            for i in range(len(strs)):
                self.y[self.y == strs[i]] = i
                dict[i] = strs[i]
        self.ls.append(dict)

    def entropy(self, y):
        m  = len(y)
        entropy = 0
        if m != 0:
            p = sum(y)/m
            if p != 0 and p != 1:
                entropy = - p*np.log(p) - (1-p) * np.log(1-p)
        return entropy

    def ratio(self, root_index, root_entropy):
        ratio = []
        features = np.random.choice(self.n, self.max_feature, replace=False)
        for i in features:
            cats, counts = np.unique(self.x[root_index,i], return_counts=True)
            total_entropy = 0
            weights = []
            indices = [root_index[np.where(cat == self.x[root_index, i])[0]] for cat in cats]

            for indice, count in zip(indices, counts):
                cat_y = self.y[indice]
                cat_entropy = self.entropy(cat_y)
                weight = count / sum(counts)
                weights.append(weight)
                total_entropy += cat_entropy * weight

            weights = np.array(weights)
            weights_entropy = - weights @ np.log(weights).T
            if weights_entropy == 0:
                weights_entropy = -1

            ratio.append((root_entropy - total_entropy) / weights_entropy)
        return ratio

    def best_split(self, root_index):
        end = False
        root_entropy = self.entropy(self.y[root_index])
        ratio = self.ratio(root_index,root_entropy)
        if len(set(ratio)) == 1:
            end = True
            best_feature = None
            return best_feature, end
        ratio = np.array(ratio)
        best_feature = np.argmax(ratio)
        return best_feature,end

    def split(self, best_feature, root_index):
        cats = np.unique(self.x[root_index,best_feature])
        indices = [root_index[cat == self.x[root_index,best_feature]] for cat in cats]
        return indices, cats


    def tree_recursive(self, root_index, branch_name, max_depth, current_depth=0):

        formatting = '-' * current_depth

        if all(self.y[root_index] == np.zeros(len(self.y[root_index]))):
            print('%s Depth %d, %s: negative'%(formatting, current_depth, branch_name))
            return
        if all(self.y[root_index] == np.ones(len(self.y[root_index]))):
            print('%s Depth %d, %s: positive'%(formatting, current_depth, branch_name))
            return

        if current_depth == max_depth:
            formatting = ' ' * current_depth + '-' * current_depth
            print(formatting, '%s leaf node with indices' %branch_name,
                  root_index, ', with label %s' %self.y[root_index],', hit the max depth')
            return

        best_feature, end = self.best_split(root_index)
        self.tree.append((current_depth, branch_name, best_feature, root_index))
        if end:
            formatting = ' ' * current_depth + '-' * current_depth
            print(formatting, '%s leaf node with indices' % branch_name,
                  root_index, ', with label %s' % self.y[root_index], ', all features have the same ratio, end')
            return

        print('%s Depth %d, %s: Split on feature: %d'%(formatting, current_depth,
                                                       branch_name, best_feature))


        indices, cats = self.split(best_feature,root_index)
        name = 0
        for i in indices:
            self.tree_recursive(i, self.ls[best_feature][cats[name]], max_depth, current_depth+1)
            name += 1
        return


class NeuralNetwork:
    """
    Notes:
        * only for o and 1 binary classification
        * activation func is sigmoid function, loss func is Cross-Entropy
        * three hidden layers, number of neurons can be adjusted in xavier_init

    Examples:
        nn = utils.NeuralNetwork(x_train,y_train,epoch=3000,learning_rate=0.001)
        nn.fit()
        pred = nn.predict(x_test)
        utils.plot(nn,x_test,y_test)
        print('precision:', np.mean(pred[0] == y_test))
    """
    def __init__(self, x, y, learning_rate=0.1, epoch=50):
        self.epoch = epoch
        self.alpha = learning_rate
        self.m, self.n = x.shape
        self.x, self.y = x.T, y.reshape(1,self.m)
        self.xavier_init()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def de_sigmoid(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def delta(self, w, ori_delta, z):
        return w.T @ ori_delta * self.de_sigmoid(z)

    def xavier_init(self):
        n1 = 16     # number of neurons in the first layer
        n2 = 16
        n3 = 4
        self.w1 = np.random.normal(0,(2/(n1+self.n))**(1/4),size=(n1,self.n))
        self.w2 = np.random.normal(0,(2/n1+n2)**(1/4),size=(n2,n1))
        self.w3 = np.random.normal(0,(2/n3+n2)**(1/4),size=(n3,n2))
        self.w4 = np.random.normal(0,(2/n3+1)**(1/4),size=(1,n3))
        self.b1 = np.random.normal(0,(2/(n1+self.n))**(1/4),size=(n1,self.m))
        self.b2 = np.random.normal(0,(2/n1+n2)**(1/4),size=(n2,self.m))
        self.b3 = np.random.normal(0,(2/n3+n2)**(1/4),size=(n3,self.m))
        self.b4 = np.random.normal(0,(2/n3+1)**(1/4),size=(1,self.m))

    def fwprop(self):
        self.z1 = self.w1 @ self.x + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.w2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = self.w3 @ self.a2 + self.b3
        self.a3 = self.sigmoid(self.z3)
        self.z4 = self.w4 @ self.a3 + self.b4
        self.a4 = self.sigmoid(self.z4)

    def bwprop(self):
        eps = 1e-8 # avoid dividing 0
        de_crossentropy = (1-self.y) / (eps + 1-self.a4) - self.y / (eps + self.a4)
        self.delta4 = de_crossentropy * self.de_sigmoid(self.z4)
        self.delta3 = self.delta(self.w4, self.delta4, self.z3)
        self.delta2 = self.delta(self.w3, self.delta3, self.z2)
        self.delta1 = self.delta(self.w2, self.delta2, self.z1)

    def gd(self):
        self.w4 -= self.alpha * self.delta4 @ self.a3.T
        self.w3 -= self.alpha * self.delta3 @ self.a2.T
        self.w2 -= self.alpha * self.delta2 @ self.a1.T
        self.w1 -= self.alpha * self.delta1 @ self.x.T
        self.b4 -= self.alpha * self.delta4
        self.b3 -= self.alpha * self.delta3
        self.b2 -= self.alpha * self.delta2
        self.b1 -= self.alpha * self.delta1

    def loss(self):
        return - (1-self.y) @ (np.log(1-self.a4)).T - self.y @ (np.log(self.a4)).T

    def fit(self):
        loss = []
        for i in range(self.epoch):
            self.fwprop()
            self.bwprop()
            self.gd()
            loss.append(self.loss()[0][0])
        plot_loss(loss)

    def predict(self, x):
        x = x.T
        z1 = self.w1 @ x + np.mean(self.b1,axis=1,keepdims=True)
        a1 = self.sigmoid(z1)
        z2 = self.w2 @ a1 + np.mean(self.b2,axis=1,keepdims=True)
        a2 = self.sigmoid(z2)
        z3 = self.w3 @ a2 + np.mean(self.b3,axis=1,keepdims=True)
        a3 = self.sigmoid(z3)
        z4 = self.w4 @ a3 + np.mean(self.b4,axis=1,keepdims=True)
        a4 = self.sigmoid(z4)
        return np.round(a4)


class GMM:
    """
    Gaussian Mixture Model
    Notes:
        # if it's semi supervised, z must be given

    Args:
        k: number of categories
        alpha: weight for the labeled examples

    Examples:
        a) unsupervised:
            gmm = utils.GMM(x_train)
            gmm.fit()
            pred = gmm.predict()
            plt.figure(figsize=(12,8))
            plt.scatter(x_train[:,0],x_train[:,1],c=pred)
            plt.show()
        b) semi-supervised:
            gmm = utils.GMM(x_train, y_train, is_semi_supervised=True)
            gmm.fit()
            pred = gmm.predict()
            plt.figure(figsize=(12,8))
            plt.scatter(x_train[y_train!=-1,0], x_train[y_train!=-1, 1], c=y_train[y_train!=-1])
            plt.scatter(x_train[y_train==-1,0],x_train[y_train==-1,1],c=pred)
            plt.show()
    """
    def __init__(self, x, z=None, is_semi_supervised=False, k=4, alpha=20):
        self.z = z
        self.is_semi_supervised = is_semi_supervised
        if self.is_semi_supervised:
            UNLABELED = -1
            labeled_idx = (z != UNLABELED)
            self.x = x[~labeled_idx, :]
            self.m, self.n = self.x.shape
            self.x_tilde = x[labeled_idx, :]
        else:
            self.m, self.n = x.shape
            self.x = x
        self.mu = []
        self.sigma = []
        self.k = k
        self.alpha = alpha  # Weight for the labeled examples
        # Initialize mu, sigma, phi and w
        self._init()

    def _init(self):
        # Initialize mu and sigma by splitting the m data points uniformly at random
        # into k groups, then calculating the sample mean and covariance for each group
        index = np.arange(0, self.m)
        np.random.shuffle(index)
        number = self.m // self.k

        for i in range(self.k):
            start = i * number
            end = start + number if i != self.k - 1 else self.m
            data = self.x[index[start:end], :]
            self.mu.append(np.mean(data, axis=0))
            self.sigma.append(np.cov(data, rowvar=False))

        # Initialize phi by placing equal probability on each Gaussian
        # phi should be a numpy array of shape (k,)
        phi = np.ones(self.k)
        self.phi = phi / self.k

        # Initialize w by placing equal probability on each Gaussian
        # w should be a numpy array of shape (m, k)
        w = np.ones((self.m, self.k))
        self.w = w / self.k

    def fit(self):
        """
        Args:
            self.x: shape (m, n)
            self.x_tilde: shape (m_tilde, n)
            self.z: shape (m_tilde, )
            self.w: shape (m, k)
            self.phi: shape (k, )
            self.mu: k arrays of shape (n, )
            self.sigma: k arrays of shape (n, n)
        """
        eps = 1e-3  # Convergence threshold
        max_iter = 1000

        if self.is_semi_supervised:
            UNLABELED = -1
            labeled_idx = (self.z != UNLABELED)
            z = self.z[labeled_idx]

            it = 0
            ll = prev_ll = None
            m_tilde, n_tilde = self.x_tilde.shape

            temp = np.zeros((self.m ,self.k))
            temp_super = np.zeros(self.k)
            label_sum = np.zeros(self.k)

            while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
                for i in range(self.k):
                    label_sum[i] = np.sum(z == i)

                # E-step: Update w
                for i in range(self.k):
                    tempj = np.exp(- 1 / 2 * (self.x - self.mu[i].reshape(1, self.n)) @ np.linalg.inv(self.sigma[i]) @ (
                                self.x - self.mu[i].reshape(1, self.n)).T).diagonal()
                    temp[:, i] = tempj
                w = temp * self.phi.reshape(1, self.k) * (1 / np.sqrt(np.linalg.det(self.sigma))).reshape(1, -1)
                row_sum = np.sum(w, axis=1, keepdims=True)
                self.w = w / row_sum

                # M-step: Update phi, mu, and sigma
                self.phi = (np.sum(self.w, axis=0) + self.alpha * label_sum) / (self.m + self.alpha * m_tilde)
                for i in range(self.k):
                    self.mu[i] = (self.w[:, i].reshape(1, -1) @ self.x + self.alpha * np.sum(self.x_tilde[z == i, :], axis=0)) / (
                                np.sum(self.w[:, i]) + self.alpha * label_sum[i])
                    self.sigma[i] = (((self.x.T - self.mu[i].reshape(self.n, 1)) * self.w[:, i].reshape(1, -1)) @ (self.x - self.mu[i].reshape(1, self.n))
                                + self.alpha * (self.x_tilde[z == i, :].T - self.mu[i].reshape(self.n, 1)) @ (
                                            self.x_tilde[z == i, :] - self.mu[i].reshape(1, self.n))) / (np.sum(
                        self.w[:, i]) + self.alpha * label_sum[i])

                # Compute the log-likelihood of the data to check for convergence
                for i in range(self.k):
                    tempj = np.exp(- 1 / 2 * (self.x - self.mu[i].reshape(1, self.n)) @ np.linalg.inv(self.sigma[i]) @ (
                                self.x - self.mu[i].reshape(1, self.n)).T).diagonal()
                    temp[:, i] = tempj
                prev_ll = ll
                ll = np.sum(np.log(np.sum(
                    temp * self.phi.reshape(1, self.k) * (1 / np.sqrt(np.linalg.det(self.sigma))).reshape(1, -1) * (2 * np.pi) ** (
                                - self.k / 2),
                    axis=1)))

                for i in range(self.k):
                    tempj = (np.exp(
                        - 1 / 2 * (self.x_tilde[z == i] - self.mu[i].reshape(1, self.n)) @ np.linalg.inv(self.sigma[i]) @ (self.x_tilde[z == i]
                                                                                                       - self.mu[i].reshape(
                                    1, self.n)).T).diagonal() * (1 / np.sqrt(np.linalg.det(self.sigma[i])))
                             * (2 * np.pi) ** (-self.k / 2)) * label_sum[i] / m_tilde
                    temp_super[i] = np.sum(np.log(tempj))
                l_super = self.alpha * np.sum(temp_super)

                ll = ll + l_super
                it += 1
        else:
            it = 0
            ll = prev_ll = None
            temp = np.zeros((self.m, self.k))
            while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
                # E-step: Update w
                for i in range(self.k):
                    tempj = np.exp(- 1 / 2 * (self.x - self.mu[i].reshape(1, self.n)) @ np.linalg.inv(self.sigma[i]) @ (
                                self.x - self.mu[i].reshape(1, self.n)).T).diagonal()
                    temp[:, i] = tempj
                w = temp * self.phi.reshape(1, self.k) * (1 / np.sqrt(np.linalg.det(self.sigma))).reshape(1, -1)
                row_sum = np.sum(w, axis=1, keepdims=True)
                self.w = w / row_sum

                # M-step: Update phi, mu, and sigma
                self.phi = np.sum(self.w, axis=0) / self.m
                for i in range(self.k):
                    self.mu[i] = self.w[:, i].reshape(1, self.m) @ self.x / np.sum(self.w[:, i])
                    self.sigma[i] = ((self.x.T - self.mu[i].reshape(self.n, 1)) * self.w[:, i].reshape(1, self.m)) @ (
                                self.x - self.mu[i].reshape(1, -1)) / np.sum(self.w[:, i])

                # Compute the log-likelihood of the data to check for convergence
                for i in range(self.k):
                    tempj = np.exp(- 1 / 2 * (self.x - self.mu[i].reshape(1, -1)) @ np.linalg.inv(self.sigma[i]) @ (
                                self.x - self.mu[i].reshape(1, -1)).T).diagonal()
                    temp[:, i] = tempj
                prev_ll = ll
                ll = np.sum(np.log(np.sum(
                    temp * self.phi.reshape(1, self.k) * (1 / np.sqrt(np.linalg.det(self.sigma))).reshape(1, -1) * (2 * np.pi) ** (
                                -self.k / 2), axis=1)))
                it += 1
        return

    def predict(self):
        return np.argmax(self.w, axis=1)


class KMeans:
    """
    Examples(image compression):
        kmeans = utils.KMeans(path=your image path, is_image=True)
        img = kmeans.fit()
        plt.imshow(img)
        plt.show()
    """
    def __init__(self, x=None, path=None, is_image=False, K=16, max_iter=50):
        self.is_image = is_image
        if is_image:
            # Reshape img to (m, 3)
            ori_img = plt.imread(path)
            ori_img = ori_img / 255  # Make sure all values are between 0-1
            self.x = np.reshape(ori_img, (ori_img.shape[0] * ori_img.shape[1], 3))
            self.ori_shape = ori_img.shape
        else:
            self.x = x
        self.k = K
        self.max_iter = max_iter

    def fit(self):
        m, n = self.x.shape
        centroids = self.x[np.random.randint(0, m, size=self.k)]
        c = np.zeros(m)
        it = 0
        eps = 1e-3
        J = 0
        prev_J = None

        while it < self.max_iter and (prev_J is None or np.abs(J - prev_J) >= eps):
            prev_J = J
            for i in range(m):
                distances = np.linalg.norm(self.x[i] - centroids, ord=2, axis=1)
                c[i] = np.argmin(distances)

            for i in range(self.k):
                centroids[i] = np.mean(self.x[c==i], axis=0)
                J += np.sum(np.linalg.norm(self.x[c==i] - centroids[i], ord=2, axis=1))
            it += 1

        for i in range(m):
            self.x[i] = centroids[int(c[i])]
        if self.is_image:
            self.x = self.x.reshape(self.ori_shape)
            self.x = self.x * 255
            self.x = self.x.round( ).astype(np.uint8)
        return self.x


class CNN:
    """
    Examples:
        def read_data(images_file, labels_file):
            x = np.loadtxt(images_file, delimiter=',')
            y = np.loadtxt(labels_file, delimiter=',')
            x = np.reshape(x, (x.shape[0], 1, 28, 28))
            return x, y
        data, labels = read_data('../data/images_train.csv', '../data/labels_train.csv')
        cnn = cnn.CNN(data, labels)
        cnn.fit()

    Args:
        data: size: number * channels * image_width * image_height(would be auto split to train and validation set)
        labels: size: number * labels(would be auto transformed to one_hot_encoding)

    Architectures:
        convolutional layer: convolution_filters * input_channels * convolution_size * convolution_size ->
        max pooling layer: max_pool_size * max_pool_size -> ReLu activation layer -> flatten ->
        linear layer: output size: categories -> softmax layer: compute probabilities for each class ->
        cross entropy loss as loss function
    """

    def __init__(self, x, y, max_pool_size=5, convolution_size=4, convolution_filters=2, categories=10, num_batches=400
                 , batch_size=16, learning_rate=1e-2):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.x, self.y = x, y
        self.data_preprocess()
        self.max_pool_size = max_pool_size
        self.convolution_size = convolution_size
        self.convolution_filters = convolution_filters
        self.categories = categories
        _, self.input_channels, self.input_size, self.input_height = x.shape
        self.init_parameters()

    def data_preprocess(self):
        self.y = one_hot_encoding(self.y)
        mean = np.mean(self.x)
        std = np.std(self.x)
        self.x = (self.x - mean) / std
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.y = self.y[p]
        split = self.batch_size * 25  # we test validation error every 100 iteration, thus multi 100/4
        self.x_train = self.x[split:]
        self.y_train = self.y[split:]
        self.x_valid = self.x[:split]
        self.y_valid = self.y[:split]

    def init_parameters(self):
        conv_output_size = self.input_size - self.convolution_size + 1
        max_pool_output_size = conv_output_size // self.max_pool_size
        linear_input_size = self.convolution_filters * max_pool_output_size * max_pool_output_size
        self.conv_w = np.random.normal(size=(self.convolution_filters, self.input_channels, self.convolution_size,
                                             self.convolution_size), scale=1/np.sqrt(self.convolution_size *
                                                                                     self.convolution_size))
        self.conv_b = np.zeros(self.convolution_filters)
        self.linear_w = np.random.normal(size=(linear_input_size, self.categories), scale=1/np.sqrt(linear_input_size))
        self.linear_b = np.zeros(self.categories)

    def forward_conv(self, data):
        outputchannels, _, conv_width, conv_height = self.conv_w.shape
        input_channels, input_width, input_height = data.shape
        output_width = input_width - conv_width + 1
        output_height = input_height - conv_height + 1
        output = np.zeros((outputchannels, output_width, output_height))

        for x in range(output_width):
            for y in range(output_height):
                for outputchannel in range(outputchannels):
                    output[outputchannel, x, y] = np.sum(data[:, x:(x+conv_width), y:(y+conv_height)] *
                                                          self.conv_w[outputchannel, :, :, :]) + self.conv_b[outputchannel]

        return output

    def forward_max_pooling(self, data):
        channels, width, height = data.shape
        output_width = width // self.max_pool_size
        output_height = height // self.max_pool_size
        output = np.zeros((channels, output_width, output_height))
        for x in range(0, width, self.max_pool_size):
            for y in range(0, height, self.max_pool_size):
                output[:, x//self.max_pool_size, y//self.max_pool_size] = np.max(data[:,x:(x+self.max_pool_size),
                                                                                 y:(y+self.max_pool_size)], axis=(1,2))
        return output

    def forward_relu(self, data):
        data[data<=0] = 0
        return data

    def forward_linear(self, data):
        return data @ self.linear_w + self.linear_b

    def forward_softmax(self, x):
        # minus the max term to avoid overflow
        x = x - np.max(x, axis=0)
        exp = np.exp(x)
        s = exp / np.sum(exp, axis=0)
        return s

    def forward_cross_entropy(self, labels, data):
        return - np.log(data[labels==1])

    def backward_cross_entropy_loss(self, probabilities, labels):
        return -labels / probabilities

    def backward_softmax(self, x, grad_outputs):
        return self.forward_softmax(x) - (grad_outputs != 0).astype(int)

    def backward_cross_entropy_and_softmax(self, probabilities, labels):
        return probabilities - labels

    def backward_linear(self, grad, data):
        grad_w = data.reshape(-1, 1) * grad.reshape(1, -1)
        grad_b = grad.reshape(self.linear_b.shape)
        grad_data = (grad @ self.linear_w.T).reshape(data.shape)
        return grad_w, grad_b, grad_data

    def backward_relu(self, data, grad):
        # grad must be the same shape as the output of RelU
        grad[data<=0] = 0
        return grad

    def backward_max_pool(self, data, grad):
        channels, width, height = data.shape
        grad_data = np.zeros(data.shape)
        for i in range(channels):
            for x in range(0, width, self.max_pool_size):
                for y in range(0, height, self.max_pool_size):
                    t = data[i, x:(x+self.max_pool_size), y:(y+self.max_pool_size)]
                    max_pos = np.unravel_index(np.argmax(t), t.shape)
                    grad_data[i, x + max_pos[0], y + max_pos[1]] = grad[
                        i, x // self.max_pool_size, y // self.max_pool_size]
        return grad_data

    def backward_convolution(self, data, grad):
        grad_b = np.sum(grad, axis=(1,2))
        grad_w = np.zeros(self.conv_w.shape)
        grad_data = np.zeros(data.shape)
        output_channels, output_width, output_height = grad.shape

        for x in range(output_width):
            for y in range(output_height):
                for channel in range(output_channels):
                    grad_w[channel, :, :, :] += grad[channel, x, y] * data[:, x:(x+self.convolution_size), y:(y+self.convolution_size)]
                    grad_data[:, x:(x+self.convolution_size), y:(y+self.convolution_size)] += self.conv_w[channel, :, :, :] * grad[channel, x, y]
        return grad_w, grad_b, grad_data

    def forward_propagation(self, data, labels):
        # use for compute validation loss
        conv_o = self.forward_conv(data)
        max_pool_o = self.forward_max_pooling(conv_o)
        relu_o = self.forward_relu(max_pool_o)
        flatten_o = relu_o.reshape(-1)
        linear_o = self.forward_linear(flatten_o)
        softmax_o = self.forward_softmax(linear_o)
        loss = self.forward_cross_entropy(labels,softmax_o)
        return loss, softmax_o

    def backward_propagation(self, data, labels):
        conv_o = self.forward_conv(data)
        max_pool_o = self.forward_max_pooling(conv_o)
        relu_o = self.forward_relu(max_pool_o)
        flatten_o = relu_o.reshape(-1)
        linear_o = self.forward_linear(flatten_o)
        softmax_o = self.forward_softmax(linear_o)
        loss = self.forward_cross_entropy(labels,softmax_o)

        grad_CE_loss = self.backward_cross_entropy_loss(softmax_o, labels)
        grad_softmax = self.backward_softmax(linear_o, grad_CE_loss)
        linear_w_grad, linear_b_grad, grad_linear = self.backward_linear(grad_softmax, flatten_o)
        grad_relu = self.backward_relu(max_pool_o, grad_linear.reshape(relu_o.shape))
        grad_max_pool = self.backward_max_pool(conv_o, grad_relu)
        conv_w_grad, conv_b_grad, _ = self.backward_convolution(data, grad_max_pool)

        return loss, {'conv_w':conv_w_grad, 'conv_b':conv_b_grad, 'linear_w':linear_w_grad, 'linear_b':linear_b_grad}

    def batch_gradient_descend(self, batch_data, batch_labels):
        total_grad = {}
        loss_total = 0
        for i in range(len(batch_data)):
            loss, grad = self.backward_propagation(batch_data[i], batch_labels[i])
            for key, value in grad.items():
                if key not in total_grad:
                    total_grad[key] = np.zeros(value.shape)
                total_grad[key] += value
            loss_total += loss

        self.conv_w -= self.learning_rate * total_grad['conv_w']
        self.conv_b -= self.learning_rate * total_grad['conv_b']
        self.linear_w -= self.learning_rate * total_grad['linear_w']
        self.linear_b -= self.learning_rate * total_grad['linear_b']
        return loss_total / len(batch_data)

    def accuracy(self, x, y):
        accuracy = 0
        for data, label in zip(x,y):
            _, pred = self.forward_propagation(data, label)
            accuracy += (np.argmax(pred)==np.argmax(label))
        return accuracy / len(x)

    def plot_loss(self, loss_history, ax):
        ax.clear()
        x = np.arange(1, len(loss_history) + 1)
        ax.plot(x, loss_history)
        ax.set_xlabel('Iterations of each batch')
        ax.set_ylabel('Loss')
        plt.pause(0.1)

    def fit(self):
        loss_history = []
        iter = 0

        fig, ax = plt.subplots()
        plt.ion()

        for i in range(0, self.num_batches):
            loss = self.batch_gradient_descend(self.x_train[i*self.batch_size:(i+1)*self.batch_size], self.y_train[i*self.batch_size:(i+1)*self.batch_size])
            loss_history.append(loss)
            iter+=1
            print('current iteration:', iter)
            if iter % 50 == 0:
                print('validation accuracy:', self.accuracy(self.x_valid, self.y_valid))
            if iter % 10 == 0:
                self.plot_loss(loss_history, ax)

        plt.ioff()
        self.plot_loss(loss_history, ax)
        plt.show()


class ICA:
    """
    Independent Components Analysis
    Examples:
        ica = utils.ICA(data)
        ica.fit()
    Notes:
        * use for transforming mixing audios(if you want to split four types of audios you have to give four feature)
         into original audios
    Args:
        x: size: number of samples * types of audios wanted to split
        func: laplace function or sigmoid function
    """
    def __init__(self, x, func='laplace'):
        self.x = x
        self.m, self.n = self.x.shape
        self.func = func
        self.learning_rate =  [0.1 , 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01 ,
                               0.01, 0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
        self.init_parameters()

    def init_parameters(self):
        self.w = np.eye(self.n)
        self.x = 0.99 * self.x / np.max(np.abs(self.x))  # it's important in dealing with audio

    def update_w(self):
        if self.func == 'laplace':
            for learning_rate in self.learning_rate:
                index = np.random.permutation(self.m)  # it's important in dealing with time series data
                for i in index:
                    self.w = self.w + learning_rate * (np.linalg.inv(self.w.T) - (
                        np.sign(self.w @ self.x[i])).reshape(-1, 1) @ self.x[i].reshape(1, -1))
        elif self.func == 'sigmoid':
            for learning_rate in self.learning_rate:
                index = np.random.permutation(self.m)
                for i in index:
                    self.w = self.w + learning_rate * (np.linalg.inv(self.w.T) +
                                                       np.outer(1 - 2 * sigmoid(self.w @ self.x[i],
                                                                                if_single_input=True), self.x[i].T))

    def fit(self):
        Fs = 11025  # sample rate
        self.update_w()
        s = self.x @ self.w.T
        s = 0.99 * s / np.max(np.abs(s))
        for i in range(s.shape[1]):
            plt.plot(s[:, i])
            plt.show()
            if os.path.exists('split_{}'.format(i)):
                os.unlink('split_{}'.format(i))
            scipy.io.wavfile.write('../output/split_{}.wav'.format(i), Fs, s[:,i])
        print('successfully split')


class PCA:
    """Principal Components Analysis"""
    def __init__(self, x, k=3):
        self.x = x
        self.m, self.n = self.x.shape
        self.k = k
        return

    def init_parameters(self):
        mean = np.mean(self.x, axis=0, keepdims=True)
        std = np.std(self.x, axis=0, keepdims=True)
        self.x = (self.x - mean) / std
        return

    def fit(self):
        s = np.cov(self.x, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(s)
        sorted_index = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_index][:self.k]
        eigen_vectors = eigen_vectors[:, sorted_index]
        eigen_vectors = eigen_vectors[:, :self.k]
        x = self.x @ eigen_vectors
        return x, eigen_values


class StandardScaler:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit_transform(self, x):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        x = (x-self.mean) / self.std
        return x

    def inverse_transform(self, x):
        x = self.std * x + self.mean
        return x


def mse(x, y):
    mse = np.mean((x - y) ** 2)
    return mse


















