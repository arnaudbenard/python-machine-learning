import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions

class AdalineGD(object):
    """ADAptive LInear NEuron classifier"""
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
y = df.iloc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

# Without feature scaling
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# ax[0].plot(range(1, len(ada1.cost_) + 1),
#     np.log10(ada1.cost_), marker='o')

# ax[0].set_xlabel('Epoches')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline - Learning rate 0.01')

# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# ax[1].plot(range(1, len(ada1.cost_) + 1),
#     np.log10(ada2.cost_), marker='o')

# ax[1].set_xlabel('Epoches')
# ax[1].set_ylabel('log(Sum-squared-error)')
# ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.show()

# With feature scaling
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, clf=ada)
plt.title('Adaline - Gradient descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-square-error')
plt.show()