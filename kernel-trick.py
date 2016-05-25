import matplotlib.pyplot as plt
from sklearn import datasets, svm
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import plot_decision_regions

np.random.seed(0)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

clf = svm.SVC(kernel='rbf', random_state=0, gamma=0.3, C=10.0)
clf.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor, clf=clf)


# plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
# plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')

# plt.ylim(-3.0)
# plt.legend()
plt.show()