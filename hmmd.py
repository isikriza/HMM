import numpy as np
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M):
        self.M = M

    def fit(self, X, max_iter=30):
        np.random.seed(42)

        V = max(max(x) for x in X) + 1
        N = len(X)

        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.B = random_normalized(self.M, V)