import numpy as np
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    def __init__(self, M):
        self.M = M

    def fit(self, X, max_iter=30, seed=42):
        np.random.seed(seed)

        V = max(max(x) for x in X) + 1
        N = len(X)

        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.B = random_normalized(self.M, V)

        costs = []
        for it in range(max_iter):
            if it % 10 == 0: # print every 10 epochs
                print("it:", it)

            alphas = []
            betas = []
            P = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi * self.B[:, x[0]]
                for t in range(1, T):
                    alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                P[n] = alpha[-1].sum()
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]]) * beta[t+1]
                betas.append(beta)

            cost = np.sum(np.log(P))
            costs.append(cost)

            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range(N)) / N

            den1 = np.zeros((self.M, 1))
            den2 = np.zeros((self.M, 1))
            a_num = 0
            b_num = 0
            for n in range(N):
                x = X[n]
                T = len(x)

                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]

                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i, j] += alphas[n][t, i] * self.A[i, j] * self.B[j, x[t+1]] * betas[n][t+1, j]
                a_num += a_num_n / P[n]

                b_num_n = np.zeros((self.M, V))
                for i in range(self.M):
                    for j in range(V):
                        for t in range(T):
                            if x[t] == j:
                                b_num_n[i, j] += alphas[n][t, i] * betas[n][t, i]
                b_num += b_num_n / P[n]
            self.A = a_num / den1
            self.B = b_num / den2

        print("A:", self.A)
        print("B:", self.B)
        print("pi", self.pi)

        plt.plot(costs)
        plt.show()

    def likelihood(self, x):
        T = len(x)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
        return alpha[-1].sum()

    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

    def get_state_sequence(self, x):    # Viterbi
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi * self.B[: x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t, j] = np.max(delta[t-1] * self.A[:, j]) * self.B[j, x[t]]
                psi[t, j] = np.argmax(delta[t-1] * self.A[:, j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states


def fit_coin():
    X = []
    for line in open('coin_data.txt'):
        # 1 for H, 0 for T
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.log_likelihood_multi(X).sum()
    print("LL with fitted params:", L)

    # try true values
    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = hmm.log_likelihood_multi(X).sum()
    print("LL with true params:", L)

    # try viterbi
    print("Best state sequence for:", X[0])
    print(hmm.get_state_sequence(X[0]))


if __name__ == '__main__':
    fit_coin()
