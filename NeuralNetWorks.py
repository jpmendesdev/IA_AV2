import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, X_train, y_train, learning_rate=1e-3, plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.d = y_train.reshape(-1)
        self.lr = learning_rate
        self.w = np.random.random_sample((self.p + 1, 1)) - .5
        self.plot = plot
        if plot and self.p >= 2:
            self.fig, self.ax = plt.subplots()
            self.ax.scatter(self.X_train[1, self.d == 1], self.X_train[2, self.d == 1], marker='s', s=120, label='+1')
            self.ax.scatter(self.X_train[1, self.d == -1], self.X_train[2, self.d == -1], marker='o', s=120, label='-1')
            self.ax.set_xlim(self.X_train[1, :].min() - 1, self.X_train[1, :].max() + 1)
            self.ax.set_ylim(self.X_train[2, :].min() - 1, self.X_train[2, :].max() + 1)
            self.x1 = np.linspace(self.X_train[1, :].min() - 1, self.X_train[1, :].max() + 1)
            self.ax.legend()
            self.draw_line()

    def draw_line(self, c='k', alpha=1, lw=2):
        if not self.plot or abs(self.w[2, 0]) < 1e-12:
            return
        x2 = -self.w[1, 0] / self.w[2, 0] * self.x1 + self.w[0, 0] / self.w[2, 0]
        self.ax.plot(self.x1, x2, c=c, alpha=alpha, lw=lw)

    def activation_function(self, u):
        return 1 if u >= 0 else -1

    def fit(self, max_epochs=1000):
        epochs = 0
        errors_history = []
        while epochs < max_epochs:
            errors = 0
            for k in range(self.N):
                x_k = self.X_train[:, [k]]
                y_k = self.activation_function((self.w.T @ x_k)[0, 0])
                e_k = self.d[k] - y_k
                if e_k != 0:
                    self.w += self.lr * e_k * x_k
                    errors += 1
            errors_history.append(errors)
            if errors == 0:
                break
            epochs += 1
        return {'errors_history': errors_history, 'epochs': epochs}

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xb = np.vstack((-np.ones((1, X.shape[1])), X))
        u = self.w.T @ Xb
        return np.where(u >= 0, 1, -1).flatten()


class ADALINE:
    def __init__(self, X_train, y_train, learning_rate=1e-3, max_epoch=10000, tol=1e-5, plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.d = y_train.reshape(-1)
        self.lr = learning_rate
        self.max_epoch = max_epoch
        self.tol = tol
        self.w = np.random.random_sample((self.p + 1, 1)) - .5
        self.plot = plot

    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, [k]]
            y = (self.w.T @ x_k)[0, 0]
            s += (self.d[k] - y) ** 2
        return s / (2 * self.N)

    def fit(self):
        eqm_hist = []
        for epoch in range(self.max_epoch):
            for k in range(self.N):
                x_k = self.X_train[:, [k]]
                y = (self.w.T @ x_k)[0, 0]
                e = self.d[k] - y
                self.w += self.lr * e * x_k
            eqm = self.EQM()
            eqm_hist.append(eqm)
            if len(eqm_hist) > 2 and abs(eqm_hist[-1] - eqm_hist[-2]) < self.tol:
                break
        return {'eqm_history': eqm_hist, 'epochs': epoch + 1}

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xb = np.vstack((-np.ones((1, X.shape[1])), X))
        y = self.w.T @ Xb
        return np.where(y >= 0, 1, -1).flatten()

class MultilayerPerceptron:
    def __init__(self, X_train, Y_train, topology, learning_rate=1e-3, tol=1e-6, max_epoch=10000):
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.D = Y_train
        self.lr = learning_rate
        self.tol = tol
        self.max_epoch = max_epoch

        topology = topology + [self.m]
        self.W = []
        for i in range(len(topology)):
            in_size = self.p + 1 if i == 0 else topology[i - 1] + 1
            self.W.append(np.random.uniform(-0.5, 0.5, (topology[i], in_size)))

    def g(self, u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))

    def g_d(self, u):
        y = self.g(u)
        return 0.5 * (1 - y ** 2)

    def forward(self, X, add_bias=True):
        if add_bias:
            X = np.vstack((-np.ones((1, X.shape[1])), X))
        y = X
        U, Y = [], []
        for W in self.W:
            U_i = W @ y
            y = self.g(U_i)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            U.append(U_i)
            Y.append(y)
            y = np.vstack((-np.ones((1, y.shape[1])), y))
        return Y[-1], U, Y

    def fit(self):
        eqm_hist = []
        for epoch in range(self.max_epoch):
            s = 0
            for k in range(self.N):
                x = self.X_train[:, [k]]
                d = self.D[:, [k]]
                y, U, Y = self.forward(x, add_bias=False)
                e = d - y
                s += np.sum(e ** 2)
                delta = [None] * len(self.W)
                delta[-1] = e * self.g_d(U[-1])
                for i in reversed(range(len(self.W) - 1)):
                    delta[i] = self.g_d(U[i]) * (self.W[i + 1][:, 1:].T @ delta[i + 1])
                for i, W in enumerate(self.W):
                    x_b = x if i == 0 else np.vstack((-np.ones((1, 1)), Y[i - 1]))
                    self.W[i] += self.lr * (delta[i] @ x_b.T)
            eqm = s / (2 * self.N)
            eqm_hist.append(eqm)
            if len(eqm_hist) > 2 and abs(eqm_hist[-1] - eqm_hist[-2]) < self.tol:
                break
        return {'eqm_history': eqm_hist, 'epochs': epoch + 1}

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y, _, _ = self.forward(X, add_bias=True)
        return y
