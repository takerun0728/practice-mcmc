import numpy as np

class RBM:

    def __init__(self, n_v, n_h):
        self.w = np.random.randn(n_h, n_v)
        self.b = np.random.randn(n_h, 1)
        self.a = np.random.randn(n_v, 1)

    def train(self, V, epsilon, epoch, T):
        for epo in range(epoch):
            for (n, v_0) in enumerate(V.T):
                v_0 = np.copy(v_0).reshape(-1, 1)
                p_h_0 = np.copy(self.sigmoid(self.w.dot(v_0) + self.b))

                v, p_h = self.encode_decode(np.copy(v_0), T)
                self.update(v_0, v, p_h_0, p_h, epsilon)

    def encode_decode(self, v, T):
        for t in range(T):
            # visible
            p_h = self.sigmoid(self.w.dot(v) + self.b)
            h = (np.random.rand(n_h, 1) < p_h).astype('float64')

            # hidden
            p_v = self.sigmoid(self.w.T.dot(h) + self.a)
            v = (np.random.rand(n_v, 1) < p_v).astype('float64')

        return (v, p_h)

    def update(self, v_0, v, p_h_0, p_h, epsilon):
        self.w += epsilon * (v_0.T * p_h_0 - v.T * p_h)
        self.a += epsilon * (v_0 - v)
        self.b += epsilon * (p_h_0 - p_h)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    # init parameters
    N = 20
    side = 32
    n_v = side ** 2
    n_h = 32
    V = np.round(np.random.rand(n_v, N))
    T = 5

    # train
    rbm = RBM(n_v, n_h)
    rbm.train(V, 0.1, 200, T)