import numpy as np


class Layer:
    def __init__(self, n, activation):
        self.n = n
        self.w = None
        self.b = None

        self.x = None
        self.z = None
        self.a = None

        self.sigma = activation.sigma
        self.sigma_prime = activation.sigma_prime
        self.activation = activation

    def delta(self, a, y):
        if hasattr(self.activation, 'delta'):
            return self.activation.delta(a, y, self.z)
        return self.sigma_prime(self.z)*self.activation.cost_prime(a, y)

    def init_matrices(self, m):
        n = self.n
        self.w = np.zeros([n, m])
        self.b = np.zeros([n, 1])  # TODO: check why not zeros([m])

    def feed(self, x):
        self.x = x
        self.z = self.w @ x + self.b
        return self.sigma(self.z)

    def backprop(self, delta, z):
        """
        self is (l+1)th layer
        :param delta: error vector on (l+1)th layer
        :param z: z of l-th layer
        :return: error vector on l-th layer
        """
        return (self.w.T @ delta) * self.sigma_prime(z)


class DropoutLayer(Layer):
    def __init__(self, n, activation):
        super().__init__(n, activation)
        assert n % 2 == 0
        self.percent = 1.
        self._dropout_mask = np.concatenate((
            np.zeros((self.n / 2,)),
            np.ones((self.n / 2,))
        )).astype(np.bool)
        self._full_mask = np.ones((self.n,)).astype(np.bool)
        self.mask = self._full_mask

    def dropout(self):
        np.random.shuffle(self._dropout_mask)
        self.mask = self._dropout_mask

    def dropout_restore(self):
        self.mask = self._full_mask

    def z(self, x):
        return (self.w @ x + self.b) * self.mask

    def feed(self, x):
        return self.sigma((self.w @ x + self.b) * self.mask)
