from random import shuffle

import numpy as np
from py_tools.seq import grouped, cons, pairwise

from layers import DropoutLayer
from mnist_loader import load_data


def _vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class DNN:
    def __init__(self, input, layers, eta, lmbda=0):
        self.eta = eta
        self.lmbda = lmbda
        self.w = []
        self.b = []
        self.layers = []
        for layer, (m, n) in zip(layers, pairwise(cons(input, (l.n for l in layers)))):
            assert layer.n == n
            layer.init_matrices(m)
            if self.layers:
                layer.previous = self.layers[-1]
            self.layers.append(layer)

    def initialize_rand(self):
        for l in self.layers:
            l.init_random()

    def stats(self):
        return ([np.average(l.w) for l in self.layers],
                [np.std(l.w) for l in self.layers],
                [np.average(l.b) for l in self.layers],
                [np.std(l.b) for l in self.layers])

    def _backprop(self, x, y):
        """
        :param x: input
        :param y: output
        :return: nabla_b, nabla_w
        """

        a = x
        for l in self.layers:
            a = l.feed(a)

        # backward pass
        last_layer = self.layers[-1]
        delta = last_layer.delta(a, y)
        nabla_b = [delta]
        nabla_w = [delta @ last_layer.x.T]
        for l in reversed(self.layers[1:]):
            delta = l.backprop(delta)
            nabla_b.append(delta)
            nabla_w.append(delta @ l.previous.x.T)

        return list(reversed(nabla_b)), list(reversed(nabla_w))

    def epoch(self, examples, batch_size):
        """
        :param examples: list of [(x, y)]
        :return:
        """
        regularization = 'L2'
        examples_vectorized = [(x, _vectorized_result(y)) for x, y in examples]
        eta = self.eta  # rate of learning
        lmbda = self.lmbda
        if regularization == 'L2':
            decay = 1 - eta * lmbda / len(examples)

            def regularize(w):
                w *= decay
        elif regularization == 'L1':
            decay = eta * lmbda / len(examples)

            def regularize(w):
                w -= decay * np.sign(w)
        else:
            raise ValueError(regularization)

        for i, batch in enumerate(grouped(examples_vectorized, batch_size)):
            xx, yy = zip(*batch)
            nabla_b, nabla_w = self._backprop(np.concatenate(xx, axis=1), np.concatenate(yy, axis=1))
            for l, nw, nb in zip(self.layers, nabla_w, nabla_b):
                l.b -= eta * np.mean(nb, axis=1).reshape(l.b.shape)
                if lmbda:
                    regularize(l.w)
                l.w -= eta * nw
            # ----

    def learn(self, examples, batch_size=30, epochs=1, test=None):
        self.dropout()
        for i in range(epochs):
            shuffle(examples)
            self.epoch(examples, batch_size)
            if test:
                print(self.test(test))

    def test(self, examples):
        self.dropout_restore()
        success = 0
        for x, y in examples:
            if y == np.argmax(self._fit(x)):
                success += 1
        return success / len(examples)

    def _fit(self, x):
        for l in self.layers:
            x = l.feed(x)
        return x

    def dropout(self):
        for l in self.layers:
            if isinstance(l, DropoutLayer):
                l.dropout()

    def dropout_restore(self):
        for l in self.layers:
            if isinstance(l, DropoutLayer):
                l.dropout_restore()
