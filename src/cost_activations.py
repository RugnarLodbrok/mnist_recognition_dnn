import numpy as np
from scipy.special import expit as logistic


class CostActivation:
    @staticmethod
    def sigma(z):
        raise NotImplementedError("Abstract")

    @staticmethod
    def sigma_prime(z):
        raise NotImplementedError("Abstract")

    @staticmethod
    def delta(a, y, z):
        raise NotImplementedError("Abstract")


class LogisticQuadratic(CostActivation):
    """logistic activation + quadratic cost"""

    @staticmethod
    def sigma(z):
        return logistic(z)

    @staticmethod
    def sigma_prime(z):
        s = logistic(z)
        return s * (1 - s)

    @staticmethod
    def delta(a, y, z):
        s = logistic(z)
        return s * (1 - s) * (a - y)


class LogisticCrossEntropy(LogisticQuadratic):
    """logistic activation + corss-entropy cost"""

    @staticmethod
    def delta(a, y, z):
        return a - y


class SoftMax(CostActivation):
    @staticmethod
    def sigma(z):
        e = np.exp(z)
        return e / np.sum(e, axis=0)

    @staticmethod
    def sigma_prime(z):
        e = np.exp(z)
        sigma = e / np.sum(e, axis=0)
        return sigma * (1 - sigma)

    @staticmethod
    def delta(a, y, z):
        return a - y


class ReLU(CostActivation):
    @staticmethod
    def sigma(z):
        return np.maximum(0, z)

    @staticmethod
    def sigma_prime(z):
        return (z > 0).astype(np.int)

    @staticmethod
    def delta(a, y, z):
        raise NotImplementedError
