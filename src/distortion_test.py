import numpy as np
import scipy.ndimage as sp_ndi
from mnist_loader import load_mnist_simple
from matplotlib import pyplot as plt


def find_transform(x01, y01, x02, y02, x03, y03, x1, y1, x2, y2, x3, y3):
    D = x03 * (y01 - y02) + x01 * (y02 - y03) + x02 * (y03 - y01)
    A1 = (x1 * (y02 - y03) + x2 * (y03 - y01) + x3 * (y01 - y02))
    A2 = -(x1 * (x02 - x03) + x2 * (x03 - x01) + x3 * (x01 - x02))
    A3 = (y1 * (y02 - y03) + y2 * (y03 - y01) + y3 * (y01 - y02))
    A4 = -(y1 * (x02 - x03) + y2 * (x03 - x01) + y3 * (x01 - x02))

    B1 = (x1 * (x02 * y03 - x03 * y02) + x2 * (x03 * y01 - x01 * y03) + x3 * (x01 * y02 - x02 * y01))
    B2 = (y1 * (x02 * y03 - x03 * y02) + y2 * (x03 * y01 - x01 * y03) + y3 * (x01 * y02 - x02 * y01))

    A = np.array([[A1, A2], [A3, A4]]) / D
    b = np.array([B1, B2]) / D

    return A, b


def prepare_digit():
    train, _, _ = load_mnist_simple()
    digit = train[1][0].reshape((28, 28))
    np.save('../data/digit.npy', digit)


def distort(m, A, b):
    result = np.zeros(m.shape)
    for i, row in enumerate(result):
        for j, p in enumerate(row):
            new_indices = A @ np.array([i, j]) + b
            i0, j0 = new_indices

            try:
                result[i, j] = m[int(i0), int(j0)]
            except IndexError:
                print(f"out of bounds: ({i}, {j}) -> ({i0}, {j0})")
    return result


def distort2(m, A, b):
    def f(x0):
        r = A @ x0 + b
        return tuple(r)

    r = sp_ndi.geometric_transform(m, f, prefilter=False)
    return r


if __name__ == '__main__':
    # prepare_digit()
    digit = np.load('../data/digit.npy')
    A, b = find_transform(0, 0, 27, 0, 13.5, 27,
                          27, 0, 0, 0, 23.5, 27)
    plt.subplot(2, 1, 1)
    plt.matshow(digit, fignum=False)
    plt.subplot(2, 1, 2)
    # plt.matshow(distort(digit, A, b), fignum=False)
    plt.matshow(distort2(digit, A, b), fignum=False)
    # plt.matshow(sp_ndi.affine_transform(digit, A, b), fignum=False)
    plt.show()
