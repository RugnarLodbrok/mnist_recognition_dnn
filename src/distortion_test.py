import numpy as np
import scipy.ndimage as sp_ndi

from expand_data import generate_distortions, transform_f
from mnist_loader import load_mnist_simple, load_data, DATA_PATH
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
    digit = train[7][0].reshape((28, 28))
    np.save('../data/digit.npy', digit)


def transforms_to_show(digit):
    for a in generate_distortions():
        yield sp_ndi.geometric_transform(digit, transform_f(a), prefilter=False)


def distort_digit():
    prepare_digit()
    digit = np.load('../data/digit.npy')
    n, m = 4, 4
    total = n * m
    plt.subplot(n, m, 1)
    plt.matshow(digit, fignum=False)
    for i, distorted_digit in enumerate(transforms_to_show(digit), start=2):
        if i > total:
            print("no space on the plot left")
            break
        plt.subplot(n, m, i)
        plt.matshow(distorted_digit, fignum=False)
    plt.show()


def all_digits_average():
    from functools import reduce
    from operator import add
    train, test, valid = load_mnist_simple((28, 28))
    xx, _ = zip(*train, *test, *valid)
    s = reduce(add, xx)
    print(s.shape)
    plt.matshow(s)
    plt.show()


def check_generation():
    from functools import reduce
    from operator import add
    digit = np.load('../data/digit.npy')
    s = reduce(add,
               (sp_ndi.geometric_transform(digit, transform_f(a), prefilter=False) for a in generate_distortions()))
    m = (np.max(s) - np.min(s)) / 2
    print(m)
    plt.matshow(np.abs(s - m))
    plt.show()


def check_dataset():
    data = load_data(f"{DATA_PATH}/mnist_expaned_k00.pkl.gz")
    print(len(data))
    x, y = data[4324]
    plt.matshow(x)
    plt.show()


if __name__ == '__main__':
    # check_generation()
    # distort_digit()
    check_dataset()
