from math import sin, cos, pi

import numpy as np
import scipy.ndimage as sp_ndi
from py_tools.seq import conj

from mnist_loader import load_mnist_simple, dump_data, DATA_PATH
from utils import timing


def rot(a):
    s = sin(a)
    c = cos(a)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def dis(offset):
    return np.array([[1, 0, offset[0]],
                     [0, 1, offset[1]],
                     [0, 0, 1]])


def squeeze(d, st, sn):  # todo: tt, tn
    """
    :param d: - direction (degrees)
    :param st: squeeze tangent direction
    :param sn: squeeze normal direction
    :return: transformation matrix
    """
    R = rot(d)
    R1 = np.linalg.inv(R)
    A = np.array([[1 / st, 0, 0],
                  [0, 1 / sn, 0],
                  [0, 0, 1]])
    return R1 @ A @ R


def generate_distortions():
    I = np.eye(3)
    D = np.array([[1, 0, 13.5],
                  [0, 1, 13.5],
                  [0, 0, 1]])
    D1 = np.linalg.inv(D)
    rotations = [rot(a) for a in [-0.2, -0.1, 0., 0.1, 0.2]]
    displacements = [dis(x) for x in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]]  # add diagonals?
    squeezees = []
    for a in np.linspace(0, pi/2, 4)[:-1]:
        for s1 in [1, 1.1]:
            for s2 in [1, 1.1]:
                if s1 == 1 and s2 == 1 and a:
                    continue
                squeezees.append(squeeze(a, s1, s2))

    for r in rotations:
        for d in displacements:
            for s in squeezees:
                yield D @ r @ d @ s @ D1


def transform_f(A):
    def transform(x):
        x = np.array(list(conj(x, 1)))
        r = tuple((A @ x)[:2])
        assert len(r) == 2
        return r

    return transform


def expand(data):
    # transforms = [transform_f(A) for A in generate_distortions()]
    transforms = list(generate_distortions())
    total = len(data)
    for i, (x, y) in enumerate(data):
        for f in transforms:
            yield sp_ndi.affine_transform(x, f, prefilter=False), y
            # yield sp_ndi.geometric_transform(x, f, prefilter=False), y
        print(f"{i}/{total} done")


if __name__ == '__main__':
    train, test, validate = load_mnist_simple(shape=(28, 28))
    # expand(train)
    # dump_mnist('mnist_test_dump.pkl.gz', train)
    chunk = 1000
    for i in range(0, 50):
        a = i * chunk
        b = (i + 1) * chunk
        print(a, b)
        data = list(expand(train[a:b]))
        # with timing("npz"):
        #     np.savez_compressed(f'{DATA_PATH}/mnist_expaned_k0{i}.npz', data)
        with timing("pickle gz"):
            dump_data(f'{DATA_PATH}/mnist_expaned_k0{i}.pkl.gz', data)
    # np.save('mnist_test_dump.npy', train)
