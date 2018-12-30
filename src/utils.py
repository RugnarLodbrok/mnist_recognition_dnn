from contextlib import contextmanager
from time import time

import numpy as np


@contextmanager
def timing(msg):
    print(msg, '...', end='\r')
    t0 = time()
    yield
    print(msg, time() - t0)


def n_non_zeros(m):
    return sum(sum(np.abs(m) > 0.000000001)) / 29
