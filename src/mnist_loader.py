"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
from os.path import join as j
import pickle
import gzip

# Third-party libraries
import numpy as np

DATA_PATH = '../data'


def _load_mnist(path='mnist.pkl.gz'):
    with gzip.open(j(DATA_PATH, path), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def dump_data(f_name, data):
    with gzip.open(j(DATA_PATH, f_name), 'wb') as f:
        pickle.dump(data, f)


def load_data(f_name):
    with gzip.open(j(DATA_PATH, f_name), 'rb') as f:
        return pickle.load(f)


def load_mnist_simple(shape=(784, 1)):
    return [list(zip([np.reshape(x, shape) for x in xx], yy))
            for xx, yy in _load_mnist()]


def load_mnist_theano():
    import theano.tensor as T
    import theano
    def f(data):
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return map(f, _load_mnist())
