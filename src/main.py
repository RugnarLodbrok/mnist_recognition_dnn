import numpy as np

from dnn import DNN
from layers import Layer, DropoutLayer
from mnist_loader import load_mnist_simple
from contextlib import contextmanager
from time import time
from cost_activations import LogisticCrossEntropy as LCE, LogisticQuadratic as LQ, SoftMax as SM


@contextmanager
def timing(msg):
    t0 = time()
    yield
    print(msg, time() - t0)


"""
ideas
adjust eta:
    - accuracy stops to improve
    - assume that we run in cycle in space of all micro-parameters
    - reduce `eta'
    - learn
    - either get better results then with bigger `eta' or go back and and move outside of area in step 2 
        (add random vals to all micro-params?)
dnn curiosity:
    - shft in random direction (on each step or each epoch?)
    - shift more as accuracy improves less
    
todo: check how weights change form initial random init to final state, how much this depends on depth of a layer
which weights change earlier, and which later
check how total weight-vector length changes during trainging - this is needed to get correct scale for curiosity shift

TODO: check kurtosis of weights distro


"""


def main():
    train, test, vadilation = load_mnist_simple()
    # x, y = train[0]
    # print("x: ", x.shape)
    # print("y: ", y)

    with timing(f""):
        # dnn = DNN(input=28 * 28, layers=[Layer(30, LQ), Layer(10, LCE)], eta=0.05)  # 96%
        # dnn = DNN(input=28 * 28, layers=[Layer(30, LQ), Layer(10, SM)], eta=0.001)  # 68%
        # dnn = DNN(input=28 * 28, layers=[Layer(100, LQ), Layer(10, LCE)], eta=0.05, lmbda=5)  # 98%
        # dnn = DNN(input=28 * 28, layers=[DropoutLayer(100, LQ), Layer(10, LCE)], eta=0.05)  # 97.5%
        dnn = DNN(input=28 * 28, layers=[DropoutLayer(160, LQ), Layer(10, LCE)], eta=0.05, lmbda=3)
        dnn.initialize_rand()
        dnn.learn(train, epochs=30, test=vadilation, batch_size=29)

    print('test:', dnn.test(test))
    print(dnn.stats())


if __name__ == '__main__':
    main()
