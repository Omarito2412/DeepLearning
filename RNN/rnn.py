"""Trying a toy RNN."""
import numpy as np

import theano
import theano.tensor as T
from matplotlib import pyplot as plt

LEARNINGRATE = 0.01
NUM_HIDDEN = 2
NUM_OUT = 1
NUM_IN = 1
EPOCHS = 500

x = np.array(
    [
        [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
        [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
    ])
t = np.array(
    [
        [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
        [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
    ])

# x = np.array(
#     [
#         [[1, 2, 4, 6]]
#     ])
# t = np.array(
#     [
#         [[0, 3, 6, 10]]
#     ])


X = T.dmatrix('sequences')
Target = T.dmatrix('target')
h0 = T.dvector('Initial state')
LR = T.dscalar('Learning rate')
wi = theano.shared(value=np.random.randn(NUM_IN, NUM_HIDDEN) * 0.01, name="wi")
wj = theano.shared(value=np.random.randn(NUM_HIDDEN, NUM_OUT) * 0.01, name="wj")
wk = theano.shared(value=np.random.randn(NUM_HIDDEN, NUM_HIDDEN) * 0.01, name="wk")
bi = theano.shared(value=np.zeros((NUM_HIDDEN)), name="bi")
bj = theano.shared(value=np.zeros((NUM_OUT)), name="bj")


def next(xt, htm1, wi, wj, wk, bi, bj):
    """Perform a single forward pass in the RNN."""
    ht = T.tanh(T.dot(xt, wi) + T.dot(htm1, wk) + bi)
    yt = T.nnet.sigmoid(T.dot(ht, wj) + bj)
    return ht, yt

[h, y], _ = theano.scan(fn=next,
                        sequences=X,
                        outputs_info=[h0, None],
                        non_sequences=[wi, wj, wk, bi, bj]
                        )
cost = T.sum((y - Target)**2)
grad_i = T.grad(cost, wi)
grad_j = T.grad(cost, wj)
grad_k = T.grad(cost, wk)
grad_bi = T.grad(cost, bi)
grad_bj = T.grad(cost, bj)
updates = [(wi, wi - LR * grad_i), (wj, wj - LR * grad_j),
           (wk, wk - LR * grad_k), (bi, bi - LR * grad_bi),
           (bj, bj - LR * grad_bj)]
run = theano.function([X, Target, h0, LR], cost, updates=updates)
generate = theano.function([X, h0], y)

costs = []
for i in range(EPOCHS):
    for row_idx in range(0, x.shape[0]):
        costs.append(run(x[row_idx].T, t[row_idx].T, np.ones((NUM_HIDDEN)), LEARNINGRATE))
    if(i % 25 == 0 and i > 0):
        LEARNINGRATE *= 0.98

plt.plot(range(0, EPOCHS * x.shape[0]), costs)
plt.show()
print generate(x[0].T, np.ones((NUM_HIDDEN)))
