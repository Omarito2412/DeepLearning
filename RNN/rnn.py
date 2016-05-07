"""Trying a toy RNN."""
import numpy as np

import theano
import theano.tensor as T

# x = np.array(
#     [
#         [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
#         [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]
#     ])
# t = np.array(
#     [
#         [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
#         [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
#     ])

x = np.array(
    [
        [[1, 2, 6, -4, 3, 7, 1, 2, 9, 1, 4, 5, 5, 1, 1, 0, 1, 0, 1]]
    ])
t = np.array(
    [
        [[0, 1, 2, 6, -4, 3, 7, 1, 2, 9, 1, 4, 5, 5, 1, 1, 0, 1, 0]]
    ])

LR = 0.08
X = T.dmatrix('sequences')
Target = T.dmatrix('target')
h0 = T.dvector('Initial state')
wi = theano.shared(value=np.random.randn(1) * 0.1)
wj = theano.shared(value=np.random.randn(1) * 0.1)
wk = theano.shared(value=np.random.randn(1) * 0.1)
bi = theano.shared(value=np.zeros((1)))
bj = theano.shared(value=np.zeros((1)))


def next(xt, htm1, wi, wj, wk, bi, bj):
    """Perform a single forward pass in the RNN."""
    ht = T.tanh(T.dot(xt, wi) + T.dot(htm1, wk) + bi)
    yt = T.tanh(T.dot(ht, wj) + bj)
    return ht, yt

[h, y], _ = theano.scan(fn=next,
                        sequences=X,
                        outputs_info=[h0, None],
                        non_sequences=[wi, wj, wk, bi, bj]
                        )
cost = T.mean(T.sum((y - Target)**2))
grad_i = T.grad(cost, wi)
grad_j = T.grad(cost, wj)
grad_k = T.grad(cost, wk)
grad_bi = T.grad(cost, bi)
grad_bj = T.grad(cost, bj)
updates = [(wi, wi - LR * grad_i), (wj, wj - LR * grad_j),
           (wk, wk - LR * grad_k), (bi, bi - LR * grad_bi),
           (bj, bj - LR * grad_bj)]
run = theano.function([X, Target, h0], cost, updates=updates)
generate = theano.function([X, h0], y)

for i in range(50):
    for row_idx in range(0, x.shape[0]):
        print run(x[0].T, t[0].T, np.ones((1)))
print generate(x[0].T, np.ones((1)))
