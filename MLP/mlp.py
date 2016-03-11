import theano
import theano.tensor as T
import numpy as np

HIDDEN_NEURONS = 100

# Declare symbols
x = T.dmatrix('Inputs')
y = T.dmatrix('Outputs')
t = T.dmatrix('Target Values')
Netj = T.dmatrix('Net of hidden layer')
Netk = T.dmatrix('Net of output layer')
Aj = T.dmatrix('Activation of hidden layer')
Wj = theano.shared(value=np.random.rand(785, HIDDEN_NEURONS),
                   name="Weights of hidden layer")
Wk = theano.shared(value=np.random.rand(HIDDEN_NEURONS, 10),
                   name="Weights of output layer")

Netj = T.dot(x, Wj)
Aj = T.nnet.sigmoid(Netj)

Netk = T.dot(Aj, Wk)
y = T.nnet.softmax(T.nnet.sigmoid(Netk))


ForwardProp = theano.function([x], y)

Training_set = np.loadtxt("/media/omarito/DATA/Data Sets/MNIST/train.csv",
                          dtype=np.uint8, skiprows=1, delimiter=",")
X_Train = Training_set[:, 1:]
X_Train = np.c_[np.ones((X_Train.shape[0], 1)), X_Train]
Y_Train = Training_set[:, 0]
