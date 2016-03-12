import theano
import theano.tensor as T
import numpy as np

# Declare constants
HIDDEN_NEURONS = 100
NUM_CLASSES = 10
NUM_FEATURES = 0

# Load Data
Training_set = np.loadtxt("/media/omarito/DATA/Data Sets/MNIST/train.csv",
                          dtype=np.uint8, skiprows=1, delimiter=",")
X_Train = Training_set[:, 1:]
X_Train = np.c_[np.ones((X_Train.shape[0], 1)), X_Train]
Y_Train = Training_set[:, 0]
Y_Train_onehot = np.zeros((Y_Train.shape[0], NUM_CLASSES))
# One hot encode Target Y
Y_Train_onehot[np.arange(X_Train.shape[0]), Y_Train] = 1

NUM_FEATURES = X_Train.shape[1]

# Declare symbols
x = T.dmatrix('Inputs')
y = T.dmatrix('Outputs')
t = T.dmatrix('Target Values')
Netj = T.dmatrix('Net of hidden layer')
Netk = T.dmatrix('Net of output layer')
Aj = T.dmatrix('Activation of hidden layer')
Wj = theano.shared(value=np.random.rand(NUM_FEATURES, HIDDEN_NEURONS),
                   name="Weights of hidden layer")
Wk = theano.shared(value=np.random.rand(HIDDEN_NEURONS, NUM_CLASSES),
                   name="Weights of output layer")

Netj = T.dot(x, Wj)
Aj = T.nnet.sigmoid(Netj)

Netk = T.dot(Aj, Wk)
y = T.nnet.softmax(Netk)

cost = T.mean(T.nnet.categorical_crossentropy(y, t))

computeCost = theano.function([y, t], cost)

forwardProp = theano.function([x], y)

print computeCost(forwardProp(X_Train), Y_Train_onehot)
