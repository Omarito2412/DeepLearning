import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
import argparse

# Initialize Argument Parser
parser = argparse.ArgumentParser(
    description='Encode the given input')

parser.add_argument('sets',
                    help='The Training set and the Test set to encode', nargs='+')
sets = parser.parse_args().sets


parser.add_argument('-b', '--bias', dest='bias', default=True,
                    help='Add bias column?')
bias = parser.parse_args().bias

# Declare constants
HIDDEN_NEURONS = 300
LEARNING_RATE = 10 ** -6
NUM_EPOCHS = 1000
ACTIVATION = "sigmoid"

Training_set = cPickle.load(open(sets[0], "rb"))

Test_set = cPickle.load(open(sets[1], "rb"))

# Training Data
X_Train = Training_set[:, 1:]
X_Train = (X_Train - X_Train.mean()) / X_Train.std()
if(bias):
    X_Train = np.c_[np.ones((X_Train.shape[0], 1)), X_Train]

# Test Data
X_Test = Test_set[:, 1:]
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
if(bias):
    X_Test = np.c_[np.ones((X_Test.shape[0], 1)), X_Test]

# Count features
NUM_FEATURES = X_Train.shape[1]

# Declare symbols
x = T.dmatrix('Inputs')
y = T.dmatrix('Outputs')
Netj = T.dmatrix('Net of hidden layer')
Netk = T.dmatrix('Net of output layer')
Aj = T.dmatrix('Activation of hidden layer')
Wj = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS) * 10 ** -2
Wk = np.random.rand(HIDDEN_NEURONS, NUM_FEATURES) * 10 ** -2
Weights = theano.shared(value=np.concatenate((Wj.ravel(), Wk.ravel()), axis=0),
                        name="Weights ravelled")

Netj = T.dot(x, Weights[0:NUM_FEATURES * HIDDEN_NEURONS]
             .reshape((NUM_FEATURES, HIDDEN_NEURONS)))
Aj = T.nnet.sigmoid(Netj)

Netk = T.dot(Aj, Weights[NUM_FEATURES * HIDDEN_NEURONS:]
             .reshape((HIDDEN_NEURONS, NUM_FEATURES)))
y = Netk

cost = T.mean(T.sum((y - x) ** 2))

Grads = T.grad(cost, Weights)
forwardProp = theano.function([x], y)


updates = [(Weights, Weights - LEARNING_RATE * (Grads))]
trainModel = theano.function([x], [cost, Grads], updates=updates)

costs = list()
for i in range(NUM_EPOCHS):
    train = trainModel(X_Train)
    costs.append(train[0])
    print "The error at epoch %i is: " % i, train[0]
    if(i % 10 == 0 and i > 0):
        Test_Result = forwardProp(X_Test)
        euc_distance = ((Test_Result - X_Test)**2).mean()
        print "The two examples: ", Test_Result[1, 0:6], X_Test[1, 0:6]
        plt.plot(range(i + 1), costs)
        # plt.show()
# Save the encoded input
cPickle.dump(forwardProp(X_Train), open("training_encoded.pickle", "wb"),
             cPickle.HIGHEST_PROTOCOL)
cPickle.dump(forwardProp(X_Test), open("test_encoded.pickle", "wb"),
             cPickle.HIGHEST_PROTOCOL)
