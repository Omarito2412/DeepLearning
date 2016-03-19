import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle

# Declare constants
HIDDEN_NEURONS = 50
NUM_CLASSES = 10
NUM_FEATURES = 0
LEARNING_RATE = 0.4
XVAL_SIZE = 0.3
NUM_EPOCHS = 1000
NUM_RAND_INITS = 5

# Load Data
# Training_set = np.loadtxt("/media/omarito/DATA/Data Sets/MNIST/train.csv",
#                           dtype=np.uint8, skiprows=1, delimiter=",")

# Test_set = np.loadtxt("/media/omarito/DATA/Data Sets/MNIST/mnist_test.csv",
#                       dtype=np.uint8, skiprows=1, delimiter=",")

Training_set = cPickle.load(open("train.pickle", "rb"))

Test_set = cPickle.load(open("test.pickle", "rb"))

# Training Data
X_Train = Training_set[XVAL_SIZE * Training_set.shape[0]:, 1:]
X_Train = (X_Train - X_Train.mean()) / X_Train.std()
X_Train = np.c_[np.ones((X_Train.shape[0], 1)), X_Train]
Y_Train = Training_set[XVAL_SIZE * Training_set.shape[0]:, 0]
Y_Train_onehot = np.zeros((Y_Train.shape[0], NUM_CLASSES))

# Cross Validation data
X_XVal = Training_set[0:XVAL_SIZE * Training_set.shape[0], 1:]
X_XVal = np.c_[np.ones((X_XVal.shape[0], 1)), X_XVal]
Y_XVal = Training_set[0:XVAL_SIZE * Training_set.shape[0], 0]
Y_XVal_onehot = np.zeros((Y_XVal.shape[0], NUM_CLASSES))

# Test Data
X_Test = Test_set[:, 1:]
X_Test = (X_Test - X_Test.mean()) / X_Train.std()
X_Test = np.c_[np.ones((X_Test.shape[0], 1)), X_Test]
Y_Test = Test_set[:, 0]

# One hot encode Target Y
Y_Train_onehot[np.arange(X_Train.shape[0]), Y_Train] = 1
Y_XVal_onehot[np.arange(X_XVal.shape[0]), Y_XVal] = 1

NUM_FEATURES = X_Train.shape[1]

# Declare symbols
x = T.dmatrix('Inputs')
y = T.dmatrix('Outputs')
t = T.dmatrix('Target Values')
Netj = T.dmatrix('Net of hidden layer')
Netk = T.dmatrix('Net of output layer')
Aj = T.dmatrix('Activation of hidden layer')
Wj = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS) * 0.01
Wk = np.random.rand(HIDDEN_NEURONS, NUM_CLASSES) * 0.01
Weights = theano.shared(value=np.concatenate((Wj.ravel(), Wk.ravel()), axis=0),
                        name="Weights ravelled")

# Define equations
Netj = T.dot(x, Weights[0:NUM_FEATURES * HIDDEN_NEURONS]
             .reshape((NUM_FEATURES, HIDDEN_NEURONS)))
Aj = T.maximum(Netj, 0.01 * Netj)
# Aj = T.tanh(Netj)
# Aj = T.nnet.sigmoid(Netj)
# Aj = Netj

Netk = T.dot(Aj, Weights[NUM_FEATURES * HIDDEN_NEURONS:]
             .reshape((HIDDEN_NEURONS, NUM_CLASSES)))
y = T.nnet.softmax(Netk)

cost = T.mean(T.nnet.categorical_crossentropy(y, t))

Grads = T.grad(cost, Weights)

# Define Functions

computeCost = theano.function([y, t], cost)

forwardProp = theano.function([x], y)

updates = [(Weights, Weights - LEARNING_RATE * Grads)]
trainModel = theano.function([x, t], cost, updates=updates)

costs = {'training': list(), 'xval': list()}
for i in range(NUM_EPOCHS):
    print "Epoch number: " + str(i + 1)
    costs['training'].append(trainModel(X_Train, Y_Train_onehot))
    costs['xval'].append(computeCost(forwardProp(X_XVal), Y_XVal_onehot))
    if(i % 25 == 0 and i > 0):
        Test_Result = np.argmax(forwardProp(X_Test), axis=1)
        Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
            (Y_Test.shape[0])) * 100
        print "The model performed with an accuracy of: %.2f" % (float(Score)) + "%"
        plt.plot(range(i + 1), costs['training'], range(
            i + 1), costs['xval'])
        plt.show()

# plt.plot(range(NUM_EPOCHS), costs['training'], range(
#             NUM_EPOCHS), costs['xval'])
#         plt.show()
Test_Result = np.argmax(forwardProp(X_Test), axis=1)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model performed with an accuracy of: %.2f" % (float(Score)) + "%"
