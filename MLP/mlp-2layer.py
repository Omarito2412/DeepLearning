"""A Two layer Neural Network tested on MNIST."""
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle

# Declare constants
HIDDEN_NEURONS = 30
HIDDEN_NEURONS_2 = 20
NUM_CLASSES = 10
NUM_FEATURES = 0
LEARNING_RATE = 0.5
XVAL_SIZE = 0.3
NUM_EPOCHS = 1000
NUM_RAND_INITS = 5
ACTIVATION = "ReLU"

theano.config.exception_verbosity = "high"
# Initialize random stream
rng = np.random.RandomState(1234)

# Load Pickle files
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
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
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
Netk = T.dmatrix('Net of second hidden layer')
Netl = T.dmatrix('Net of output hidden')
Aj = T.dmatrix('Activation of hidden layer')
Ak = T.dmatrix('Activation of hidden layer 2')
Wj = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS) * 0.01
Wk = np.random.rand(HIDDEN_NEURONS, HIDDEN_NEURONS_2) * 0.01
Wl = np.random.rand(HIDDEN_NEURONS_2, NUM_CLASSES) * 0.01
Weights = theano.shared(
    value=np.concatenate((Wj.ravel(), Wk.ravel(), Wl.ravel()), axis=0),
    name="Weights ravelled")

# Define equations
Netj = T.dot(x, Weights[0:NUM_FEATURES * HIDDEN_NEURONS]
             .reshape((NUM_FEATURES, HIDDEN_NEURONS)))

# Activation
if(ACTIVATION is "ReLU"):
    Aj = T.maximum(Netj, 0.01 * Netj)
elif(ACTIVATION is "tanh"):
    Aj = T.tanh(Netj)
else:
    Aj = T.nnet.sigmoid(Netj)

Netk = T.dot(Aj,
             Weights[NUM_FEATURES * HIDDEN_NEURONS:(NUM_FEATURES *
                     HIDDEN_NEURONS + HIDDEN_NEURONS * HIDDEN_NEURONS_2)]
             .reshape((HIDDEN_NEURONS, HIDDEN_NEURONS_2)))

# Activation 2
if(ACTIVATION is "ReLU"):
    Ak = T.maximum(Netk, 0.01 * Netk)
elif(ACTIVATION is "tanh"):
    Ak = T.tanh(Netk)
else:
    Ak = T.nnet.sigmoid(Netk)

Netl = T.dot(Ak, Weights[
    (NUM_FEATURES * HIDDEN_NEURONS + HIDDEN_NEURONS * HIDDEN_NEURONS_2):]
    .reshape((HIDDEN_NEURONS_2, NUM_CLASSES)))
y = T.nnet.softmax(Netl)
cost = T.mean(T.nnet.categorical_crossentropy(y, t))
Grads = T.grad(cost, Weights)

# Define Functions

computeCost = theano.function([y, t], cost)

forwardProp = theano.function([x], y)

updates = [(Weights, Weights - LEARNING_RATE * (Grads))]
trainModel = theano.function([x, t], cost, updates=updates)

# Best of N random initializations
minimum = 9999
minimum_weights = 0
for i in range(NUM_RAND_INITS):
    tmp = computeCost(forwardProp(X_Train), Y_Train_onehot)
    if (tmp < minimum):
        minimum = tmp
        minimum_weights = Weights.get_value()
    Wj = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS) * 0.01
    Wk = np.random.rand(HIDDEN_NEURONS, HIDDEN_NEURONS_2) * 0.01
    Wl = np.random.rand(HIDDEN_NEURONS_2, NUM_CLASSES) * 0.01
    Weights.set_value(
        np.concatenate((Wj.ravel(), Wk.ravel(), Wl.ravel()), axis=0))

Weights.set_value(minimum_weights)
costs = {'training': list(), 'xval': list()}
for i in range(NUM_EPOCHS):
    print "Epoch number: " + str(i + 1)
    costs['training'].append(trainModel(X_Train, Y_Train_onehot))
    costs['xval'].append(computeCost(forwardProp(X_XVal), Y_XVal_onehot))
    if(i % 10 == 0 and i > 0):
        Test_Result = np.argmax(forwardProp(X_Test), axis=1)
        Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
            (Y_Test.shape[0])) * 100
        print "The model classified with an accuracy of: %.2f" % (
            float(Score)) + "%"
        # plt.plot(range(i + 1), costs['training'], range(
        #     i + 1), costs['xval'])
        # plt.show()

# plt.plot(range(NUM_EPOCHS), costs['training'], range(
#             NUM_EPOCHS), costs['xval'])
#         plt.show()
Test_Result = np.argmax(forwardProp(X_Test), axis=1)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model classified with an accuracy of: %.2f" % (float(Score)) + "%"
