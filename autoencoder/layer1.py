import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle

# Declare constants
HIDDEN_NEURONS = 50
NUM_CLASSES = 0
NUM_FEATURES = 0
LEARNING_RATE = 0.5
XVAL_SIZE = 0.3
NUM_EPOCHS = 1000
NUM_RAND_INITS = 5
L1_reg = 2.5 * 10 ** -4
L2_reg = 2.5 * 10 ** -4
DROPOUT_PROBABILITY = 0.5
REGULZN = "Dropout"
ACTIVATION = "tanh"

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
NUM_CLASSES = X_Train.shape[1]

# Declare symbols
x = T.dmatrix('Inputs')
y = T.dmatrix('Outputs')
Netj = T.dmatrix('Net of hidden layer')
Netk = T.dmatrix('Net of output layer')
Aj = T.dmatrix('Activation of hidden layer')
Wj = np.random.rand(NUM_FEATURES, HIDDEN_NEURONS) * 0.01
Wk = np.random.rand(HIDDEN_NEURONS, NUM_CLASSES) * 0.01
Weights = theano.shared(value=np.concatenate((Wj.ravel(), Wk.ravel()), axis=0),
                        name="Weights ravelled")

Netj = T.dot(x, Weights[0:NUM_FEATURES * HIDDEN_NEURONS]
             .reshape((NUM_FEATURES, HIDDEN_NEURONS)))
Aj = T.nnet.sigmoid(Netj)

Netk = T.dot(Aj, Weights[NUM_FEATURES * HIDDEN_NEURONS:]
             .reshape((HIDDEN_NEURONS, NUM_CLASSES)))
y = T.nnet.softmax(Netk)

cost = T.mean(T.nnet.categorical_crossentropy(y, x))
Grads = T.grad(cost, Weights)
forwardProp = theano.function([x], y)


updates = [(Weights, Weights - LEARNING_RATE * (Grads))]
trainModel = theano.function([x], cost, updates=updates)

for i in range(50):
    trainModel(X_Train)

Test_Result = np.argmax(forwardProp(X_Test), axis=1)
Score = float(len(np.where(Test_Result == X_Test)[0])) / float(
    (X_Test.shape[0])) * 100
