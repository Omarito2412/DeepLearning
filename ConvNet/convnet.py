"""A hard coded sized ConvNet."""
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
from PIL import Image


LEARNING_RATE = 0.7
EPOCHES = 200
IMAGE_DIMS = (32, 32, 3)

# Read CIFAR-10
cifar = cPickle.load(open("../../../cifar.pickle"))
test = cPickle.load(open("../../../cifar_test", "rb"))

X_Train = cifar["data"].reshape(
    (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))
X_Train = (X_Train - X_Train.mean()) / X_Train.std()
Y_Train = cifar["labels"]
Y_Train_onehot = np.zeros((Y_Train.shape[0], 10))
Y_Train_onehot[np.arange(Y_Train.shape[0]), Y_Train.astype(int)] = 1

X_Test = test["data"].reshape(
    (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
Y_Test = test["labels"]

# Random image index
rand_index = np.random.randint(0, X_Train.shape[0], 1)[0]

# Grab image shape
Image_Height = X_Train[rand_index].shape[1]
Image_Width = X_Train[rand_index].shape[2]
Image_channels = X_Train[rand_index].shape[0]

# plt.subplot(2, 1, 1)
# plt.imshow(images[rand_index])

# Define network layers
imgs = T.ftensor4('input image')

w = theano.shared(value=np.random.randn(5, 3, 3, 3)
                  .astype(imgs.dtype), name="Filter")
target = T.dmatrix('target values')
# Averaging filter for testing purposes
# w = theano.shared(value=(np.ones((1, 3, 3, 3)) / 9.)
#                   .astype(imgs.dtype), name="Filter")

conv_output = T.nnet.conv2d(imgs, w)
ConvLayer = theano.function([imgs], conv_output)

ReLU_output = T.maximum(0.001 * conv_output, conv_output)
ReLU_layer = theano.function([imgs], ReLU_output)

pooling_shape = (2, 2)
pool_output = downsample.max_pool_2d(ReLU_output, pooling_shape)
pool_layer = theano.function([imgs], pool_output)


Wi = theano.shared(np.random.randn(1125, 100) * 0.01)
Wj = theano.shared(np.random.randn(100, 10) * 0.01)

Neti = T.dot(pool_output.reshape((-1, 1125)), Wi)
Ai = T.nnet.sigmoid(Neti)
Netj = T.dot(Ai, Wj)
y = T.nnet.softmax(Netj)
cost = T.mean(T.nnet.categorical_crossentropy(y, target))
FC = theano.function([imgs, target], [cost])

W_grad = T.grad(cost, w)
Wi_grad = T.grad(cost, Wi)
Wj_grad = T.grad(cost, Wj)
updates = [(w, w - W_grad), (Wi, Wi - LEARNING_RATE * Wi_grad),
           (Wj, Wj - LEARNING_RATE * Wj_grad)]
Conv_net = theano.function([imgs, target], [cost, y], updates=updates)

Grads = theano.function([], [W_grad, Wi_grad, Wj_grad])

for i in range(EPOCHES):
    epoch = Conv_net(X_Train, Y_Train_onehot)
    print "Cost at Epoch %i is: %1.5f" % (i, epoch[0])
    grads = Grads()
    print grads[0], grads[1], grads[2]

# Test_Result = np.argmax(Conv_net(X_Test, Y_Test)[1], axis=1)
# Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    # (Y_Test.shape[0])) * 100
# print "The model performed with an accuracy of: %.2f" % (float(Score)) + "%"
