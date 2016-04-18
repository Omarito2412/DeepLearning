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

# Read mini batch
data = cPickle.load(open("../../../data_batch_1", "rb"))
batch1 = data["data"]
t = np.array(data["labels"])

t_onehot = np.zeros((t.shape[0], 10))
t_onehot[np.arange(t.shape[0]), t] = 1

# Reshape array to (10000, 32, 32, 3)
# batch1 = batch1 / 256.
images = batch1.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32').reshape((-1, 32, 32, 3)) / 256.
batch1 = batch1.reshape((-1, 3, 32, 32)).astype("float32")
batch1 = (batch1 - batch1.mean()) / batch1.std()
# img = Image.open(open('LP.JPG'))
# img = np.asarray(img, dtype='float32') / 256.
# batch1 = img.transpose(2, 0, 1).reshape(1, 3, 1536, 2048)
X_Test = batch1[0:1000, :, :, :]
Y_Test = t[0:1000]
Y_Test_onehot = t_onehot[0:1000, :]

batch1 = batch1[1000:, :, :, :]
t_onehot = t_onehot[1000:, :]

# Random image index
rand_index = np.random.randint(0, batch1.shape[0], 1)[0]

# Grab image shape
Image_Height = batch1[rand_index].shape[1]
Image_Width = batch1[rand_index].shape[2]
Image_channels = batch1[rand_index].shape[0]

plt.subplot(2, 1, 1)
plt.imshow(images[rand_index])

# Define network layers
imgs = T.ftensor4('input image')

w = theano.shared(value=np.random.randn(5, 3, 3, 3)
                  .astype(imgs.dtype), name="Filter")
target = T.dmatrix('target values')
# Averaging filter for testing purposes
# w = theano.shared(value=(np.ones((1, 3, 3, 3)) / 9.)
#                   .astype(imgs.dtype), name="Filter")

conv_output = T.nnet.conv.conv2d(imgs, w)
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
updates = [(w, w - W_grad), (Wi, Wi - LEARNING_RATE * Wi_grad), (Wj, Wj - LEARNING_RATE * Wj_grad)]
Conv_net = theano.function([imgs, target], [cost, y], updates=updates)

for i in range(EPOCHES):
    epoch = Conv_net(batch1, t_onehot)
    print "Cost at Epoch %i is: %1.5f" % (i, epoch[0])

Test_Result = np.argmax(Conv_net(X_Test, Y_Test_onehot)[1], axis=1)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model performed with an accuracy of: %.2f" % (float(Score)) + "%"
