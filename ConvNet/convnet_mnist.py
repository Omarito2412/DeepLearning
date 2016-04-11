import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt

Training_set = cPickle.load(open("../MLP/train.pickle", "rb"))[:, 1:].reshape((-1, 1, 28, 28))
Training_set = Training_set.astype("float32") / 256.

# Define network layers
imgs = T.ftensor4('input image')

# Averaging filter for testing purposes
# x = np.array([[np.ones((3, 3)) / 9.]])
x = np.random.randn(3, 3).reshape((1, 1, 3, 3))
w = theano.shared(value=x
                  .astype(imgs.dtype), name="Filter")

output = T.nnet.conv.conv2d(imgs, w)

ConvLayer = theano.function([imgs], T.nnet.sigmoid(output))
convolved = ConvLayer(Training_set)

random_index = np.random.randint(0, Training_set.shape[0], 1)[0]

plt.subplot(2, 1, 1)
plt.imshow(Training_set[random_index][0], cmap='binary')

plt.subplot(2, 1, 2)
plt.imshow(convolved[random_index][0], cmap='binary')
plt.show()
