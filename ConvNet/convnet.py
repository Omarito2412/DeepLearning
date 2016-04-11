import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
from PIL import Image

# Read mini batch
batch1 = cPickle.load(open("data_batch_1", "rb"))["data"]

# Reshape array to (10000, 32, 32, 3)
batch1 = batch1.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype('float32').reshape((-1, 3, 32, 32))
batch1 = batch1 / 256.

# img = Image.open(open('LP.JPG'))
# img = np.asarray(img, dtype='float32') / 256.
# batch1 = img.transpose(2, 0, 1).reshape(1, 3, 1536, 2048)

# Random image index
rand_index = np.random.randint(0, batch1.shape[0], 1)[0]

# Grab image shape
Image_Height = batch1[rand_index].shape[1]
Image_Width = batch1[rand_index].shape[2]
Image_channels = batch1[rand_index].shape[0]

plt.subplot(2, 1, 1)
plt.imshow(batch1[rand_index].reshape((Image_Height, Image_Width, Image_channels)))

# Define network layers
imgs = T.ftensor4('input image')

w = theano.shared(value=np.random.randn(1, 3, 3, 3)
                  .astype(imgs.dtype), name="Filter")

# Averaging filter for testing purposes
# w = theano.shared(value=(np.ones((1, 3, 3, 3)) / 9.)
#                   .astype(imgs.dtype), name="Filter")

output = T.nnet.conv.conv2d(imgs, w)

ConvLayer = theano.function([imgs], T.nnet.sigmoid(output))

convolved = ConvLayer(batch1)
plt.subplot(2, 1, 2)
plt.imshow(convolved[rand_index][0], cmap='binary')
plt.show()
