"""In this module I will try to create a CNN using modular code.

this allows us to easily attach and detach layers
and see how it behaves.
"""
import theano
import theano.tensor as T
import theano.tensor.signal.pool as pool
import numpy as np
from six.moves import cPickle

# Define our model's hyper parameters
NUM_HIDDEN_NEURONS = 500
IMAGE_DIMS = (28, 28, 1)
FILTER_DIMS = (1, 5, 5)
NUM_CONV_FILTERS = 10
POOL_SHAPE = (2, 2)
STRIDE = 1
LEARNING_RATE = 0.1
EPOCHS = 3000
BATCH_SIZE = 500
NUM_CLASSES = 10

# Read a batch from CIFAR-10
# data = cPickle.load(open("../../../data_batch_1", "rb"))
# batch = data["data"].astype("float32")
# batch = (batch - batch.mean()) / batch.std()
# target = np.array(data["labels"]).astype("float32")


# Read the MNIST data
batch = cPickle.load(open("../MLP/train.pickle", "rb")).astype("float32")
test = cPickle.load(open("../MLP/test.pickle", "rb")).astype("float32")
target = batch[:, 0]
batch = batch[:, 1:]
batch = (batch - batch.mean()) / batch.std()

X_Test = test[:, 1:]
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
Y_Test = test[:, 0]
X_Test = X_Test.reshape(
    (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))

X_Train = batch.reshape(
    (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))
Y_Train = target

Y_Train_onehot = np.zeros((Y_Train.shape[0], 10))
Y_Train_onehot[np.arange(Y_Train.shape[0]), Y_Train.astype(int)] = 1


class ConvLayer:
    """Defines a simple convolution layer."""

    def __init__(self, images, filters_count, filter_dims, stride):
        """Initialize the layer."""
        self.filters = theano.shared(
            np.random.randn
            (
                filters_count, filter_dims[0],
                filter_dims[1], filter_dims[2]
            ).astype(images.dtype) * np.sqrt(2.0 / (
                filters_count * np.prod(filter_dims))),
            "Filters"
        )

        self.output = T.nnet.conv2d(
            input=images,
            filters=self.filters,
            subsample=(stride, stride),
            border_mode="half"
        )

        self.params = self.filters


class ReLULayer:
    """Defines a simple ReLU layer."""

    def __init__(self, images, filters_count):
        """Initialize the layer."""
        self.b = theano.shared(
            value=np.zeros(
                (
                    filters_count,
                )
            ).astype(images.dtype))
        self.output = T.nnet.relu(images + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = self.b


class PoolLayer:
    """Defines a simple max pooling layer."""

    def __init__(self, images, pool_shape):
        """Initialize the layer."""
        self.output = pool.pool_2d(
            images,
            pool_shape,
            ignore_border=False
        )


class FullyConnectedNetwork:
    """Defines a fully connected network."""

    def __init__(self, images, image_size, hidden_neurons, classes):
        """Initialize the layer."""
        self.Wi = theano.shared(
            value=np.random.randn
            (
                image_size,
                hidden_neurons
            ).astype(images.dtype) * np.sqrt(2.0 / (
                image_size + hidden_neurons))
        )
        self.Wj = theano.shared(
            value=np.random.randn
            (
                hidden_neurons,
                classes
            ).astype(images.dtype) * np.sqrt(2.0 / (
                hidden_neurons + 10))
        )

        self.bi = theano.shared(
            value=np.zeros(
                (
                    hidden_neurons,
                )
            ).astype(images.dtype)
        )
        self.bj = theano.shared(
            value=np.zeros(
                (
                    classes,
                )
            ).astype(images.dtype)
        )
        neti = T.dot(images, self.Wi)
        ai = T.nnet.relu(neti + self.bi)

        netj = T.dot(ai, self.Wj)
        self.output = T.nnet.softmax(netj + self.bj)

        self.params = [self.Wi, self.Wj, self.bi, self.bj]

# Our symbols
imgs = T.ftensor4('Input batch')
target = T.fmatrix('Target values')

# Stack the network layers
# First conv-relu-pool
conv1 = ConvLayer(imgs, NUM_CONV_FILTERS, FILTER_DIMS, STRIDE)
relu1 = ReLULayer(conv1.output, NUM_CONV_FILTERS)
pool1 = PoolLayer(relu1.output, POOL_SHAPE)

# Updating parameters for the next layers
FILTER2_DIMS = (NUM_CONV_FILTERS, FILTER_DIMS[1], FILTER_DIMS[2])
NUM_CONV_FILTERS = NUM_CONV_FILTERS * 2

# Second con-relu-pool
conv2 = ConvLayer(pool1.output, NUM_CONV_FILTERS, FILTER2_DIMS, STRIDE)
relu2 = ReLULayer(conv2.output, NUM_CONV_FILTERS)
pool2 = PoolLayer(relu2.output, POOL_SHAPE)


# Calculate the rolled image size
pooled_image_size = int(IMAGE_DIMS[0] / (2 * POOL_SHAPE[0]))

full_image_size = (pooled_image_size ** 2) * NUM_CONV_FILTERS

pool2_output_reshaped = pool2.output.reshape((-1, full_image_size))

# Stack the FullyConnected network
FC = FullyConnectedNetwork(
    pool2_output_reshaped, full_image_size, NUM_HIDDEN_NEURONS, NUM_CLASSES)

# Calculate the cost
cost = T.mean(T.nnet.categorical_crossentropy(FC.output, target))

updates = []

# Calculate gradients of all layer params
network_layers = [conv1, FC, relu1]

# Loop through all the network's
# parameters and calculate their
# gradients then create the
# corresponding update
# rules

for layer in network_layers:
    if(type(layer.params) is list):
        for item in layer.params:
            updates.append((item,
                            item - LEARNING_RATE * T.grad(cost, item)))
    else:
        updates.append((layer.params,
                        layer.params - LEARNING_RATE * T.grad(
                            cost, layer.params)))

# Define functions to run our model
Train = theano.function(
    [imgs, target],
    cost,
    updates=updates,
    allow_input_downcast=True
)
Test = theano.function([imgs], FC.output)


# Loop through the epochs count
for i in range(EPOCHS):
    # Get a random pivot point and
    # extract a mini batch
    rand_index = np.random.randint(0, X_Train.shape[0] - BATCH_SIZE)
    # Calculate the cost and perform an update
    Cost = Train(X_Train[rand_index:rand_index + BATCH_SIZE, :],
                 Y_Train_onehot[rand_index:rand_index + BATCH_SIZE, :])
    # Print epoch stats
    print "Cost at epoch %i is equal to: %1.3f" % (i, Cost)

    # Every 50 epoch calculate accuracy
    if(i % 50 == 0 and i > 0):
        Test_Result = np.argmax(Test(X_Test.astype("float32")), axis=1)
        Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
            (Y_Test.shape[0])) * 100
        print "The model performed with an accuracy of: %.2f" % (
            float(Score)) + "%"

    # Decay the learning rate
    LEARNING_RATE = LEARNING_RATE - 0.1 * LEARNING_RATE


# Calculate accuracy one last time
Test_Result = np.argmax(Test(X_Test), axis=1)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model performed with an accuracy of: %.2f" % (
    float(Score)) + "%"
