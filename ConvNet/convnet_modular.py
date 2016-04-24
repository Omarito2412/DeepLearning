"""In this module I will try to create a CNN using modular code.

this allows us to easily attach and detach layers
and see how it behaves.
"""
import numpy as np

from six.moves import cPickle

import theano
import theano.tensor as T
import theano.tensor.signal.pool as pool

# Define our model's hyper parameters
NUM_HIDDEN_NEURONS = 500
IMAGE_DIMS = (32, 32, 3)
FILTER_DIMS = (3, 5, 5)
NUM_CONV_FILTERS = 20
POOL_SHAPE = (2, 2)
STRIDE = 1
LEARNING_RATE = 0.1
EPOCHS = 30000
BATCH_SIZE = 100
NUM_CLASSES = 10
MOMENTUM = 0.9

# Read CIFAR-10
cifar = cPickle.load(open("../../../cifar.pickle", "rb"))
test = cPickle.load(open("../../../cifar_test.pickle", "rb"))

X_Train = cifar["data"].reshape(
    (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))
X_Train = (X_Train - X_Train.mean()) / X_Train.std()
Y_Train = cifar["labels"]
Y_Train_onehot = np.zeros((Y_Train.shape[0], 10))
Y_Train_onehot[np.arange(Y_Train.shape[0]), Y_Train.astype(int)] = 1

X_Test = test["data"][0:4000].reshape(
    (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))
X_Test = (X_Test - X_Test.mean()) / X_Test.std()
Y_Test = np.array(test["labels"])[0:4000]

# Read the MNIST data
# batch = cPickle.load(open("../MLP/train.pickle", "rb")).astype("float32")
# test = cPickle.load(open("../MLP/test.pickle", "rb")).astype("float32")
# target = batch[:, 0]
# batch = batch[:, 1:]
# batch = (batch - batch.mean()) / batch.std()

# X_Test = test[:, 1:]
# X_Test = (X_Test - X_Test.mean()) / X_Test.std()
# Y_Test = test[:, 0]
# X_Test = X_Test.reshape(
#     (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))

# X_Train = batch.reshape(
#     (-1, IMAGE_DIMS[2], IMAGE_DIMS[0], IMAGE_DIMS[1]))
# Y_Train = target

# Y_Train_onehot = np.zeros((Y_Train.shape[0], 10))
# Y_Train_onehot[np.arange(Y_Train.shape[0]), Y_Train.astype(int)] = 1


class ConvLayer:
    """Defines a simple convolution layer."""

    def __init__(self, images, filters_count, filter_dims, stride):
        """Initialize the layer."""
        self.name = "Conv"
        self.filters = theano.shared(
            np.random.randn
            (
                filters_count, filter_dims[0],
                filter_dims[1], filter_dims[2]
            ).astype(images.dtype) * np.sqrt(2.0 / (
                filters_count * np.prod(filter_dims))),
            "Filters"
        )

        self.Vfilters = theano.shared(
            value=np.zeros(
                (
                    filters_count, filter_dims[0],
                    filter_dims[1], filter_dims[2]
                )).astype(images.dtype))

        self.output = T.nnet.conv2d(
            input=images,
            filters=self.filters,
            subsample=(stride, stride),
            border_mode="half"
        )

        self.params = [self.filters]
        self.velocity = [self.Vfilters]


class ReLULayer:
    """Defines a simple ReLU layer."""

    def __init__(self, images, filters_count):
        """Initialize the layer."""
        self.name = "ReLU"
        self.b = theano.shared(
            value=np.zeros(
                (
                    filters_count,
                )
            ).astype(images.dtype))
        self.Vb = theano.shared(
            value=np.zeros(
                (
                    filters_count,
                )
            ).astype(images.dtype))
        self.output = T.nnet.relu(images + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.b]
        self.velocity = [self.Vb]


class PoolLayer:
    """Defines a simple max pooling layer."""

    def __init__(self, images, pool_shape):
        """Initialize the layer."""
        self.name = "Pool"
        self.output = pool.pool_2d(
            images,
            pool_shape,
            ignore_border=False
        )
        self.params = []


class FullyConnectedNetwork:
    """Defines a fully connected network."""

    def __init__(self, images, image_size, hidden_neurons, classes):
        """Initialize the layer."""
        self.name = "FC"
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
        # Define velocity vectors
        self.VWi = theano.shared(
            value=np.zeros(
                (
                    image_size, hidden_neurons
                )
            ).astype(images.dtype))
        self.VWj = theano.shared(
            value=np.zeros(
                (
                    hidden_neurons, classes
                )
            ).astype(images.dtype))
        self.VBi = theano.shared(
            value=np.zeros(
                (
                    hidden_neurons,
                )
            ).astype(images.dtype))
        self.VBj = theano.shared(
            value=np.zeros(
                (
                    classes,
                )
            ).astype(images.dtype))
        self.velocity = [self.VWi, self.VWj, self.VBi, self.VBj]


def update_velocity(v, dx, l_rate):
    """Calculate V_t+1."""
    result = (v * MOMENTUM) - (l_rate * dx)
    return result


def count_pools(network):
    """Count pool layers in a given network stack."""
    counter = 0
    for layer in network:
        if(layer.name == "Pool"):
            counter += 1
    return counter

# Our symbols
imgs = T.ftensor4('Input batch')
target = T.fmatrix('Target values')
LR = T.scalar('Learning Rate')


# This stack will hold the network layer
# I put them in this stack so I can
# easily loop over them and
# perform updates while
# maintaining the ease
# of adding layers.
network_layers = []

# Stack the network layers

# First conv-relu
network_layers.append(ConvLayer(imgs, NUM_CONV_FILTERS, FILTER_DIMS, STRIDE))
network_layers.append(ReLULayer(network_layers[-1].output, NUM_CONV_FILTERS))

# Updating parameters for the next layers
# The filters need to change from
# (Channels, Height, Width) to
# (Filters count from last layer, Height, Width)
FILTER2_DIMS = (NUM_CONV_FILTERS, FILTER_DIMS[1], FILTER_DIMS[2])
NUM_CONV_FILTERS = 50

# Next conv-relu-pool
network_layers.append(ConvLayer(
    network_layers[-1].output, NUM_CONV_FILTERS, FILTER2_DIMS, STRIDE)
)
network_layers.append(ReLULayer(network_layers[-1].output, NUM_CONV_FILTERS))
network_layers.append(PoolLayer(network_layers[-1].output, POOL_SHAPE))

# Updating parameters for the next layers
NUM_CONV_FILTERS = 50
FILTER2_DIMS = (NUM_CONV_FILTERS, FILTER_DIMS[1], FILTER_DIMS[2])

# Another conv-relu-pool
network_layers.append(ConvLayer(
    network_layers[-1].output, NUM_CONV_FILTERS, FILTER2_DIMS, STRIDE)
)
network_layers.append(ReLULayer(network_layers[-1].output, NUM_CONV_FILTERS))
network_layers.append(PoolLayer(network_layers[-1].output, POOL_SHAPE))

# Calculate the rolled image size
pooled_image_size = int(IMAGE_DIMS[0] / (count_pools(
    network_layers) * POOL_SHAPE[0]))

# Rolled image size = DIM0 * DIM1 * Channels
full_image_size = (pooled_image_size ** 2) * NUM_CONV_FILTERS

output_flat = network_layers[-1].output.flatten(2)

# Stack the FullyConnected network
network_layers.append(FullyConnectedNetwork(
    output_flat, full_image_size, NUM_HIDDEN_NEURONS, NUM_CLASSES))

# Calculate the cost
cost = T.mean(
    T.nnet.categorical_crossentropy(
        network_layers[-1].output, target
    )
)

updates = []

# Calculate gradients of all layer params
# Loop through all the network's
# parameters and calculate their
# gradients then create the
# corresponding update
# rules

# Nesterov momentum as introduced by Ilya Sustskever
# with a few modifications by Ian Goodfellow
# cite: https://github.com/lisa-lab/pylearn2/issues/677
for layer in network_layers:
    for i in range(len(layer.params)):
        V_t = layer.velocity[i]
        V_tp1 = update_velocity(V_t, T.grad(cost, layer.params[i]), LR)
        updates.append((layer.params[i],
                        layer.params[i] + (
                            MOMENTUM * V_tp1 - LR * T.grad(
                                cost, layer.params[i]))))
        updates.append((layer.velocity[i], V_tp1))

# Define functions to run our model
Train = theano.function(
    [imgs, target, LR],
    cost,
    updates=updates,
    allow_input_downcast=True
)
Test = theano.function([imgs], network_layers[-1].output)


# Loop through the epochs count
for i in range(EPOCHS):
    # Get a random pivot point and
    # extract a mini batch
    rand_index = np.random.randint(0, X_Train.shape[0] - BATCH_SIZE)
    # Calculate the cost and perform an update
    Cost = Train(X_Train[rand_index:rand_index + BATCH_SIZE, :],
                 Y_Train_onehot[rand_index:rand_index + BATCH_SIZE, :],
                 LEARNING_RATE
                 )
    if(np.isnan(Cost)):
        exit()
    # Print iteration stats
    print "Cost at iteration %i is equal to: %1.3f" % (i, Cost)

    # Notify when an epoch is complete
    if(i % (X_Train.shape[0] / BATCH_SIZE) == 0 and i > 0):
        print "An epoch is complete."

    # Every 50 iteration calculate accuracy
    if(i % 200 == 0 and i > 0):
        Test_Result = np.argmax(Test(X_Test.astype("float32")), axis=1)
        Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
            (Y_Test.shape[0])) * 100
        print "The model performed with an accuracy of: %.2f" % (
            float(Score)) + "%"
        print "Be advised the Learning rate is: %.4f" % (LEARNING_RATE)

    # Decay the learning rate
    LEARNING_RATE = LEARNING_RATE - (0.5 * 10**-4) * LEARNING_RATE


# Calculate accuracy one last time
Test_Result = np.argmax(Test(X_Test), axis=1)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model performed with an accuracy of: %.2f" % (
    float(Score)) + "%"
