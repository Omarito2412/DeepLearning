import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


# Draw the training data and the classifier
def plotClassifier(x, y, w):
    # Gather the two classes
    ones = x[np.where(y == 1)[0]]
    n_ones = x[np.where(y == -1)[0]]

    # Plot them
    plt.scatter(ones[:, 1], ones[:, 2], marker='o')
    plt.scatter(n_ones[:, 1], n_ones[:, 2], marker='x')

    # Get two points on the line
    x1 = [-5, 5]
    x2 = list()
    x2 = [-w[:, 1] / w[:, 2] * i - w[:, 0] / w[:, 2] for i in x1]
    plt.plot(x1, x2, 'k-')
    plt.show()

# Declare constants
LEARNING_RATE = 0.005  # Learning rate

# Declare symbols
y = T.dmatrix('Predictions')
t = T.dmatrix('Actual_Output')
# cost = T.dscalar('Cost')
x = T.dmatrix('Input')
w = theano.shared(value=np.zeros(
                  (1, 3),
                  dtype=theano.config.floatX),
                  name='Weights')

# Define the predictions
y = T.dot(x, w.transpose())

# Define the cost
cost = (T.sum(T.sub(y, t)**2) / (2 * y.shape[0]))

# Define the gradients
delta = T.grad(cost, w)

# Define gradient descent
updates = [(w, w - LEARNING_RATE * delta)]
train_model = theano.function([x, t], cost, updates=updates)

# Hypothesis
predict = theano.function([x], y)


# Load Data
inputs = np.load("x.npy")
results = np.load("t.npy").T

inputs_test = np.load("x_t.npy")
results_test = np.load("t_t.npy").T

# Train the model
for i in range(500):
    print train_model(inputs, results)
print "The computed weights are: ", w.eval()

# Solve the test set
predictions = predict(inputs_test)
predictions = np.piecewise(predictions,
                           [predictions > 0, predictions <= 0], [1, -1])
# Calculate the errors
errors = np.where((predictions - results_test) != 0)[0]

# Calculate Accuracy and print metric
print "The model performed with an accuracy of: %.2f" % ((1. - (float(
    len(errors))) / (inputs_test.shape[0])) * 100) + "%"
plotClassifier(inputs, results, w.eval())
