"""Apply PCA whitening to a dataset.

This script is based on:
http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
The data is assumed to be a matrix of unrolled images
the dimensions should be Number of images * Number of pixels per image
"""
import numpy as np
from six.moves import cPickle
import argparse

# Regularization param
epsilon = 10**-5
ZCA_WHITENING = True

# Initialize Argument Parser
parser = argparse.ArgumentParser(
    description='Whiten the given input')

parser.add_argument('input',
                    help='The input pickle')
input = parser.parse_args().input

print "Reading input"
with open(input, "rb") as source:
    data = cPickle.load(source)
print "Normalizing data"
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

print "Calculating Covariance Matrix"
covariance_matrix = np.cov(data.T)
print "Computing Eigen vectors and values"
U = np.linalg.eig(covariance_matrix)
print "Rotating data"
data_rot = np.dot(data, U[1])
print "Dividing by SQRT of Eigen values"
data_rot = np.divide(data_rot, np.sqrt(U[0] + epsilon))

# Perform PCA or ZCA whitening
if(ZCA_WHITENING):
    print "Performing ZCA whitening"
    data_rot = np.dot(U, data_rot)

print "Saving file..."
with open("output.pickle", "wb") as output:
    cPickle.dump(data_rot, output, cPickle.HIGHEST_PROTOCOL)
print "Done, all is well."
