"""An attempt to reproduce the results using K-means."""

import numpy as np
from six.moves import cPickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import sklearn.preprocessing as pre

NUM_FEATURES = 500
NUM_PATCHES = 500000
PATCH_SIZE = (7, 7, 1)
WHITENING = True
IMAGE_DIMS = (28, 28, 1)


def normalize(dataset):
    """Brightness and Contrast Normalization."""
    dataset_mean = dataset.mean(axis=1)
    dataset_std = dataset.std(axis=1)
    dataset -= dataset_mean.reshape((-1, 1)).repeat(dataset.shape[1], axis=1)
    dataset /= (dataset_std.reshape((-1, 1)).repeat(
                dataset.shape[1], axis=1) + 10)
    return dataset


def whiten(dataset):
    """Perform ZCA whitening."""
    dataset = dataset.astype("float32")
    m = dataset.mean(axis=0)
    dataset -= m
    cov = np.cov(dataset.T) / dataset.shape[0]
    U, S, V = np.linalg.svd(cov)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S + 10**-5))), U.T)
    dataset = np.dot(dataset, components.T)
    return dataset, m, components


def extract_features(data, trainclusters, m=None, p=None):
    """Extract patches from data and pool."""
    print "Extracting features"
    num_ps = data.shape[1] / np.prod(PATCH_SIZE)
    xc = np.zeros((data.shape[0], num_ps * NUM_FEATURES))
    for i in range(data.shape[0]):
        patches = list()
        for j in range(0, np.prod(IMAGE_DIMS), np.prod(PATCH_SIZE)):
            patch = data[i, j:j + np.prod(PATCH_SIZE)]
            if(WHITENING):
                patch -= m
                patch = np.dot(patch, p.T)
            patch = trainclusters.transform(patch.reshape((1, -1)))
            mean = patch.mean(axis=1)
            activation = np.maximum(0, mean - patch)
            patches.append(activation)
        patches = np.array(patches)
        patches = patches.reshape((1, -1))
        patches = normalize(patches)
        xc[i] = patches
    return xc


# Read CIFAR-10
print "Reading Data and normalizing"
mnist = cPickle.load(open("train.pickle", "rb"))
scaler = pre.StandardScaler(copy=False)
# Load the training data
X_Train = mnist[:, 1:].astype("float32")[0:100]

Y_Train = mnist[:, 0][0:100]
M = None
P = None

# Extract random patches
print "Extracting random patches"
patches = list()
for i in range(NUM_PATCHES):
    r = np.random.randint(0, IMAGE_DIMS[0] - PATCH_SIZE[0] + 1, 1)[0]
    c = np.random.randint(0, IMAGE_DIMS[1] - PATCH_SIZE[0] + 1, 1)[0]
    patch = X_Train[i % X_Train.shape[0], :].reshape(IMAGE_DIMS)
    patch = patch[r:r + PATCH_SIZE[0], c:c + PATCH_SIZE[1], :]
    patches.append(patch.ravel())

patches = np.array(patches).astype("float32")
print "Normalizing patches"
patches = normalize(patches)

if(WHITENING):
    print "Whitening patches"
    patches, M, P = whiten(patches)

print "Training K-means"
# BEGIN FEATURE EXTRACTION #
TrainClusters = MiniBatchKMeans(n_clusters=NUM_FEATURES, max_iter=1000,
                                batch_size=25000)
TrainClusters.fit(patches)
del patches

# Extract features
Distances = extract_features(X_Train, TrainClusters, M, P)
Distances = np.concatenate((Distances, np.ones(
    (Distances.shape[0], 1))), axis=1)
# Standardize
Distances = scaler.fit_transform(Distances)

# BEGIN CLASSIFIER TRAINING #
print "Training the SVM"
clf = SVC(kernel="linear", max_iter=50, C=50)
clf.fit(Distances, Y_Train)

print "Testing scores"
# Load test data
test = cPickle.load(open("test.pickle", "rb"))
X_Test = test[:, 1:].astype("float32")

Y_Test = test[:, 0]

# Extract features
Distances = extract_features(X_Test, TrainClusters, M, P)
Distances = np.concatenate((Distances, np.ones(
    (Distances.shape[0], 1))), axis=1)
# Standardize
Distances = scaler.fit_transform(Distances)

Test_Result = clf.predict(Distances)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model classified with an accuracy of: %.2f" % (
    float(Score)) + "%"
