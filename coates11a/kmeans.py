"""An attempt to reproduce the results using K-means."""

import numpy as np
from six.moves import cPickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import skimage.util.shape as utl
import sklearn.decomposition as DCM
import sklearn.preprocessing as pre

NUM_FEATURES = 75
NUM_PATCHES = 400000
PATCH_SIZE = (8, 8, 1)
WHITENING = False
IMAGE_DIMS = (28, 28, 1)


def get_distances(x, trainclusters):
    """Retrieve the triangular K-means mapping."""
    z = trainclusters.transform(x)
    meanz = z.mean(axis=1)
    meanz = meanz.reshape((-1, 1)).repeat(NUM_FEATURES, axis=1)
    distances = meanz - z
    distances = np.maximum(0, distances)
    return distances


def get_patches(images):
    """Transform each image into n patches."""
    images = images.reshape((-1, IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]))
    images = utl.view_as_windows(images,
                                (images.shape[0],
                                PATCH_SIZE[0],
                                PATCH_SIZE[1],
                                PATCH_SIZE[2]
                                ),
                                step=4
                                )
    images = images.reshape((images.shape[1] * images.shape[2] * images.shape[4],
                            np.prod(PATCH_SIZE)
                            ))
    print "Done transforming images"
    return images

# Read CIFAR-10
print "Reading Data and normalizing"
mnist = cPickle.load(open("train.pickle", "rb"))

# Load the training data
X_Train = mnist[:, 1:].astype("float32")

Y_Train = mnist[:, 0]

# Whiten input
if(WHITENING):
    print "Whitening data"
    pca = DCM.PCA(whiten=True, copy=False)
    pca.fit(X_Train)
    X_Train = pca.transform(X_Train)
    del pca

# Extract random patches
print "Extracting random patches"
patches = list()
for i in range(NUM_PATCHES):
    r = np.random.randint(0, IMAGE_DIMS[0] - PATCH_SIZE[0] + 1, 1)[0]
    c = np.random.randint(0, IMAGE_DIMS[1] - PATCH_SIZE[0] + 1, 1)[0]
    patch = X_Train[i % X_Train.shape[0], :].reshape((IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]))
    patch = patch[r:r + PATCH_SIZE[0], c:c + PATCH_SIZE[1], :]
    patches.append(patch.ravel())
    del patch

patches = np.array(patches).astype("float32")
# scaler = pre.StandardScaler(copy=False)
# patches = scaler.fit_transform(patches)

# Normalize
# X_Train = scaler.fit_transform(X_Train)

print "Training K-means"
# BEGIN FEATURE EXTRACTION #
TrainClusters = MiniBatchKMeans(n_clusters=NUM_FEATURES, max_iter=1000,
                                batch_size=10000)
TrainClusters.fit(patches)

Distances = get_distances(get_patches(
    X_Train.reshape((-1, IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]))), TrainClusters).reshape(
    (X_Train.shape[0], -1))

# Distances = scaler.fit_transform(Distances)

# BEGIN CLASSIFIER TRAINING #
print "Training the SVM"
clf = SVC(kernel="linear", max_iter=1, C=100)
clf.fit(Distances, Y_Train)

print "Testing scores"

# Load test data
test = cPickle.load(open("test.pickle", "rb"))
X_Test = test[:, 1:].astype("float32")

Y_Test = test[:, 0]
del test
if(WHITENING):
    pca = DCM.PCA(whiten=True, copy=False)
    pca.fit(X_Test)
    X_Test = pca.transform(X_Test)

# Normalize
# X_Test = scaler.fit_transform(X_Test)

Distances = get_distances(get_patches(
    X_Test.reshape(
        (-1, IMAGE_DIMS[0], IMAGE_DIMS[1], IMAGE_DIMS[2]))), TrainClusters).reshape(X_Test.shape[0], -1)
# Distances = scaler.fit_transform(Distances)
Test_Result = clf.predict(Distances)
Score = float(len(np.where(Test_Result == Y_Test)[0])) / float(
    (Y_Test.shape[0])) * 100
print "The model performed with an accuracy of: %.2f" % (float(Score)) + "%"
