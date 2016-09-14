from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense

# Take the first n most common words
nb_words = 25000
# Maximum length of a sequence (Timesteps)
# Maximum length in X_train is 2494, it must
# be cut.
sequence_length = 100

# Length of each word vector
vector_length = 128

# The number of examples to train on simultaneously
batch_size = 32

# The number of epochs
epochs = 20

# Load data
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(nb_words=nb_words)
# Our data is now a matrix, each vector is a sequence of integers.
# Each integer is a word index in the index table
# You can test that out by trying index["movie"]
index = imdb.get_word_index()

# Pad each sequence, since a neural network can't deal with
# variable length input, we need to normalize to a max length
# it can be a hyperparameter anyway
# Short sequences are padded with zeros
# Convolution with mode = "Same" has the same effect
train_padded = pad_sequences(X_train, maxlen=sequence_length)
test_padded = pad_sequences(X_test, maxlen=sequence_length)

# Sequential model is a type of model which we stack
# layers on top of each other sequentially.
# Example: Convolution => ReluLayer => Dropout => Pool => Dense => Softmax
# Docs: https://keras.io/models/sequential/
model = Sequential()

# We add an embedding layer, so each integer in sequence is
# mapped to a word vector
# Example: 17 => [0, 0.1, 0.4, -1.2, 0.2]
# In fact this is only a mapping of a word to a learnable word vector
# In this example we mapped the word "Movie" to that vector
# Docs: https://keras.io/layers/embeddings/
model.add(Embedding(nb_words, vector_length, input_length=sequence_length))

# Add a layer of our beloved LSTM with some dropout
# for regularization
# Docs: https://keras.io/layers/recurrent/
model.add(LSTM(128, dropout_W=0.4, dropout_U=0.2))

# Add a dense layer with sigmoid activation
# This is to map our output to a probability
# Docs: https://keras.io/layers/core/
model.add(Dense(1, activation='sigmoid'))

# Compile the model, calculate loss and choose optimizer algorithm
model.compile(loss='binary_crossentropy', optimizer='adagrad',
              metrics=['accuracy'])

# Start training
model.fit(train_padded, Y_train, batch_size=batch_size, nb_epoch=epochs)

loss, accuracy = model.evaluate(test_padded, Y_test, batch_size=batch_size)
print "The calculated loss is: " + str(loss)
print "The calculated accuracy is: " + str(accuracy)
