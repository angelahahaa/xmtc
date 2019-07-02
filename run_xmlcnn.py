import numpy as np
import os,re
import pandas as pd
import pickle

from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.metrics import categorical_accuracy, binary_accuracy

from sklearn.preprocessing import MultiLabelBinarizer

import scipy.sparse
from tools.helper import MetricsAtTopK, clean_str

# things
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 50000
EMBEDDING_DIM = 300
IN_DIR = './data/amazon_xmlcnn'

with open('{}/tokenizer.pkl'.format(IN_DIR), 'rb') as f:
    tokenizer = pickle.load(f)
with open('{}/mlb.pkl'.format(IN_DIR), 'rb') as f:
    mlb = pickle.load(f)
embedding_matrix = np.load('{}/embedding_matrix.npy'.format(IN_DIR))
x_train = np.load('{}/x_train.npy'.format(IN_DIR))
x_test = np.load('{}/x_test.npy'.format(IN_DIR))
y_train = scipy.sparse.load_npz('{}/y_train.npz'.format(IN_DIR))
y_test = scipy.sparse.load_npz('{}/y_test.npz'.format(IN_DIR))

labels_dim = len(mlb.classes_)
num_words = min(MAX_NUM_WORDS, len(tokenizer.word_index)) + 1

# set up embedding layer
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

filter_sizes = [2,4,8]
pooling_units = 32

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
convs = []
for fsz in filter_sizes:
    l = Conv1D(filters = 128, kernel_size = fsz, strides = 2, activation = 'relu')(embedded_sequences)
    s = int(l.shape[-2])
    pool_size = s//pooling_units
    l = MaxPooling1D(pool_size,padding = 'same')(l)
    l = Flatten()(l)
    convs.append(l)
x = Concatenate(axis=-1)(convs)
x = Dense(512, activation = 'relu')(x)
x = Dense(labels_dim, activation = 'sigmoid')(x)
model = Model(sequence_input, x)

pat1 = MetricsAtTopK(k=1)
pat5 = MetricsAtTopK(k=5)
def p1(x,y):
    return pat1.precision_at_k(x,y)
def p5(x,y):
    return pat5.precision_at_k(x,y)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[binary_accuracy,p1,p5])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test),
         )
