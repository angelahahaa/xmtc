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
from keras.callbacks import CSVLogger

from keras.layers import CuDNNLSTM, Bidirectional, LSTM, Dropout
from keras.layers import TimeDistributed, Lambda, Softmax, merge
import tensorflow as tf
import keras.backend as K

from sklearn.preprocessing import MultiLabelBinarizer

import scipy.sparse
from tools.helper import MetricsAtTopK, clean_str

# argparse
import argparse
parser = argparse.ArgumentParser(description='run xmlcnn')
parser.add_argument('-i','--input',required = True,help='input directory e.g. ./data/dl_amazon_1')
parser.add_argument('-o','--output',default = '',help='output directory for model e.g. ./xmlcnn/models/amazon_')
parser.add_argument('--log',default = 'dump.csv', help= 'log file in csv format e.g. ./log.csv')
parser.add_argument('--epoch',type=int,default=5,help='epochs')
args = parser.parse_args()

# things
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 50000
EMBEDDING_DIM = 300
IN_DIR = args.input
if not os.path.exists(IN_DIR):
    raise Exception('input path does not exist: {}'.format(IN_DIR))
with open('{}/tokenizer.pkl'.format(IN_DIR), 'rb') as f:
    tokenizer = pickle.load(f)
with open('{}/mlb.pkl'.format(IN_DIR), 'rb') as f:
    mlb = pickle.load(f)
embedding_matrix = np.load('{}/embedding_matrix.npy'.format(IN_DIR))
x_train = np.load('{}/x_train.npy'.format(IN_DIR))
x_test = np.load('{}/x_test.npy'.format(IN_DIR))
y_train = scipy.sparse.load_npz('{}/y_train.npz'.format(IN_DIR))
y_test = scipy.sparse.load_npz('{}/y_test.npz'.format(IN_DIR))
y_train = y_train.todense()
y_test = y_test.todense()

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
x = Dropout(0.5)(x)
x = Dense(labels_dim, activation = None)(x)

def loss_function(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true,y_pred,from_logits=True),axis=-1)
def binary_accuracy_with_logits(y_true, y_pred):
    return K.mean(K.equal(y_true, K.tf.cast(K.less(0.0,y_pred), y_true.dtype)))

model = Model(sequence_input, x)
pat1 = MetricsAtTopK(k=1)
pat5 = MetricsAtTopK(k=5)
def p1(x,y):
    return pat1.precision_at_k(x,y)
def p5(x,y):
    return pat5.precision_at_k(x,y)

model.compile(loss=loss_function,
              optimizer='adam',
              metrics=[binary_accuracy_with_logits,p1,p5])
print(model.summary())
csv_logger = CSVLogger(args.log,append=True)
model.fit(x_train, y_train,
          batch_size=128,
          epochs=args.epoch,
          validation_data=(x_test, y_test),
          callbacks = [csv_logger]
         )
# save things
if args.output:
    model.save_weights('{}.h5'.format(args.output))
