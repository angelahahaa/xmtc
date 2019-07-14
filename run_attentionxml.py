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
parser = argparse.ArgumentParser(description='run attentionxml')
parser.add_argument('-i','--input',default = './data/dl_amazon_1', help='input directory e.g. ./data/dl_amazon_1')
parser.add_argument('-o','--output',default = '',help='output directory for model e.g. ./xmlcnn/models/amazon_')
parser.add_argument('--log',default = 'dump.csv', help= 'log file in csv format e.g. ./log.csv')
parser.add_argument('--epoch',type=int,default=5,help='epochs')
parser.add_argument('--batch_size',type=int,default=20,help='batch size')
parser.add_argument('--max_sequence_length',type=int,default = 500)
parser.add_argument('--max_num_words',type=int,default = 50000)
args = parser.parse_args()

# things
MAX_SEQUENCE_LENGTH = args.max_sequence_length
MAX_NUM_WORDS = args.max_num_words
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


L = labels_dim
def apply_attention(inputs):
    input1, input2 = inputs
    outer_product = tf.einsum('ghj, ghk -> gjk', input1, input2)
    return outer_product
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# lstm
x = Bidirectional(CuDNNLSTM(512, return_sequences=True))(embedded_sequences) # [h> h<]
# x = TimeDistributed(Dense(64,activation=None))(x) # not in original paper
attention = Dense(L,activation=None,name='attention_dense')(x)
attention = Softmax(axis=1,name='attention_softmax')(attention)
x = Lambda(apply_attention,name = 'apply_attention')([x, attention])
x = Lambda(lambda x:K.permute_dimensions(x,(0,2,1)),name='transpose')(x)
x = TimeDistributed(Dense(512,activation='relu'))(x)
x = Dropout(0.5)(x)
x = TimeDistributed(Dense(256,activation='relu'))(x)
x = Dropout(0.5)(x)
x = TimeDistributed(Dense(1,activation=None))(x)
x = Lambda(lambda x:K.squeeze(x,axis=-1))(x)

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
model.fit(x_train[:,:MAX_SEQUENCE_LENGTH], y_train,
          batch_size=args.batch_size,
          epochs=args.epoch,
          validation_data=(x_test[:,:MAX_SEQUENCE_LENGTH], y_test),
          callbacks = [csv_logger],
          shuffle=True,
         )
# save things
if args.output:
    model.save_weights('{}.h5'.format(args.output))
