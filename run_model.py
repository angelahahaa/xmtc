#!/usr/bin/env python
# coding: utf-8

# # IMPORTS




# basic
import argparse
import os,datetime
# data input
import numpy as np
import scipy.sparse
# metric and loss function
import keras.backend as K
import tensorflow as tf
from keras.metrics import categorical_accuracy, binary_accuracy, top_k_categorical_accuracy
# embedding
from keras.initializers import Constant
from keras.layers import Embedding
# models
from keras.layers import Dense, Input, Flatten, Concatenate, Conv1D, MaxPooling1D, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, TimeDistributed, Lambda, Softmax
from keras.initializers import Constant
from keras.models import Model
# misc
from tools.MyClock import MyClock
# save things
import pandas as pd
from keras.callbacks import CSVLogger





def Coloured(string):
    return "\033[1;30;43m {} \033[0m".format(string)


# # ARGPARSE




parser = argparse.ArgumentParser(description = 'run baseline models')
parser.add_argument('-i','--input', required = True, type = str, help = 'input directory e.g. ./data/dl_amazon_1/')
parser.add_argument('-m','--model', required = True, type = str, help = 'model, one in: xmlcnn, attentionxml, attention,')
parser.add_argument('-l','--loss', default = 'categorical', type = str, help = "loss type, categorical or binary ")
parser.add_argument('-o','--output', default = 'outputs', type = str, help = 'output directory')
parser.add_argument('--epoch', default = 5, type = int, help = 'epochs')
parser.add_argument('--batch_size', default = 0, type = int, help = 'batch size')
parser.add_argument('--save_weights', default = True, action = 'store_true', help = 'save trained weights')
parser.add_argument('--save_model', default = True, action = 'store_true', help = 'save trained model architecture')
parser.add_argument('--save_prediction', default = True, action = 'store_true', help = 'save top 10 prediction and corresponding probabilities (not implemented yet)')
args = parser.parse_args()

# argparse validation
default_batch_size = {'xmlcnn':128,'attentionxml':20,'attention':25}
if not os.path.exists(args.input):
    raise Exception('Input path does not exist: {}'.format(args.input))
if not os.path.exists(args.output):
    os.mkdir(args.output)
    print(Coloured("Create Output Directory: {}".format(args.output)))
if not args.batch_size:
    args.batch_size = default_batch_size[args.model]
IN_DIR = args.input
OUT_DIR = os.path.join(
    args.output,
    datetime.datetime.now().strftime('%y%m%d_%H%M%S_{}'.format(args.model)),
)
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)


# # INPUT




def get_input(in_dir):
    x_train = np.load(os.path.join(in_dir,'x_train.npy'))
    dirs = [os.path.join(in_dir,d) for d in sorted(os.listdir(in_dir)) if d.startswith('y_train')]
    y_trains = [scipy.sparse.load_npz(d).todense() for d in dirs]

    x_test = np.load(os.path.join(in_dir,'x_test.npy'))
    dirs = [os.path.join(in_dir,d) for d in sorted(os.listdir(in_dir)) if d.startswith('y_test')]
    y_tests = [scipy.sparse.load_npz(d).todense() for d in dirs]
    return x_train,y_trains,x_test,y_tests


# # EMBEDDING LAYER




def get_embedding_layer():
    embedding_matrix = np.load(os.path.join(IN_DIR,'embedding_matrix.npy'))
    num_words, embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                trainable=False)
    return embedding_layer


# # MODELS




def apply_attention(inputs):
    input1, input2 = inputs
    outer_product = tf.einsum('ghj, ghk -> gjk', input1, input2)
    return outer_product

def get_model(model_name, max_sequence_length, labels_dims):
    embedding_layer = get_embedding_layer()
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if model_name == 'xmlcnn':
        filter_sizes = [2,4,8]
        pooling_units = 32
        bottle_neck = 512
        convs = []
        for fsz in filter_sizes:
            l = Conv1D(filters = 128, kernel_size = fsz, strides = 2, activation = 'relu')(embedded_sequences)
            s = int(l.shape[-2])
            pool_size = s//pooling_units
            l = MaxPooling1D(pool_size,padding = 'same')(l)
            l = Flatten()(l)
            convs.append(l)
        x = Concatenate(axis=-1)(convs)
        x = Dense(bottle_neck, activation = 'relu')(x)
        x = Dropout(0.5)(x)
        outs = []
        for i,labels_dim in enumerate(labels_dims):
            outs.append(Dense(labels_dim, activation = None, name = 'H{}'.format(i))(x))
    elif model_name in ['attentionxml','attention']:
        labels_dim = sum(labels_dims)
        if model_name == 'attentionxml':
            # with lstm
            x = Bidirectional(CuDNNLSTM(512, return_sequences=True))(embedded_sequences)
        else:
            # without lstm
            x = embedded_sequences
        attention = Dense(labels_dim,activation=None,name='attention_dense')(x)
        attention = Softmax(axis=1,name='attention_softmax')(attention)
        x = Lambda(apply_attention,name = 'apply_attention')([x, attention])
        x = Lambda(lambda x:K.permute_dimensions(x,(0,2,1)),name='transpose')(x)
        x = TimeDistributed(Dense(512,activation='relu'))(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(256,activation='relu'))(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(1,activation=None))(x)
        x = Lambda(lambda x:K.squeeze(x,axis=-1))(x)
        outs = []
        start = 0
        for i,labels_dim in enumerate(labels_dims):
            outs.append(Lambda(lambda x:x[:,start:start+labels_dim],name = 'H{}'.format(i))(x))
            start+=labels_dim
    else:
        raise Exception('Invalid model_name : {}'.format(model_name))
    return Model(sequence_input, outs)


# # LOSSES




def binary_cross_entropy_with_logits(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true,y_pred,from_logits=True),axis=-1)
def categorical_cross_entropy_with_logits(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true,y_pred,from_logits=True),axis=-1)
loss_dict = {'binary':binary_cross_entropy_with_logits,
             'categorical':categorical_cross_entropy_with_logits,}


# # METRICS




def binary_accuracy_with_logits(y_true, y_pred):
    return K.mean(K.equal(y_true, K.tf.cast(K.less(0.0,y_pred), y_true.dtype)))
def pAt1(y_true,y_pred):
    return categorical_accuracy(y_true, y_pred)
def pAt5(y_true,y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# # MISC




def save_predictions(model,x_test,y_tests,out_dir):
    print('SAVE PREDICTIONS TO : {}'.format(out_dir))
    out_probs = model.predict(x_test,verbose=1)
    ind_dirs = [os.path.join(out_dir,'pred_outputs{}.txt'.format(i)) for i in range(len(y_tests))]
    log_dirs = [os.path.join(out_dir,'pred_logits{}.txt'.format(i)) for i in range(len(y_tests))]
    f_ind = [open(ind_dir,'ab') for ind_dir in ind_dirs]
    f_log = [open(log_dir,'ab') for log_dir in log_dirs]
    for i,out_prob in enumerate(out_probs):
        ind = np.argsort(out_prob,axis=1)[:,:-11:-1]
        logits = np.take_along_axis(out_prob, ind, axis=1)
        np.savetxt(f_ind[i],ind,fmt='%d')
        np.savetxt(f_log[i],logits,fmt='%1.3f')
    for f in f_ind + f_log:
        f.close()


# # MAIN




csv_logger = CSVLogger(os.path.join(OUT_DIR,'train.log'),append=False)
# inputs
x_train,y_trains,x_test,y_tests = get_input(IN_DIR)
_,max_sequence_length = x_train.shape
labels_dims = [l.shape[-1] for l in y_tests]
model = get_model(args.model, max_sequence_length, labels_dims)
model.compile(loss=loss_dict[args.loss],
              optimizer='adam',
              metrics=[pAt1,pAt5])
model.fit(x_train, y_trains,
          batch_size = args.batch_size,
          epochs = args.epoch,
          validation_data = (x_test, y_tests),
          callbacks = [csv_logger],
          shuffle=True,
         )


# # save things




if args.save_weights:
    model.save_weights(os.path.join(OUT_DIR,'weights.h5'))
if args.save_model:
    with open(os.path.join(OUT_DIR,'model.json'),'w') as f:
        f.write(model.to_json())
if args.save_prediction:
    save_predictions(model,x_test,y_tests,OUT_DIR)
pd.DataFrame.from_dict([vars(args)]).to_csv(os.path.join(OUT_DIR,'args.csv'))


