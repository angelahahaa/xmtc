# basic
import argparse
import os,datetime

# data input
import pandas as pd
import numpy as np
import scipy.sparse
import pickle
# metric and loss function
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy, top_k_categorical_accuracy
# embedding
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding
# models
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import CuDNNLSTM, Bidirectional, TimeDistributed, Lambda, Softmax
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model


def Coloured(string):
    return "\033[1;30;43m {} \033[0m".format(string)

# # INPUT


def get_input(in_dir, mode, sparse = False, get_output = [True]*4):
    # get_output: bool of get this var ['x_train','y_trains','x_test','y_tests']
    x_train,y_trains,x_test,y_tests = [None]*4
    if get_output[0]:
        x_train = np.load(os.path.join(in_dir,'x_train.npy'))
    if get_output[1]:
        dirs = [os.path.join(in_dir,d) for d in sorted(os.listdir(in_dir)) if d.startswith('y_train_{}'.format(mode))]
        y_trains = [scipy.sparse.load_npz(d) for d in dirs]
        if not sparse:
            y_trains = [y.toarray() for y in y_trains]
    if get_output[2]:
        x_test = np.load(os.path.join(in_dir,'x_test.npy'))
    if get_output[3]:
        dirs = [os.path.join(in_dir,d) for d in sorted(os.listdir(in_dir)) if d.startswith('y_test_{}'.format(mode))]
        y_tests = [scipy.sparse.load_npz(d) for d in dirs]
        if not sparse:
            y_tests = [y.toarray() for y in y_tests]
    return [x_train],y_trains,[x_test],y_tests

def get_bert_input(in_dir,mode):
    df = pd.read_pickle(os.path.join(in_dir,'bert_x.pkl'))
    train_df = df[df['train/test']=='train']
    test_df = df[df['train/test']=='test']
    train_sequence = train_df['sequence'].to_list()
    train_mask = train_df['mask'].to_list()
    train_segment = [[0]*len(train_mask[0]) ]*len(train_mask)
    test_sequence = test_df['sequence'].to_list()
    test_mask = test_df['mask'].to_list()
    test_segment = [[0]*len(test_mask[0]) ]*len(test_mask)
    x_trains = [train_sequence,train_mask,train_segment]
    x_trains = [np.array(x) for x in x_trains]
    x_tests = [test_sequence,test_mask,test_segment]
    x_tests = [np.array(x) for x in x_tests]
    _,y_trains,_,y_tests = get_input(in_dir, mode, get_output=[0,1,0,1])
    return x_trains, y_trains, x_tests, y_tests

def mask_ys(y_trues,in_dir):
    # slow an stupid way to mask non important values to -1
    ds = pickle.load(open(os.path.join(in_dir,'child_to_siblings.pkl'),'rb'))
    outs = [y_trues[0]]
    for i,y in enumerate(y_trues):
        if i==0:
            continue
        d = ds[i]
        y_true = y.argmax(axis=1).flatten()
        out = -np.ones(shape=y.shape)
        for j,yt in enumerate(y_true):
            out[j,d[yt]]=0
            out[j,yt]=1
        outs.append(out)
    return outs

def get_unbiased_train_val_split(x_trains,y_trains,in_dir,print_progress = True):
    train_inds_dir = os.path.join(in_dir,'train_inds.npy')
    val_inds_dir = os.path.join(in_dir,'val_inds.npy')

    if not os.path.exists(train_inds_dir) or not os.path.exists(val_inds_dir):
        print('SPLIT BY CLASS')
        val_split=0.2
        sparse_ys = [np.argmax(y_train,axis=1) for y_train in y_trains]
        yt = np.column_stack(sparse_ys)
        unique_h,unique_inverse = np.unique(yt,return_inverse=True,axis=0)
        too_small = []
        train_inds = np.array([])
        val_inds = np.array([])
        for s in range(len(unique_h)):
            inds = np.argwhere(unique_inverse==s)
            if len(inds)<=1:
                too_small.append(inds[0])
                train_inds = np.append(train_inds,inds)
                val_inds = np.append(val_inds,inds)
            else:
                split = int(len(inds)*val_split)
                np.random.seed(s)
                np.random.shuffle(inds)
                train_inds = np.append(train_inds,inds[max(1,split):])
                val_inds = np.append(val_inds,inds[:max(1,split)])
            if print_progress and s%(len(unique_h)//10)==0:
                print("{:.0f}%".format((s+1)/len(unique_h)*100),end='\r')
        print("Duplicated inds: {}".format(len(too_small)))
        train_inds = train_inds.astype(int)
        val_inds = val_inds.astype(int)
        np.save(train_inds_dir,train_inds)
        np.save(val_inds_dir,val_inds)
    else:
        print('LOAD EXISTING VAL INDS')
        train_inds = np.load(train_inds_dir)
        val_inds = np.load(val_inds_dir)

    x_vs = [x[val_inds,:] for x in x_trains]
    y_vs = [y[val_inds,:] for y in y_trains]
    x_ts = [x[train_inds,:] for x in x_trains]
    y_ts = [y[train_inds,:] for y in y_trains]

    return x_ts,y_ts,x_vs,y_vs

# # EMBEDDING LAYER




def get_embedding_layer(in_dir):
    embedding_matrix = np.load(os.path.join(in_dir,'embedding_matrix.npy'))
    num_words, embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                trainable=False)
    return embedding_layer

## BERT LAYER

class BertLayer(Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_dim = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# # MODELS




def apply_attention(inputs):
    input1, input2 = inputs
    outer_product = tf.einsum('ghj, ghk -> gjk', input1, input2)
    return outer_product

def get_model(model_name, max_sequence_length, labels_dims, embedding_layer):
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

def get_bert_model(max_sequence_length, labels_dims, bottle_neck, trainable_layers, sess):
    bert_inputs = [
        Input(shape=(max_sequence_length,), name="input_sequence"),
        Input(shape=(max_sequence_length,), name="input_mask"),
        Input(shape=(max_sequence_length,), name="input_segment"),
    ]
    x = BertLayer(n_fine_tune_layers = trainable_layers, pooling="first")(bert_inputs)
    if bottle_neck:
        x = Dense(bottle_neck, activation='relu')(x)
        x = Dropout(0.5)(x)
    outs = []
    for i,labels_dim in enumerate(labels_dims):
        outs.append(Dense(labels_dim, activation = None, name = 'H{}'.format(i))(x))
    initialize_vars(sess)
    return Model(inputs=bert_inputs, outputs=outs)
# # LOSSES




def binary_cross_entropy_with_logits(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true,y_pred,from_logits=True),axis=-1)
def categorical_cross_entropy_with_logits(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true,y_pred,from_logits=True),axis=-1)
def masked_categorical_cross_entropy_with_logits(y_true, y_pred):
    y_pred = tf.where(K.not_equal(y_true, -1), y_pred, -1e7*tf.ones_like(y_pred))
    loss = K.categorical_crossentropy(tf.maximum(y_true,0.) ,y_pred, from_logits=True)
    return K.mean(loss,axis=-1)


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
    if len(y_tests)==1:
        out_probs = [out_probs]
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
def save_hs_predictions(model,x_test,y_tests,out_dir,IN_DIR):
    print('SAVE HS PREDICTIONS TO : {}'.format(out_dir))
    out_logits = model.predict(x_test,verbose=1)
    out_probs = get_cascade_sm(out_logits,IN_DIR)
    if len(y_tests)==1:
        out_probs = [out_probs]
    ind_dirs = [os.path.join(out_dir,'pred_outputs{}.txt'.format(i)) for i in range(len(y_tests))]
    prob_dirs = [os.path.join(out_dir,'pred_probs{}.txt'.format(i)) for i in range(len(y_tests))]
    for i,out_prob in enumerate(out_probs):
        ind = np.argsort(out_prob,axis=1)[:,:-11:-1]
        probs = np.take_along_axis(out_prob, ind, axis=1)
        np.savetxt(ind_dirs[i],ind,fmt='%d')
        np.savetxt(prob_dirs[i],probs,fmt='%1.3f')

# function
def get_cascade_sm(y_preds,IN_DIR):
    child_to_siblings = pickle.load(open(os.path.join(IN_DIR,'child_to_siblings.pkl'),'rb'))
    parent_to_child = pickle.load(open(os.path.join(IN_DIR,'parent_to_child.pkl'),'rb'))
    cascade_sm = [softmax(y_preds[0],axis=1)]
    for i in range(len(y_preds)-1):
        data = []
        row_ind = []
        col_ind = []
        sms = cascade_sm[-1]
        for key,val in parent_to_child[i].items():
            child_sm = softmax(y_preds[i+1][:,val],axis=1)
            data.append(np.multiply(child_sm,sms[:,key,np.newaxis]).reshape(-1))
            row_ind.append(np.repeat(np.arange(sms.shape[0]),len(val)))
            col_ind.append(np.tile(val,sms.shape[0]))
        data = np.concatenate(data)
        row_ind = np.concatenate(row_ind)
        col_ind = np.concatenate(col_ind)
        cascade_sm.append(sp.csr_matrix((data,(row_ind,col_ind))).toarray())
    return cascade_sm
