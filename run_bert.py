# basic
import os,datetime

# save things
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import scipy.sparse

# get input
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
    return x_train,y_trains,x_test,y_tests

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
    x_tests = [test_sequence,test_mask,test_segment]
    _,y_trains,_,y_tests = get_input(in_dir, mode, get_output=[0,1,0,1])
    return x_trains, y_trains, x_tests, y_tests

# build model
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

# build bert layer (original from https://github.com/strongio/keras-bert/blob/master/keras-bert.ipynb)
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

# Build model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

IN_DIR = 'data/sic_hierarchy'
x_trains, y_trains, x_tests, y_tests = get_bert_input(IN_DIR,'cat')
max_sequence_length = len(x_trains[0][0])
labels_dims = [len(y[0]) for y in y_tests]

bert_inputs = [
    Input(shape=(max_sequence_length,), name="input_sequence"),
    Input(shape=(max_sequence_length,), name="input_mask"),
    Input(shape=(max_sequence_length,), name="input_segment"),
]
bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)
# dense = Dense(256, activation='relu')(bert_output)
# dense = Dropout(0.5)(dense)
outs = []
for i,labels_dim in enumerate(labels_dims):
    outs.append(Dense(labels_dim, activation = None, name = 'H{}'.format(i))(bert_output))
def myloss(y_true,y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=True)
with tf.Session() as sess:
    initialize_vars(sess)
    model = Model(inputs=bert_inputs, outputs=outs)
    model.summary()
    model.compile(loss = myloss,
              optimizer = 'adam',
              metrics = ['acc'])
    model.fit(x_trains, y_trains,
          batch_size = 256,
          epochs = 1,
          validation_data = (x_tests, y_tests),
          shuffle = True,
         )
