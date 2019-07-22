from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.layers import CuDNNLSTM, Bidirectional, LSTM, Dropout
from keras.layers import TimeDistributed, Lambda, Softmax, merge
from keras.initializers import Constant
from keras.layers import Input, Embedding
import tensorflow as tf
import keras.backend as K


def apply_attention(inputs):
    input1, input2 = inputs
    outer_product = tf.einsum('ghj, ghk -> gjk', input1, input2)
    return outer_product

def get_model(model_name,embedding_matrix,max_sequence_length,labels_dim):
    num_words,embedding_dim = embedding_matrix.shape
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_sequence_length,
                                trainable=False)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    if model_name == 'xmlcnn':
        filter_sizes = [2,4,8]
        pooling_units = 32
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
    elif model_name in ['attentionxml','attention']:
        if model_name == 'attentionxml':
            x = Bidirectional(CuDNNLSTM(512, return_sequences=True))(embedded_sequences)
        else:
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
    else:
        raise Exception('Invalid model_name : {}'.format(model_name))
    return Model(sequence_input, x)
